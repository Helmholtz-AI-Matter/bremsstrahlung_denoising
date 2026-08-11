from pathlib import Path

import bremsstrahlung_denoising.colorscheme as rofl
import bremsstrahlung_denoising.mmmUtils as mu
import matplotlib.pyplot as plt
import numpy as np
import torch
from bremsstrahlung_denoising.dataset import quantile_normalization
from bremsstrahlung_denoising.model import Denoiser, NoiseRemover


def load_model(model_path, model_type="denoiser", device="cpu"):
    train_dir = Path(model_path)
    ckpt_files = sorted(train_dir.rglob("**/*.ckpt"))
    ckpt_file = ckpt_files[0]
    print("Loading file: ", ckpt_file)
    if model_type == "denoiser":
        model = Denoiser.load_from_checkpoint(ckpt_file)
    else:
        model = NoiseRemover.load_from_checkpoint(ckpt_file)

    print("Model loaded, ", ckpt_file)
    print(
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1.0e6,
        "mio params",
    )
    model.eval()
    model.freeze()
    # accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    return model


def patchwise_prediction2(
    noisy_input,
    model,
    patch_size=128,
    stride=128,
    normalization="tile",
    qhigh=0.9995,
    offset=[0, 0],
):
    assert normalization in ["tile", "image", "none"]
    assert noisy_input.ndim == 3
    assert noisy_input.shape[0] == 1

    # input: 1 x H x W
    n_rows = noisy_input.shape[1] // patch_size
    n_cols = noisy_input.shape[2] // patch_size
    if offset[0] != 0:
        n_rows = n_rows - 1
    if offset[1] != 0:
        n_cols = n_cols - 1

    patch_size = (patch_size,) * 2
    stride = (stride,) * 2

    assert normalization == "tile"

    prediction = torch.zeros_like(noisy_input)
    i = 0
    with torch.no_grad():
        for ii in range(n_rows):
            row_start = ii * stride[0] + offset[0]
            row_end = row_start + patch_size[0]
            for jj in range(n_cols):
                mu.update_progress(i, n_rows * n_cols)
                i += 1
                col_start = jj * stride[1] + offset[1]
                col_end = col_start + patch_size[1]
                # print(f"ii={ii}, jj={jj}, [:,:,{row_start}:{row_end}, {col_start}:{col_end}]")
                # H x W
                patch = (
                    noisy_input[0, row_start:row_end, col_start:col_end]
                    .detach()
                    .clone()
                )
                # NOTE: we now normalize the patch, as we have done during training.
                # This assumes that no prior normalization was applied!
                if normalization == "tile":
                    patch_norm, qlow_tile, qhigh_tile = quantile_normalization(
                        patch, quantile_high=qhigh
                    )
                    patch = torch.tensor(patch_norm)
                    # print("Normalizing on tile level before making prediction")

                # append batch and channel dimension and discard it afterwards
                patch_pred, _ = model(patch.unsqueeze(0).unsqueeze(0))
                patch_pred = patch_pred[0, 0]
                # NOTE: prediction is now based on normalized image and should be compared to normalized
                # signal (by tile)
                # However, we now might have the problem that we lose information on the scales between neighboring
                # tiles, as each tile is normalized on its own, so we'll have to renormalize
                if normalization == "tile":
                    patch_pred = qlow_tile + (qhigh_tile - qlow_tile) * patch_pred

                prediction[0, row_start:row_end, col_start:col_end] = (
                    patch_pred.detach().cpu()
                )
    return prediction


def get_boundary_weights(patch_size=128, stride=128, offset=[0, 0], dims=(1024, 768)):
    n_rows = dims[0] // patch_size
    n_cols = dims[1] // patch_size
    if offset[0] != 0:
        n_rows = n_rows - 1
    if offset[1] != 0:
        n_cols = n_cols - 1

    edges = np.zeros(dims)
    with torch.no_grad():
        for ii in range(n_rows):
            row_start = ii * stride + offset[0]
            for jj in range(n_cols):
                col_start = jj * stride + offset[1]
                for x in np.arange(patch_size):
                    for y in np.arange(patch_size):
                        dist = [x, y, patch_size - x, patch_size - y]
                        smallestdist = np.min(dist) + 1
                        edges[row_start + x, col_start + y] = smallestdist
    return edges


def doit(noisy_signal, model):
    dims = np.shape(noisy_signal)
    noisy_signal_test_raw_torch = torch.tensor(noisy_signal[np.newaxis, np.newaxis])
    print("Noisy signal range", noisy_signal.min(), noisy_signal.max())

    # Predictions when normalizing each tile and undoing the normalization after the prediction
    offsets = [[0, 0], [96, 32], [32, 96], [64, 64]]
    preds = np.zeros((np.shape(offsets)[0], dims[0], dims[1]))
    for oi, off in enumerate(offsets):
        print("Doing tiling offset #{:}: {:}".format(oi, off))
        pred = patchwise_prediction2(
            noisy_signal_test_raw_torch[0], model, normalization="tile", offset=off
        )
        preds[oi] = pred

    mean = np.mean(preds, 0)
    bigmean = preds * 0.0
    for oi, off in enumerate(offsets):
        bigmean[oi, :, :] = mean
    offdiff = np.abs(preds - bigmean)

    # The advanced weighting system   ***********
    reloffdiff = offdiff / mean
    weights = 1 / (reloffdiff + 2)
    weights = mu.normalize(weights)
    for oi, off in enumerate(offsets):
        bw = get_boundary_weights(offset=off, dims=dims)
        bw[bw > 30] = 30
        bw = mu.normalize(bw)

        weights[oi, :, :] = weights[oi, :, :] * bw

    # calculating the weighted mean:
    prediction = np.sum(weights * preds, 0) / np.sum(weights, 0)

    return prediction


def draw_it(
    noisy_signal,
    prediction,
    runNo,
    case,
    fn,
    draw_output,
    ax,
    noise_cmax,
    out_fig_dir,
    cl=None,
):
    pure_noise = noisy_signal - prediction

    sm = np.max(noisy_signal)
    if cl is None:
        cl = np.array([1e-5 * sm, sm])

    mu.figure(18, 10)
    plt.subplot(131)
    plt.imshow(noisy_signal, cmap=rofl.cmap())
    plt.colorbar()
    plt.clim(cl)
    plt.axis(ax)
    plt.axis("off")
    plt.title("Raw data for #{:04.0f}".format(runNo))

    plt.subplot(132)
    plt.imshow(prediction, cmap=rofl.cmap())
    plt.colorbar()
    plt.clim(cl)
    plt.axis("off")
    plt.title("Prediction #{:04.0f}, case {:}".format(runNo, case))
    plt.axis(ax)

    plt.subplot(133)
    plt.imshow(pure_noise, cmap=rofl.cmap())
    plt.colorbar()
    plt.clim([0, noise_cmax])
    plt.title("Pure noise #{:04.0f}".format(runNo))
    plt.axis("off")
    plt.axis(ax)

    mu.savefig(out_fig_dir + "/{:}_denoised-fig{:}".format(fn, case))
