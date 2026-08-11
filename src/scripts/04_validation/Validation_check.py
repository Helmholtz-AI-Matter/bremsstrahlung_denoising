from pathlib import Path

import bremsstrahlung_denoising.mmmUtils as mu
import matplotlib.pyplot as plt
import numpy as np

tilesize = 128
random_rotation = 1

# tiles_attempts_per_image=1
debug = 0
do_noise = 1
do_signal = 1 - do_noise
rotsuf = ""

noise_run = 713
signal_run = 712

noise = np.zeros((1024, 512))
patch_size = 128
stride = 128

n_rows = noise.shape[0] // patch_size
n_cols = noise.shape[1] // patch_size


# files=os.listdir('export/noise_tiles/{:04.0f}*'.format(noise_run))

noise_tiles_dir = Path("export/noise_tiles/")
noise_files = sorted(noise_tiles_dir.rglob("**/{:04.0f}*.tiff".format(noise_run)))
ni = 0
fs = [1, 3, 10, 30, 100]
fs = [3]
ax = [250, 500, 220, 440]
for f in fs:
    sn = mu.loadTiff("validation/Val_f{:03.0f}_sn.tif".format(f))
    sig = mu.loadTiff("validation/Val_f{:03.0f}_sig.tif".format(f)) * f
    denoised = mu.loadTiff(
        "validation/p3129_r{:04.0f}_denoised_equnet32.tiff".format(f)
    )
    mu.figure(18, 16)
    plt.subplot(231)
    plt.imshow(sig.transpose())
    plt.colorbar()
    plt.clim(0, 50)
    plt.axis(ax)
    plt.title("Signal")

    plt.subplot(232)
    plt.imshow(sn.transpose())
    plt.colorbar()
    plt.clim(0, 50)
    plt.axis(ax)
    plt.title("S+N (Input for denoising)")

    plt.subplot(233)
    plt.imshow(denoised.transpose())
    plt.colorbar()
    plt.clim(0, 50)
    plt.axis(ax)
    plt.title("Denoised")

    plt.subplot(234)
    plt.imshow((denoised / sig).transpose())
    plt.colorbar()
    plt.clim(0.5, 1.5)
    plt.axis(ax)
    plt.title("Denoised / signal")

    # pixel plots
    # %
    rel_error = (sig.flatten() - denoised.flatten()) / sig.flatten()
    plt.subplot(235)
    plt.plot(sig.flatten(), rel_error * 100, ".")
    plt.ylabel("reltaive error [%]")
