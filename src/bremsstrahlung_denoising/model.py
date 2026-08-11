import numpy as np
import pytorch_lightning as pl
import torch
from monai.networks.blocks import Convolution, ResidualUnit
from skimage.metrics import mean_squared_error as MSE
from skimage.metrics import peak_signal_noise_ratio as PSNR


def MAE(lbl, pred):
    return np.mean(np.abs(lbl - pred))


def metrics_from_step_outputs(step_output_list):
    # metrics computed per image
    psnrs = []
    mses = []
    maes = []
    for batch_d in step_output_list:
        lbl_batch = batch_d["signal"]
        pred_batch = batch_d["prediction_signal"]

        for idx in range(len(lbl_batch)):
            pred = pred_batch[idx].cpu().numpy()
            lbl = lbl_batch[idx].cpu().numpy()

            data_range = lbl.max() - lbl.min()
            psnrs.append(PSNR(lbl, pred, data_range=data_range))
            mses.append(MSE(lbl, pred))
            maes.append(MAE(lbl, pred))

    return {"psnr": np.mean(psnrs), "mse": np.mean(mses), "mae": np.mean(maes)}


class Denoiser(pl.LightningModule):
    def __init__(self, torch_model, learning_rate=1.0e-4, loss="mse"):
        """
        This will directly estimate a noise-removed version
        of the input image using the given torch model.

        Parameters
        ----------
        """
        super().__init__()

        self.save_hyperparameters()

        self.model = torch_model

        if loss == "mse":
            self.loss_fn = torch.nn.MSELoss()
        elif loss == "mae":
            self.loss_fn = torch.nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss {loss}. Choose 'mse' or 'mae'!")

        # all signal outputs should be non-negative
        self.output_act = torch.nn.ReLU()

    def forward(self, img):
        # second output could be for estimated noise
        # which we dont predict in this model
        return self.output_act(self.model(img)), None

    def _predict(self, batch):
        # model input, f(S+N)
        noisy_signal = batch["noisy_signal"]
        estimated_signal, _ = self.forward(noisy_signal)

        retdict = {
            "noisy_signal": noisy_signal,  # the model input
            "signal": batch["signal"],  # the desired model output
            "prediction_signal": estimated_signal,
        }
        return retdict

    def shared_step(self, batch):
        res = self._predict(batch)

        res["loss"] = self.loss_fn(res["prediction_signal"], res["signal"])

        for k, v in res.items():
            if isinstance(v, torch.Tensor) and k != "loss":
                res[k] = v.detach()

        return res

    def training_step(self, batch, batch_idx):
        batch_data = self.shared_step(batch)
        self.log(
            "train_loss",
            batch_data["loss"],
            logger=True,
            on_epoch=True,
            on_step=True,
            prog_bar=False,
            sync_dist=False,
        )

        return batch_data

    def validation_step(self, batch, batch_idx):
        batch_data = self.shared_step(batch)
        self.log(
            "val_loss",
            batch_data["loss"],
            logger=True,
            on_epoch=True,
            on_step=True,
            prog_bar=False,
            sync_dist=False,
        )

        return batch_data

    def predict_step(self, batch, batch_idx):
        return self._predict(batch)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False,
        )

    def training_epoch_end(self, step_output_list):
        metrics = metrics_from_step_outputs(step_output_list)

        for metric in metrics:
            self.log(
                f"train_{metric}",
                metrics[metric],
                on_step=False,
                logger=True,
                prog_bar=True,
                on_epoch=True,
                sync_dist=False,
            )

    def validation_epoch_end(self, step_output_list):
        metrics = metrics_from_step_outputs(step_output_list)

        for metric in metrics:
            self.log(
                f"val_{metric}",
                metrics[metric],
                on_step=False,
                logger=True,
                prog_bar=True,
                on_epoch=True,
                sync_dist=False,
            )

        self.log("hp_metric", metrics["psnr"])


class RowAndColStatsLoss(torch.nn.Module):
    """
    This takes two images and compares row and
    column statistics
    """

    def __init__(self, loss_fn):
        super().__init__()

        self.loss_fn = loss_fn

    def _stats(self, batch, dim):
        means = batch.mean(dim=dim)
        stds = batch.std(dim=dim)

        return torch.cat([means, stds], dim=-1)

    def _row_stats(self, batch):
        # NOTE: batch is assumed to have shape B x C x H x W
        return self._stats(batch, dim=(1, 3))

    def _col_stats(self, batch):
        return self._stats(batch, dim=(1, 2))

    def forward(self, pred_batch, label_batch):
        """
        pred_batch: B x C x H x W
        label_batch: B x C x H x W
        """
        row_stats_pred = self._row_stats(pred_batch)
        col_stats_pred = self._col_stats(pred_batch)
        stats_pred = torch.cat([row_stats_pred, col_stats_pred], dim=-1)

        row_stats_lbl = self._row_stats(label_batch)
        col_stats_lbl = self._col_stats(label_batch)
        # concatenate everything
        stats_lbl = torch.cat([row_stats_lbl, col_stats_lbl], dim=-1)

        # apply loss
        return self.loss_fn(stats_pred, stats_lbl)


class UpperBoundedOutput(torch.nn.Module):
    """
    Applies a scaled sigmoid to the input so the output
    will be in the range of (0, upper_bound) instead of
    (0, 1)

    Parameters
    ----------
        upper_bound: a positive float
    """

    __constants__ = ["inplace"]
    inplace: bool

    def __init__(self, upper_bound):
        super().__init__()
        assert upper_bound > 0
        self.upper_bound = upper_bound
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input):
        return self.upper_bound * self.sigmoid(input)


class NoiseRemover(pl.LightningModule):
    def __init__(
        self,
        torch_model,
        torch_model_output_filters,
        learning_rate=1.0e-4,
        reco_loss="mse",
        channels=1,
        weight_reco=1.0,
        weight_noise=1.0,
        noise_loss="mse",
        noise_intensity_bound=None,
    ):
        """
        This uses the model to predict some latent space and from that predicts additive noise component that will be removed from the
        original image.
        It assumes the noisy image got created with
            noisy = clean + additive_noise

        Parameters
        ----------
        torch_model_output_filters: number of feature maps that come out of the given torch_model
        channels: number of output channels of the resulting denoised image. Should be identical
                      to the number of channels of the image that gets used as input.
        noise_intensity_bound: None or positive float
            If None, a ReLU will be used as output activation for the noise prediction, making sure predictions are
            always >= 0, but not upper bounded.
            If this is not None, we will use a sigmoid as output activation for the noise and scale it appropriately by
            the given number so that output intensity will always be below the given value
        """

        super().__init__()

        # NOTE: this discards non-picklabel hyperparameters, which leads to problems later on when trying to restore checkpoints.
        # This only happens if the torch_model is an instance of
        # 'EqUnet'
        self.save_hyperparameters()

        self.model = torch_model

        if reco_loss == "mse":
            self.reco_loss_fn = torch.nn.MSELoss()
        elif reco_loss == "mae":
            self.reco_loss_fn = torch.nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss {reco_loss}. Choose 'mse' or 'mae'!")

        if noise_loss == "mse":
            self.noise_loss_fn = torch.nn.MSELoss()
        elif noise_loss == "mae":
            self.noise_loss_fn = torch.nn.L1Loss()
        elif noise_loss == "rowcolstats":
            self.noise_loss_fn = RowAndColStatsLoss(
                loss_fn=torch.nn.SmoothL1Loss(reduction="mean")
            )
        else:
            raise ValueError(
                f"Unknown noise loss {noise_loss}. Choose 'mse', 'mae' or 'rowcolstats!"
            )

        # keeps dimensionality and is guaranteed to be >= 0

        noise_modules = [
            torch.nn.Conv2d(
                in_channels=torch_model_output_filters,
                out_channels=channels,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
        ]
        if noise_intensity_bound is None:
            noise_modules.append(torch.nn.ReLU())
        else:
            noise_modules.append(UpperBoundedOutput(upper_bound=noise_intensity_bound))

        self.additive_noise = torch.nn.Sequential(*noise_modules)

        # all signal outputs should be non-negative
        self.output_act = torch.nn.ReLU()

    def forward(self, noisy_img):
        latent = self.model(noisy_img)  # f(S+N)
        estimated_noise = self.additive_noise(latent)  # Nhat, >= 0

        # NOTE: this could become negative but the physics require
        # this to be >= 0 as well, so we apply RELU
        estimated_signal = self.output_act(noisy_img - estimated_noise)  # S+N-Nhat

        return estimated_signal, estimated_noise

    def _predict(self, batch):
        # model input, f(S+N)
        noisy_signal = batch["noisy_signal"]
        estimated_signal, estimated_noise = self.forward(noisy_signal)  # S+N-Nhat, Nhat

        retdict = {
            "noisy_signal": noisy_signal,  # the model input
            "signal": batch["signal"],  # the desired model output
            "noise": batch["noise"],
            "prediction_signal": estimated_signal,
            "prediction_noise": estimated_noise,
        }
        return retdict

    def shared_step(self, batch, train_or_val):
        res = self._predict(batch)

        # Can we have one loss component for the reconstruction
        # of the clean signal
        # and another one for getting statistics of the noise correct?

        # Loss(S+N-Nhat, S) -> for L1/MSE this would be exactly the same as the noise loss below
        # because the S cancels out, which is why we use the loss with the row/column statistics.
        # NOTE: the above hower is no longer true if we apply a ReLU to S+N-Nhat to force the predicted
        # signal to be non-negative.

        reco_loss = self.reco_loss_fn(res["prediction_signal"], res["signal"])

        # Loss(Nhat, N)
        noise_loss = self.noise_loss_fn(res["prediction_noise"], res["noise"])

        loss = (
            self.hparams.weight_noise * noise_loss
            + self.hparams.weight_reco * reco_loss
        )

        for k, v in res.items():
            if isinstance(v, torch.Tensor):
                res[k] = v.detach()

        res["loss"] = loss
        res["reco_loss"] = reco_loss
        res["noise_loss"] = noise_loss

        # logging
        logging_args = dict(
            logger=True, on_epoch=True, on_step=False, prog_bar=False, sync_dist=False
        )

        for k in res:
            if "loss" not in k:
                continue
            self.log(f"{train_or_val}_{k}", res[k], **logging_args)

        return res

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def predict_step(self, batch, batch_idx):
        return self._predict(batch)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            amsgrad=False,
        )

    def _epoch_end(self, step_output_list, train_or_val):
        metrics = metrics_from_step_outputs(step_output_list)

        logging_args = dict(
            on_step=False, logger=True, prog_bar=True, on_epoch=True, sync_dist=False
        )

        for metric in metrics:
            self.log(f"{train_or_val}_{metric}", metrics[metric], **logging_args)

        if train_or_val == "val":
            self.log("hp_metric", metrics["psnr"])

    def training_epoch_end(self, step_output_list):
        self._epoch_end(step_output_list, "train")

    def validation_epoch_end(self, step_output_list):
        self._epoch_end(step_output_list, "val")


class DnCNN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        activation=("leakyrelu", dict(negative_slope=0.2)),
        n_filters=64,
        n_resblocks=10,
        out_channels=1,
    ):
        super().__init__()

        adn_ordering = "NA"
        norm = "batch"

        # kernel = 3, padding = 1 and stride = 1 ensures that
        # spatial dimension is retained
        self.conv1 = Convolution(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=n_filters,
            kernel_size=3,
            padding=1,
            act=activation,
            adn_ordering=adn_ordering,
            norm=norm,
        )

        res_blocks = [
            ResidualUnit(
                spatial_dims=2,
                in_channels=n_filters,
                out_channels=n_filters,
                adn_ordering=adn_ordering,
                last_conv_only=False,
                subunits=1,
                act=activation,
                norm=norm,
            )
            for i in range(n_resblocks)
        ]
        self.res_blocks = torch.nn.Sequential(*res_blocks)

        # final 1x1 convolution
        self.conv_out = Convolution(
            spatial_dims=2,
            in_channels=n_filters,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            act=activation,
            adn_ordering=adn_ordering,
            norm=norm,
        )

    def forward(self, noisy_img):
        x = self.conv1(noisy_img)
        x = self.res_blocks(x)
        return self.conv_out(x)
