import json
import sys
from pathlib import Path
from pprint import pprint

import pandas as pd
import pytorch_lightning as pl
from bremsstrahlung_denoising.dataset import SAXSDataInMemory
from bremsstrahlung_denoising.equivariant_model import EqUnet
from bremsstrahlung_denoising.model import Denoiser, DnCNN, NoiseRemover
from bremsstrahlung_denoising.utils import parser_with_common_args
from monai.networks.nets import UNETR, BasicUNet, SwinUNETR
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader


def main(args):
    pl.seed_everything(args.seed, workers=True)

    train_files_signal = (
        pd.read_csv(args.files_csv_train_signal, header=None)
        .to_numpy()
        .squeeze()
        .tolist()
    )
    train_files_noise = (
        pd.read_csv(args.files_csv_train_noise, header=None)
        .to_numpy()
        .squeeze()
        .tolist()
    )

    train_dataset = SAXSDataInMemory(
        signal_dir=args.signal_dir,
        noise_dir=args.noise_dir,
        signal_files_to_use=train_files_signal,
        noise_files_to_use=train_files_noise,
        quantile_normalization=args.quantile_normalization,
        quantile_low=args.normalization_quantile_low,
        quantile_high=args.normalization_quantile_high,
        fixed_constant_normalization=args.fixed_constant_normalization,
        fixed_constant=args.fixed_constant,
        seed=args.seed,
    )

    print()
    print(f"Number of training samples {len(train_dataset)}")
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if args.files_csv_val_signal is not None:
        assert (
            args.files_csv_val_noise is not None
        ), "No validation noise files given but only signal files."

        val_files_signal = (
            pd.read_csv(args.files_csv_val_signal, header=None)
            .to_numpy()
            .squeeze()
            .tolist()
        )
        val_files_noise = (
            pd.read_csv(args.files_csv_val_noise, header=None)
            .to_numpy()
            .squeeze()
            .tolist()
        )

        val_dataset = SAXSDataInMemory(
            signal_dir=args.signal_dir,
            noise_dir=args.noise_dir,
            signal_files_to_use=val_files_signal,
            noise_files_to_use=val_files_noise,
            quantile_normalization=args.quantile_normalization,
            quantile_low=args.normalization_quantile_low,
            quantile_high=args.normalization_quantile_high,
            fixed_constant_normalization=args.fixed_constant_normalization,
            fixed_constant=args.fixed_constant,
            seed=args.seed,
        )

        print(f"number of validation samples {len(val_dataset)}")

        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    else:
        print("[WW]: No validation data is used during training!")
        val_loader = None

    # determine how many output channels
    if args.type == "denoiser":
        out_channels = 1
    else:
        out_channels = 16

    if args.model == "unet16":
        model = BasicUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=out_channels,
            features=(16, 16, 32, 64, 128, 16),
            dropout=0.0,
        )

    elif args.model == "unet32":
        model = BasicUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=out_channels,
            features=(32, 32, 64, 128, 256, 32),
            dropout=0.0,
        )
    elif args.model == "equnet16":
        model = EqUnet(
            signal_shape=(1, 128, 128),
            out_channels=out_channels,
            n_theta=4,
            features=(16, 16, 32, 64, 128, 16),
        )
    elif args.model == "equnet32":
        model = EqUnet(
            signal_shape=(1, 128, 128),
            out_channels=out_channels,
            n_theta=4,
            features=(32, 32, 64, 128, 256, 32),
        )
    elif args.model == "unetr128":
        model = UNETR(
            spatial_dims=2,
            in_channels=1,
            out_channels=out_channels,
            img_size=train_dataset[0]["signal"].shape[1:],
            feature_size=24,
            hidden_size=128,
            mlp_dim=512,
            num_heads=4,
            dropout_rate=0.0,
        )

    elif args.model == "unetr384":
        model = UNETR(
            spatial_dims=2,
            in_channels=1,
            out_channels=out_channels,
            img_size=train_dataset[0]["signal"].shape[1:],
            feature_size=24,
            hidden_size=384,
            mlp_dim=1536,
            num_heads=12,
            dropout_rate=0.0,
        )

    elif args.model == "swin_unetr":
        model = SwinUNETR(
            spatial_dims=2,
            in_channels=1,
            out_channels=out_channels,
            img_size=train_dataset[0]["signal"].shape[1:],
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            feature_size=24,
            drop_rate=0.0,
            downsample="mergingv2",
            use_checkpoint=False,
        )
    elif args.model == "dncnn":
        model = DnCNN(
            in_channels=1,
            n_filters=64,
            n_resblocks=10,
            out_channels=out_channels,
        )

    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_params / 1.e6} mio parameters!")

    if args.type == "denoiser":
        model = Denoiser(
            torch_model=model,
            learning_rate=args.learning_rate,
            loss=args.loss,
        )
    else:
        model = NoiseRemover(
            torch_model=model,
            torch_model_output_filters=out_channels,
            learning_rate=args.learning_rate,
            reco_loss=args.loss,
            channels=1,
            weight_reco=args.weight_reco,
            weight_noise=args.weight_noise,
            noise_loss=args.loss,
            noise_intensity_bound=args.noise_intensity_bound,
        )

    if val_loader is not None:
        ckpt = ModelCheckpoint(
            monitor="val_loss",
            filename="{val_loss:.3f}-{train_loss:.3f}-{epoch:02d}",
            mode="min",
            save_last=True,
            save_top_k=args.num_best_checkpoints,
            every_n_epochs=args.checkpoint_every_n_epochs,
        )

    else:
        ckpt = ModelCheckpoint(
            monitor="train_loss",
            filename="{train_loss:.3f}-{epoch:02d}",
            mode="min",
            save_last=True,
            save_top_k=args.num_best_checkpoints,
            every_n_epochs=args.checkpoint_every_n_epochs,
        )

    callbacks = [ckpt]

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)

    print("Start training")
    trainer.fit(model, train_loader, val_loader)

    return 0


if __name__ == "__main__":
    parser = parser_with_common_args("Train SAXS Denoiser")
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    print("\nParsed args are\n")
    pprint(args)

    default_root_dir = args.default_root_dir
    if default_root_dir is None:
        default_root_dir = "./experiments/bremsstrahlung_denoising/training"
    if not isinstance(default_root_dir, Path):
        default_root_dir = Path(default_root_dir)

    if not default_root_dir.is_dir():
        default_root_dir.mkdir(parents=True)
    else:
        raise ValueError(f"Default_root_dir {default_root_dir} already exists!")

    print(f"\nUsing {default_root_dir} as output directory.")

    # storing the commandline arguments to a json file
    with open(default_root_dir / "commandline_args.json", "w") as of:
        json.dump(vars(args), of, indent=2)

    # now convert to a pathlib object (not done in the beginning because they
    # could not be json-serialized)
    args.default_root_dir = default_root_dir
    assert isinstance(args.default_root_dir, Path)

    retval = main(args)

    sys.exit(retval)
