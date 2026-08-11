import json
import sys
from pathlib import Path
from pprint import pprint

import pandas as pd
import pytorch_lightning as pl
import torch
from bremsstrahlung_denoising.dataset import SAXSDataInMemory
from bremsstrahlung_denoising.model import (
    Denoiser,
    NoiseRemover,
    metrics_from_step_outputs,
)
from bremsstrahlung_denoising.utils import parser_with_common_args
from torch.utils.data import DataLoader


def main(args):
    val_files_signal = (
        pd.read_csv(args.files_csv_val_signal, header=None)
        .to_numpy()
        .squeeze()
        .tolist()
    )
    val_files_noise = (
        pd.read_csv(args.files_csv_val_noise, header=None).to_numpy().squeeze().tolist()
    )

    dataset = SAXSDataInMemory(
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

    print()
    print(f"Number of samples {len(dataset)}")

    data_loader = DataLoader(
        dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )

    if args.type == "denoiser":
        trained_model = Denoiser.load_from_checkpoint(checkpoint_path=args.ckpt_file)
    else:
        trained_model = NoiseRemover.load_from_checkpoint(
            checkpoint_path=args.ckpt_file
        )

    trained_model.eval()
    trained_model.freeze()

    print(f"Loaded trained model from checkpoint {args.ckpt_file}.")
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        # devices=args.gpus
    )

    pred_step_outputs = trainer.predict(
        trained_model, dataloaders=data_loader, ckpt_path=args.ckpt_file
    )

    metrics = metrics_from_step_outputs(pred_step_outputs)
    metrics = pd.DataFrame(metrics, index=[0])
    print()
    print(f"Metrics (storing to {args.output_dir})")
    print(metrics)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)

    return 0


if __name__ == "__main__":
    parser = parser_with_common_args("Inference for SAXS Denoiser")
    parser.add_argument(
        "--ckpt_file",
        type=str,
        help="File pointing to a trained model checkpoint for inference.",
    )
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()
    print("\nParsed args are\n")
    pprint(args)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = "./experiments/bremsstrahlung_denoising/inference"
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)
    else:
        raise ValueError(f"Output_dir {output_dir} already exists!")

    print(f"\nUsing {output_dir} as output directory.")

    # storing the commandline arguments to a json file
    with open(output_dir / "commandline_args.json", "w") as of:
        json.dump(vars(args), of, indent=2)

    # now convert to a pathlib object (not done in the beginning because they
    # could not be json-serialized)
    args.output_dir = output_dir
    assert isinstance(args.output_dir, Path)

    retval = main(args)

    sys.exit(retval)
