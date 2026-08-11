from argparse import ArgumentParser


def add_common_args(parser):
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)

    model_group = parser.add_argument_group("Model")
    model_group.add_argument(
        "--model",
        type=str,
        choices=[
            "dncnn",
            "unet16",
            "unet32",
            "equnet16",
            "equnet32",
            "unetr128",
            "unetr384",
            "swin_unetr",
        ],
        default="unetr128",
    )
    model_group.add_argument(
        "--type", type=str, choices=["denoiser", "noise_remover"], default="denoiser"
    )

    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument(
        "--signal_dir",
        type=str,
        help="Full path to the directories containing the noise-free signal images",
    )
    io_group.add_argument(
        "--noise_dir",
        type=str,
        help="Full path to the directories containing the noise images",
    )
    io_group.add_argument(
        "--files_csv_train_signal",
        type=str,
        help="A filename pointing to a csv file that contains file names of noise-free tiles that hould be used for training. Those names must be within img_dir.",
    )
    io_group.add_argument(
        "--files_csv_train_noise",
        type=str,
        help="A filename pointing to a csv file that contains file names of pure noise tiles that should be used for training. Those names must be within img_dir.",
    )
    io_group.add_argument(
        "--files_csv_val_signal",
        type=str,
        help="A filename pointing to a csv file that contains file suffix names which should be used for validation. Those names must be within img_dir.",
    )
    io_group.add_argument(
        "--files_csv_val_noise",
        type=str,
        help="A filename pointing to a csv file that contains file names of pure noise tiles that should be used for validation. Those names must be within img_dir.",
    )
    io_group.add_argument(
        "--num_best_checkpoints",
        type=int,
        default=1,
        help="Number of best models to save as checkpoints.",
    )
    io_group.add_argument(
        "--checkpoint_every_n_epochs",
        type=int,
        default=None,
        help="Frequency to write out checkpoints.",
    )
    io_group.add_argument(
        "--quantile_normalization",
        action="store_true",
        default=False,
        help="Instead of normalizing the images and labels by using quantiles, use the raw data.",
    )
    io_group.add_argument(
        "--normalization_quantile_low",
        type=float,
        default=0.01,
        help="Lower bound of the intensity quantile "
        "used for normalization of images.",
    )
    io_group.add_argument(
        "--normalization_quantile_high",
        type=float,
        default=0.9995,
        help="Upper bound of the intensity quantile "
        "used for normalization of images.",
    )
    io_group.add_argument(
        "--fixed_constant_normalization",
        action="store_true",
        default=False,
        help="Normalize images through division by a constant.",
    )
    io_group.add_argument(
        "--fixed_constant",
        type=float,
        default=10000.0,
        help="The constant to divide data by when using fixed_constant_normalization.",
    )

    return parser


def add_model_args(parser):
    model_group = parser.add_argument_group("Denoising model")
    model_group.add_argument("--loss", type=str, choices=["mse", "mae"], default="mse")
    model_group.add_argument(
        "--unet_filters",
        type=int,
        default=16,
        help="Number of feature maps after first, second and in last layer of Unet. After second layer, feature map sizes get doubled in each layer.",
    )

    remover_group = parser.add_argument_group("Noise remover model")
    remover_group.add_argument(
        "--weight_reco",
        type=float,
        default=1.0,
        help="Weight of reconstruction loss of signal",
    )
    remover_group.add_argument(
        "--weight_noise",
        type=float,
        default=1.0,
        help="Weight of loss for noise statistics",
    )
    remover_group.add_argument(
        "--noise_intensity_bound",
        type=float,
        default=None,
        help="Maximum value of pixel intensity of the prediction " "for the noise.",
    )

    return parser


def parser_with_common_args(title):
    parser = ArgumentParser(title)

    parser = add_common_args(parser)
    parser = add_model_args(parser)

    return parser
