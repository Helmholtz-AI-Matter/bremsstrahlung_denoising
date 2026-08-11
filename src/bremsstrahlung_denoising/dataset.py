from pathlib import Path

import numpy as np
import tifffile
import torch


def normalize(img, low, high, eps=1.0e-20, clip=True):
    # we have to add a small eps to handle the case where both quantiles are equal
    # to avoid dividing by zero
    scaled = (img - low) / (high - low + eps)

    if clip:
        scaled = np.clip(scaled, 0, 1)

    return scaled


def quantile_normalization(
    img, quantile_low=0.01, quantile_high=0.9995, eps=1.0e-20, clip=True
):
    """
    First scales the data so that values below quantile_low are smaller
    than 0 and values larger than quantile_high are larger than one.
    Then optionally clips to (0, 1) range.
    """

    qlow = np.quantile(img, quantile_low)
    qhigh = np.quantile(img, quantile_high)

    scaled = normalize(img, low=qlow, high=qhigh, eps=eps, clip=clip)
    return scaled, qlow, qhigh


def fixed_constant_normalize(input, constant, clip=True):
    # image / high followed by clipping optionally
    return normalize(input, low=0, high=constant, eps=0, clip=clip)


class SAXSDataFiles:
    def __init__(
        self,
        signal_dir,
        noise_dir,
        signal_files_to_use=None,
        noise_files_to_use=None,
        file_suffix=".tiff",
    ):
        self.signal_dir = signal_dir
        if not isinstance(self.signal_dir, Path):
            self.signal_dir = Path(self.signal_dir)

        self.noise_dir = noise_dir
        if not isinstance(self.noise_dir, Path):
            self.noise_dir = Path(self.noise_dir)

        # NOTE: the files do not have to be of equal length as we randomly add
        # noises to the same image
        signal_files = [
            f
            for f in self.signal_dir.iterdir()
            if f.is_file() and f.suffix == file_suffix
        ]
        noise_files = [
            f
            for f in self.noise_dir.iterdir()
            if f.is_file() and f.suffix == file_suffix
        ]

        self.signal_files = []
        for i in range(len(signal_files)):
            signal_name = signal_files[i].name
            if signal_files_to_use is None:
                self.signal_files.append(signal_files[i])
            else:
                # we need to check if this file got selected
                if signal_name in signal_files_to_use:
                    self.signal_files.append(signal_files[i])
                else:
                    # print(f"Skip {signal_suffix} because not listed in {files_csv}!")
                    pass
        # number of signal files determines the length of the dataset
        self.n_samples = len(self.signal_files)
        if signal_files_to_use is not None:
            assert self.n_samples == len(
                signal_files_to_use
            ), f"{self.n_samples} != {len(signal_files_to_use)}"

        self.noise_files = []
        for i in range(len(noise_files)):
            noise_name = noise_files[i].name
            if noise_files_to_use is None:
                self.noise_files.append(noise_files[i])
            else:
                # we need to check if this file got selected
                if noise_name in noise_files_to_use:
                    self.noise_files.append(noise_files[i])
                else:
                    # print(f"Skip {signal_suffix} because not listed in {files_csv}!")
                    pass

    def __len__(self):
        return self.n_samples


def load_data(data_files_object):
    signals = np.stack([tifffile.imread(f) for f in data_files_object.signal_files])
    noises = np.stack([tifffile.imread(f) for f in data_files_object.noise_files])

    for idx in range(len(signals)):
        if np.any(signals[idx] < 0):
            raise ValueError(
                f"Signal file {data_files_object.signal_files[idx]} contains negative values!"
            )

    for idx in range(len(noises)):
        if np.any(noises[idx] < 0):
            raise ValueError(
                f"Noise file {data_files_object.noise_files[idx]} contains negative values!"
            )

    return signals, noises


class SAXSDataInMemory(torch.utils.data.Dataset):
    def __init__(
        self,
        signal_dir,
        noise_dir,
        signal_files_to_use=None,
        noise_files_to_use=None,
        quantile_normalization=True,
        quantile_low=0.01,
        quantile_high=0.9995,
        fixed_constant_normalization=False,
        fixed_constant=10000,
        file_suffix=".tiff",
        seed=42,
    ):
        if quantile_normalization and fixed_constant_normalization:
            raise ValueError(
                "quantile_normalization and fixed_constant_normalization can't both be True!"
            )

        self.rng = np.random.RandomState(seed)

        self.quantile_normalization = quantile_normalization
        if self.quantile_normalization:
            assert 0 <= quantile_low <= 1
            assert 0 <= quantile_high <= 1
            assert quantile_low < quantile_high
            self.quantile_low = quantile_low
            self.quantile_high = quantile_high

        self.fixed_constant_normalization = fixed_constant_normalization
        if self.fixed_constant_normalization:
            assert 0 < fixed_constant
            self.fixed_constant = fixed_constant

        self.files = SAXSDataFiles(
            signal_dir=signal_dir,
            noise_dir=noise_dir,
            signal_files_to_use=signal_files_to_use,
            noise_files_to_use=noise_files_to_use,
            file_suffix=file_suffix,
        )

        # note: this data is not normalized
        self.signals_raw, self.noises_raw = load_data(self.files)

    def __len__(self):
        return len(self.signals_raw)

    def __getitem__(self, idx):
        signal = self.signals_raw[idx]
        # draw the noise from a different sample
        noise_idx = self.rng.choice(len(self.noises_raw))
        noise = self.noises_raw[noise_idx]

        # S + N
        noisy_signal = signal + noise

        if self.quantile_normalization:
            # f(S+N), NOTE: this is different from f(S) + f(N), because in the latter
            # case, different intensity scales are not reflected properly
            noisy_signal, qlow, qhigh = quantile_normalization(
                noisy_signal, self.quantile_low, self.quantile_high, clip=True
            )

            # after obtaining the quantiles from the noisy image, we reuse
            # them to normalize img and signal and do not use their
            # respective quantiles
            # f(S)
            signal = normalize(signal, low=qlow, high=qhigh, clip=True)
            # f(N)
            noise = normalize(noise, low=qlow, high=qhigh, clip=True)
        elif self.fixed_constant_normalization:
            # with low=0, eps=0 this is equivalent to division by the 'high'
            # argument
            noisy_signal = fixed_constant_normalize(
                noisy_signal, constant=self.fixed_constant, clip=True
            )
            signal = fixed_constant_normalize(
                signal, constant=self.fixed_constant, clip=True
            )
            noise = fixed_constant_normalize(
                noise, constant=self.fixed_constant, clip=True
            )

        # add a channel dimension in the front and convert to float32
        signal = np.expand_dims(signal, 0).astype(np.float32)
        noise = np.expand_dims(noise, 0).astype(np.float32)
        noisy_signal = np.expand_dims(noisy_signal, 0).astype(np.float32)

        ret_dict = {
            "signal": signal,  # f(S)
            "noise": noise,  # f(N)
            "noisy_signal": noisy_signal,  # f(S + N) where f is preprocessing function
            "noise_idx": noise_idx,
        }

        if self.quantile_normalization:
            ret_dict["normalization_quantile_low"] = qlow
            ret_dict["normalization_quantile_high"] = qhigh
        elif self.fixed_constant_normalization:
            ret_dict["normalization_fixed_constant"] = self.fixed_constant

        return ret_dict
