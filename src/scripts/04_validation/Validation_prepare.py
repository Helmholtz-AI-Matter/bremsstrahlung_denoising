import random
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

patch_size = 128
stride = 128
noise = np.zeros((1024, 512))

n_rows = noise.shape[0] // patch_size
n_cols = noise.shape[1] // patch_size


# files=os.listdir('export/noise_tiles/{:04.0f}*'.format(noise_run))

noise_tiles_dir = Path("export/noise_tiles/")
noise_files = sorted(noise_tiles_dir.rglob("**/{:04.0f}*.tiff".format(noise_run)))
ni = 0
fs = [1, 3, 10, 30, 100]
# fs=[1]
ns = [1, 2, 3, 5]
for n in ns:
    for f in fs:
        # doing noise
        num = f + n * 1000

        noise = np.zeros((1024, 512))
        for nnn in np.arange(n):
            for ii in range(n_rows):
                row_start = ii * stride
                row_end = row_start + patch_size
                for jj in range(n_cols):
                    col_start = jj * stride
                    col_end = col_start + patch_size
                    tile = mu.loadTiff(noise_files[ni])
                    ni = int(np.round(random.random() * np.size(noise_files)))
                    noise[row_start:row_end, col_start:col_end] += tile
                    ni += 1
                    if ni >= np.size(noise_files):
                        ni = 0

        mu.figure(14, 14)
        plt.subplot(311)
        plt.imshow(noise.transpose())
        plt.colorbar()
        plt.clim(0, 50)
        plt.title("noise, case {:04.0f}".format(num))

        # %%signal
        sig = mu.loadTiff("../export/signal/r{:04.0f}.tiff".format(signal_run))
        sig = sig * f
        plt.subplot(312)
        plt.imshow(sig.transpose())
        plt.colorbar()
        plt.clim(0, 50)
        plt.title("signal")

        # %%
        sn = noise + sig
        plt.subplot(313)
        plt.imshow(sn.transpose())
        plt.colorbar()
        plt.clim(0, 50)
        plt.title("signal+noise")

        mu.saveTiff(sig, "validation/Val_f{:04.0f}_sig.tif".format(num))
        mu.saveTiff(noise, "validation/Val_f{:04.0f}_noise.tif".format(num))
        mu.saveTiff(sn, "validation/Val_f{:04.0f}_sn.tif".format(num))
        mu.savefig("validation_figs/Input_{:04.0f}".format(num))
