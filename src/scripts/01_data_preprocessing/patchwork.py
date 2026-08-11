# %% Imports

import os
import random

import bremsstrahlung_denoising.colorscheme as rofl
import bremsstrahlung_denoising.mmmUtils as mu
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# %% Settings

# Run this script twice: once with do_noise=0 and once with do_noise=1, to process both signal and noise images.
do_noise = 0

tilesize = 128  # px; 128 is the best choice for JungFrau cameras
random_rotation = (
    True  # randomize rotation for signal images, to avoid bias in the training data.
)

debug = 0  # For testing purposes


do_signal = 1 - do_noise
rotsuf = ""
tiles_attempts_per_image = 60

signal_threshold = 500  # keV/px - level since when we consider the value to be a signal
signal_threshold_hot_pixels = 1000  # keV/px single pixel at which we consider it to be a hot pixel and set it to 0
signal_amplitude_randomization = 1000  # maximal factor by which we randomize the amplitude of the signal images, to avoid bias in the training data. 0 means no randomization.

if do_noise:  # Directories
    dira = "../export/noise/"
    dire = "./export/noise_tiles/"
    stri = "Noise"
    random_rotation = 0
else:
    dira = "../export/signal/"
    dire = "./export/signal_tiles" + rotsuf + "/"
    stri = "Signal"

# %% The workhorse cell

if random_rotation:
    rotsuf = "_rot"
    tiles_attempts_per_image = 100
if debug:
    tiles_attempts_per_image = 1
    # tiles_attempts_per_image=100

bigfig = random_rotation

files = os.listdir(dira)
tilpix = tilesize * tilesize
r = 5
c = 7
for file in files:
    if do_noise and file.find("masked") < 6:
        continue
    runstr = file[1:5]
    run = int(runstr)
    print(file)
    img = mu.loadTiff(dira + file)
    img[img < 0] = 0
    if not bigfig:
        mu.figure(13, 7)
        plt.imshow(img)
        plt.title(stri + " " + runstr)
        if do_signal:
            plt.clim(0, 0.1)
        else:
            plt.clim(0, 50)
    else:
        mu.figure(12, 10)
        ii = 0

    for attempt in np.arange(tiles_attempts_per_image):
        if random_rotation:
            rot = int(random.random() * (360))
            #            rot=30
            img = Image.open(dira + file)
            imgrot = img.rotate(rot, expand=1)

            x = int(random.random() * (np.shape(imgrot)[1] - tilesize))
            y = int(random.random() * (np.shape(imgrot)[0] - tilesize))
            img = np.array(imgrot)
            img[img < 0] = 0
            rect = [x, x + tilesize, y, y + tilesize]

            if not bigfig:
                mu.figure()
                plt.imshow(img)
                plt.colorbar()
                plt.clim(0, 0.1)
        else:
            x = int(random.random() * (1024 - tilesize))
            y = int(random.random() * (512 - tilesize))
            rect = [x, x + tilesize, y, y + tilesize]

        rec = mu.cutRect(rect, img)
        if do_signal:
            rec[rec > signal_threshold_hot_pixels] = 0
            suma = np.sum(rec)
            if 0:
                rec2 = rec * 1

                rec2[rec2 < signal_threshold_hot_pixels] = 0
                rec2[rec2 >= signal_threshold_hot_pixels] = 1
                hotpixels = np.sum(rec2)
                print(hotpixels)
            good = suma > signal_threshold

        if do_noise:
            suma = np.sum(np.isfinite(rec))
            good = suma == tilpix

        if good:
            col = "r"
            # amplitude randomization
            ampl = 1
            if signal_amplitude_randomization != 0:
                sar_exp = np.log10(signal_amplitude_randomization)
                ampl = 10 ** (random.random() * sar_exp) + 1
                rec = rec * ampl
            mu.saveTiff(rec, dire + runstr + "_{:03.0f}.tiff".format(attempt))
            if bigfig:
                ii += 1
                if ii > r * c:
                    plt.title(ii)
                else:
                    plt.subplot(r, c, ii)
                    plt.imshow(rec, norm=matplotlib.colors.LogNorm(), cmap=rofl.cmap())
                    plt.clim(0.1, 500)
                    plt.axis("off")
                    plt.title("{:} ({:.0f}x)".format(ii, ampl))
        else:
            col = "w"
        if not bigfig:
            mu.drawRect(rect, color=col)
            plt.text(x, y, "{:.0f}".format(suma), color="w")

    if bigfig:
        plt.subplot(r, c, 1)
        plt.title(stri + " " + runstr)
    mu.savefig("figs/Tiling{:}_{:}_{:}.png".format(stri, rotsuf, runstr))
    if debug:
        break
