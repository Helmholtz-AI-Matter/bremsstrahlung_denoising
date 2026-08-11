import bremsstrahlung_denoising.colorscheme as rofl
import bremsstrahlung_denoising.mmmUtils as mu
import matplotlib.pyplot as plt
import numpy as np

fs = [1, 3, 10, 30, 100]
ns = [1, 2, 3, 5]
ns = [2, 3, 5]
mu.figure(8, 6)
for n in ns:
    for f in fs:
        # doing noise
        colsn = rofl.b()
        coln = rofl.o()

        num = f + n * 1000
        try:
            xc, xmean, nc, nmean = mu.loadPickle(
                "validation/valres_{:04.0f}".format(num)
            )

            plt.loglog(xc, xmean, "-", color=colsn, lw=3, alpha=0.3)
            plt.plot(nc, nmean, "-", color=coln, lw=3, alpha=0.3)
        except:
            print("not")

plt.ylim(1e-1, 10000)
plt.xlim(1e-3, 1e3)
plt.loglog([1e-5], [1e-5], "-", color=coln, label="Source data", lw=2)
plt.loglog([1e-5], [1e-5], "-", color=colsn, label="Denoised", lw=2)

xax = np.logspace(-3, 3)
plt.plot(xax, 1 / xax * 100, "k-", label="Noise/Signal", lw=1)
xax2 = np.logspace(-3, 0)
plt.plot(xax2, 1 / xax2 * 10, "r-", lw=1)
plt.plot(xax, xax * 0 + 10, "k-", alpha=0.5)
plt.grid()
plt.legend()
plt.xlabel("Signal / noise [-]")
plt.ylabel("Relative difference to clean data [%]")
plt.title("Validation of ML noise removal")
plt.text(1.1e-2, 7, "10% uncertainty where S/N=1")
plt.text(3e-1, 500, "> 10 fold improvement where S/N is below 1. ", color="r")
plt.text(0.5e-1, 6000, "Each line represents one validation image")

mu.savefig("validation_figs/Validation_result")
