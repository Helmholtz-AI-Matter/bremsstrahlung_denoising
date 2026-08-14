#!/usr/bin/env python

# Imports

import shutil
import zipfile
from fnmatch import fnmatch
from pathlib import Path
from urllib.parse import urljoin

import bremsstrahlung_denoising.inference_patchwise as ip
import bremsstrahlung_denoising.mmmUtils as mu
import requests
import tifffile
from bs4 import BeautifulSoup

# settings
MODEL_TYPE = "noise_remover"
SKIP_EXISTING = True
DRAW_OUTPUT = True
RECORD_URL = (
    "https://rodare.hzdr.de/record/4786"  # includes our trained model checkpoints
)
DOWNLOAD_DIR = Path("trained_saxs_model")


ax = [600, 800, 50, 350]  # axes, just for plotting
# cl = [1, 1e2]  # color lim for data in figures (on log scale)
cl = [0, 50]  # color lim for data in figures (on log scale)
noise_cmax = 50  # colorbar maximum to show Pure noise (linear scale)


def _find_zip_download_url(record_url: str) -> str:
    """
    Parse the RODARE record page and locate the ZIP download link.
    """
    r = requests.get(record_url, timeout=30)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    # Prefer links explicitly ending in .zip
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".zip"):
            return urljoin(record_url, href)

    # Fall back to links containing "download"
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "download" in href.lower():
            return urljoin(record_url, href)

    raise RuntimeError("Could not locate ZIP download link on record page.")


def get_checkpoint_files(
    record_url: str = RECORD_URL,
    download_dir: Path | str = DOWNLOAD_DIR,
) -> list[Path]:
    """
    Downloads the ZIP from the given RODARE record (only if needed), extracts it (only if needed),
    and returns all val_loss*.ckpt files in lexicographical order.
    """
    download_dir = Path(download_dir)
    zip_path = download_dir / "dataset.zip"
    extract_dir = download_dir / "extracted"

    # If matching checkpoints already exist, return them immediately.
    existing_ckpts = sorted(extract_dir.rglob("val_loss*.ckpt"))
    if existing_ckpts:
        print("Checkpoints already present. Will not download again!")
        return existing_ckpts

    download_dir.mkdir(parents=True, exist_ok=True)

    # Download ZIP only if it is not already present.
    if not zip_path.exists():
        zip_url = _find_zip_download_url(record_url)
        print("Downloading zip file!")
        with requests.get(zip_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                shutil.copyfileobj(r.raw, f)

    # Extract only matching checkpoints from the ZIP.
    extract_dir.mkdir(parents=True, exist_ok=True)
    print("Extracting matching checkpoint files")
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            if fnmatch(Path(member.filename).name, "val_loss*.ckpt"):
                zf.extract(member, extract_dir)

    ckpts = sorted(extract_dir.rglob("val_loss*.ckpt"))

    if not ckpts:
        raise RuntimeError("No val_loss*.ckpt files found after extraction.")

    return ckpts


checkpoint_files = get_checkpoint_files(
    record_url=RECORD_URL, download_dir=DOWNLOAD_DIR
)
print("Found checkpoint files:\n", checkpoint_files)
ckpt_file = checkpoint_files[1]  # lowest validation loss

model = ip.load_model(ckpt_file, model_type=MODEL_TYPE)

# Reading the file to process
datafilename = "p3129_r0547_JF4_PPU_thl6.tif"
fn = datafilename[6:11]

noisy_signal = tifffile.imread(datafilename)

fntiff = f"{datafilename[:-4]}_denoised.tif"

if Path(fntiff).is_file() and SKIP_EXISTING:
    print("## Skipping inference because already done.")
    if DRAW_OUTPUT:
        prediction = mu.loadTiff(fntiff)

else:
    print(f"Doing prediction for {datafilename} *****************")

    prediction = ip.doit(
        noisy_signal, model,
        # those are the quantile normalization params used to train
        # the model, do not change!
        qlow=0.01,
        qhigh=0.995,
    )  # the main work is done here, the prediction is returned

    mu.saveTiff(prediction, fntiff)  # Storing the results in the tiff file


if DRAW_OUTPUT:  # drawing
    print(" Drawing output figure.")
    ip.draw_it(
        noisy_signal,
        prediction,
        runNo=fn[1:],
        case="",
        fn=fn,
        draw_output=DRAW_OUTPUT,
        ax=ax,
        noise_cmax=noise_cmax,
        out_fig_dir=".",
        cl=cl,
    )
