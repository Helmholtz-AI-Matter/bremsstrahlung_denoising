# Imports
from pathlib import Path

# import bremsstrahlung_denoising.rossendorfer_farbenliste as rofl
import bremsstrahlung_denoising.inference_patchwise as ip
import bremsstrahlung_denoising.mmmUtils as mu
import requests
import tifffile



def get_checkpoint_files_from_RODARE(
        rodare_record_id: int, output_dir: Path) -> list[Path]:
    """Download missing *.ckpt files from RODARE repository and return their paths."""

    output_dir.mkdir(parents=True, exist_ok=True)

    rodare_url = f"https://rodare.hzdr.de/api/records/{rodare_record_id}"

    # Get record metadata
    record = requests.get(rodare_url, timeout=30).json()

    checkpoints = []

    for file in record["files"]:
        filename = file["key"]

        if not filename.endswith(".ckpt"):
            continue

        path = output_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        # Don't download files that are already present.
        if not path.exists():
            url = file["links"]["self"]

            print(f"Downloading {filename}...")
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            with path.open("wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        checkpoints.append(path)

    if not checkpoints:
        raise RuntimeError("No *.ckpt files found in the RODARE record.")

    return sorted(checkpoints)


# settings
SKIP_EXISTING = True
DRAW_OUTPUT = True
RODARE_RECORD_ID = 4956  # the rodare record where we stored the trained model and gave it a DOI

ax = [600, 800, 50, 350]  # axes, just for plotting
cl = [1, 1e2]  # color lim for data in figures (on log scale)
cl = [0, 50]  # color lim for data in figures (on log scale)
noise_cmax = 50  # colorbar maximum to show Pure noise (linear scale)

download_dir = Path(f"rodare_{RODARE_RECORD_ID}")

checkpoint_files = get_checkpoint_files_from_RODARE(
    rodare_record_id=RODARE_RECORD_ID,
    output_dir=download_dir,
)

print("Found checkpoint files:\n", checkpoint_files)
model = ip.load_model(download_dir, model_type="noise_remover")

print("Starting to process")
datafilename = "p3129_r0547_JF4_PPU_thl6.tif"
print("processing file ", datafilename)

runNo = int(datafilename[7:11])

noisy_signal = tifffile.imread(datafilename)
fntiff = f"{datafilename[:-4]}_denoised.tif"

if Path(fntiff).is_file() and SKIP_EXISTING:
    print(f"## Skipping inference for {datafilename} because already done.")
    if DRAW_OUTPUT:
        prediction = mu.loadTiff(fntiff)

else:
    print(f"Doing prediction for {datafilename}   *****************")

    prediction = ip.doit(
        noisy_signal, model, qlow=0.01, qhigh=0.9995
    )  # the main work is done here, the prediction is returned
    mu.saveTiff(prediction, fntiff)  # Storing the results in the tiff file

if DRAW_OUTPUT:  # drawing
    print("Drawing output figure.")
    ip.draw_it(
        noisy_signal,
        prediction,
        runNo=runNo,
        case="",
        fn=datafilename[:-4],
        draw_output=DRAW_OUTPUT,
        ax=ax,
        noise_cmax=noise_cmax,
        out_fig_dir=".",
        cl=cl,
    )
