import os
import urllib.request
import zipfile

CHECKPOINT_URL = (
    "https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization/"
    "releases/download/v1.0/colorization_checkpoint.zip"
)
ZIP_FILENAME = "colorization_checkpoint.zip"

def main():
    """
    Downloads `colorization_checkpoint.zip` from the official repo 
    and extracts it in this folder, 
    creating `checkpoints/video_moredata_l1` and `data/`.
    """
    root_dir = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.join(root_dir, ZIP_FILENAME)

    # 1) Download if missing
    if not os.path.exists(zip_path):
        print(f"Downloading from {CHECKPOINT_URL} ...")
        urllib.request.urlretrieve(CHECKPOINT_URL, zip_path)
        print("Download complete.")
    else:
        print(f"Zip file already exists: {zip_path}")

    # 2) Extract
    print(f"Extracting {zip_path} into {root_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root_dir)
    print("Extraction complete.")

    # Optionally remove the zip after extraction
    # os.remove(zip_path)

    print("colorization_checkpoint files are installed in `checkpoints/` and `data/`.")

if __name__ == "__main__":
    main()
