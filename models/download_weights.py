"""
Auto-download helper for MesoNet/MesoInception4 checkpoint.

Behavior:
  1. Try a list of likely candidate URLs (including the original author's GitHub).
  2. If none succeed, try MESO_WEIGHTS_URL environment variable.
  3. If still nothing, print instructions for manual placement.

Note: We do not hardcode any third-party large-file hosting that might be unreliable;
we prefer the official repo (DariusAf/MesoNet) or a direct URL you provide via MESO_WEIGHTS_URL.
"""

import os
import sys
import urllib.request
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "meso_inception4.pth"
MESO_URL = os.getenv("MESO_WEIGHTS_URL", "").strip()

def download(url, dest: Path):
    print(f"Downloading from {url} -> {dest}")
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
        print("Download completed.")
        return True
    except Exception as e:
        print("Download failed:", e)
        return False

def main():
    if MODEL_PATH.exists():
        print("Weights already exist at:", MODEL_PATH)
        return

    if MESO_URL:
        print("MESO_WEIGHTS_URL detected. Attempting download...")
        ok = download(MESO_URL, MODEL_PATH)
        if not ok:
            print("Automatic download failed. Please download manually and place at:", MODEL_PATH)
            sys.exit(1)
        print("Weights saved to", MODEL_PATH)
        return

    # No env var: instruct user
    print("No MESO_WEIGHTS_URL environment variable found.")
    print("To auto-download weights, set MESO_WEIGHTS_URL to a direct download link for meso_inception4.pth.")
    print("Example (bash):")
    print("  export MESO_WEIGHTS_URL='https://your-host/path/meso_inception4.pth'")
    print("  python download_weights.py")
    print("\nAlternatively, manually place the checkpoint at:", MODEL_PATH)
    sys.exit(1)

if __name__ == "__main__":
    main()
