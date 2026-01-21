# import os
import subprocess
import yaml
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).parent.parent
MODELS_YAML = BASE_DIR / "config" / "models.yml"
CACHE_DIR = Path.home() / ".cache" / "agents_manager"

def run_cmd(cmd):
    subprocess.run(cmd, check=True)

def download_file(url, out_path):
    print(f"Downloading: {url} -> {out_path}")
    run_cmd(["curl", "-L", "--retry", "3", "-o", str(out_path), url])

def process_entry(section, item):
    name = item.get("name")
    url = item.get("url")
    if not name or not url: return

    out_dir = CACHE_DIR / section
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle ZIP files (specifically for Vosk)
    if url.endswith(".zip"):
        folder_name = name.replace(".zip", "")
        target_dir = out_dir / folder_name
        
        if target_dir.exists() and target_dir.is_dir():
            print(f"  [SKIP] Already exists: {target_dir}")
            return
        
        # FIX: Force .zip extension to avoid name collision with the extracted folder
        zip_name = name if name.endswith(".zip") else f"{name}.zip"
        zip_path = out_dir / zip_name
        
        download_file(url, zip_path)
        print(f"  Unzipping to {out_dir}...")
        run_cmd(["unzip", "-q", "-o", str(zip_path), "-d", str(out_dir)])
        
        # Clean up zip
        if zip_path.exists():
            zip_path.unlink()
        return

    # Handle Normal files
    target_file = out_dir / name
    if target_file.exists():
        print(f"  [SKIP] Already exists: {target_file}")
        return
    
    download_file(url, target_file)

def main():
    if not MODELS_YAML.exists():
        print(f"Error: {MODELS_YAML} not found")
        return

    print(f"Loading models from {MODELS_YAML}")
    with open(MODELS_YAML, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    for section in ["stt", "wake_word", "tts"]:
        print(f"\n--- Processing {section} ---")
        items = data.get(section)
        if items:
            for item in items:
                process_entry(section, item)
        else:
            print(f"No items found for {section}")

    print(f"\nAll models ready in {CACHE_DIR}")

if __name__ == "__main__":
    main()
