import os
import shutil
import urllib.request
from pathlib import Path

HIGGS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
HEPMASS_BASE_URL = "https://mlphysics.ics.uci.edu/data/hepmass"

def _download_file(url: str, target_path: Path, overwrite: bool = False) -> Path:
    """
    Download a file to target_path using aria2c if available, otherwise fall
    back to urllib with a progress display.
    """
    import shutil
    import subprocess
    import sys
    import time
    import urllib.request

    target_path = target_path.expanduser().resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists() and overwrite:
        target_path.unlink()

    if target_path.exists():
        return target_path

    # ------------------------------------------------------------
    # Fast path: use aria2c if installed
    # ------------------------------------------------------------
    aria2c_path = shutil.which("aria2c")
    if aria2c_path is not None:
        cmd = [
            aria2c_path,
            "-x", "8",          # max connections per server
            "-s", "8",          # split into 8 segments
            "-k", "1M",         # segment size
            "--dir", str(target_path.parent),
            "--out", target_path.name,
            url,
        ]

        print(f"  Downloading with aria2c: {url}")
        print(f"  Saving to               : {target_path}")

        try:
            subprocess.run(cmd, check=True)
            if target_path.exists():
                return target_path
        except subprocess.CalledProcessError as e:
            print(f"  aria2c failed, falling back to urllib: {e}")

    # ------------------------------------------------------------
    # Fallback: urllib with progress
    # ------------------------------------------------------------
    tmp_path = target_path.with_suffix(target_path.suffix + ".part")
    chunk_size = 1024 * 1024  # 1 MB

    try:
        # Preprocess: Does the final file look completely downloaded?
        if target_path.exists():
            final_size = target_path.stat().st_size
            try:
                # Ask the server how big the file is meant to be
                req_head = urllib.request.Request(url, method="HEAD")
                with urllib.request.urlopen(req_head) as response:
                    server_size = response.headers.get("Content-Length")
                    if server_size is not None and int(server_size) == final_size:
                        print(f"  Skipping (already downloaded): {target_path}")
                        return target_path
            except Exception as e:
                print(f"  Could not verify size of existing {target_path}: {e}")
                
        # Check if we should resume from a .part
        downloaded = 0
        mode = "wb"
        headers = {}
        if tmp_path.exists():
            downloaded = tmp_path.stat().st_size
            mode = "ab"
            headers["Range"] = f"bytes={downloaded}-"
            print(f"  Resuming {url} from {downloaded / 1024**2:,.1f} MB...")
        else:
            print(f"  Downloading with urllib: {url}")
            
        print(f"  Saving to               : {target_path}")

        req = urllib.request.Request(url, headers=headers)
        
        try:
            with urllib.request.urlopen(req) as response:
                total_size = response.headers.get("Content-Length")
                # If Range request succeeds, Content-Length is the *remaining* size
                total_size = int(total_size) + downloaded if total_size is not None else None

                start_time = time.time()
                last_print = 0.0

                with open(tmp_path, mode) as out_file:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break

                        out_file.write(chunk)
                        downloaded += len(chunk)

                        now = time.time()
                        if now - last_print >= 0.2:
                            elapsed = max(now - start_time, 1e-9)
                            # Speed is based only on the newly downloaded bytes
                            new_bytes = downloaded - (tmp_path.stat().st_size if mode == "ab" else 0)
                            speed = new_bytes / elapsed

                            if total_size is not None and total_size > 0:
                                pct = 100.0 * downloaded / total_size
                                remaining = max(total_size - downloaded, 0)
                                eta = remaining / speed if speed > 0 else float("inf")

                                print(
                                    f"\r  Progress: {pct:6.2f}% "
                                    f"({downloaded / 1024**2:,.1f}/{total_size / 1024**2:,.1f} MB) "
                                    f"| {speed / 1024**2:,.2f} MB/s "
                                    f"| ETA {eta:,.1f}s",
                                    end="",
                                    flush=True,
                                )
                            else:
                                print(
                                    f"\r  Downloaded: {downloaded / 1024**2:,.1f} MB "
                                    f"| {speed / 1024**2:,.2f} MB/s",
                                    end="",
                                    flush=True,
                                )
                            last_print = now
                            
        except urllib.error.HTTPError as e:
            # 416 means Requested Range Not Satisfiable (i.e. we likely already downloaded it fully)
            if e.code == 416 and mode == "ab":
                print("\n  Server returned 416 (Range Not Satisfiable). Assuming download is already complete.")
            else:
                raise

        tmp_path.replace(target_path)
        print("\n  Download complete.")
        return target_path

    except Exception as e:
        print(f"\n  Download failed: {e}")
        # We NO LONGER delete tmp_path so the user can resume it!
        raise


def _resolve_or_download(
    path_like: str | os.PathLike | None,
    *,
    env_var: str | None = None,
    default_relative: str,
    download_url: str | None = None,
    filename_hint: str | None = None,
    auto_download: bool = True,
    overwrite: bool = False,
) -> Path:
    """
    Resolve dataset path in this order:
      1. explicit path
      2. environment variable
      3. default path relative to loader.py
      4. recursive search under the default folder
      5. auto-download to the default path (if enabled)
    """
    if path_like is None and env_var is not None:
        path_like = os.environ.get(env_var)

    if path_like is not None:
        path = Path(path_like).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            return path

        if auto_download and download_url is not None:
            return _download_file(download_url, path, overwrite=overwrite)

        raise FileNotFoundError(f"Dataset file not found: {path} - please run download.py first.")

    base_dir = Path(__file__).resolve().parent
    default_path = (base_dir / default_relative).resolve()
    default_path.parent.mkdir(parents=True, exist_ok=True)

    if default_path.exists():
        return default_path

    if filename_hint is not None and default_path.parent.exists():
        matches = sorted(default_path.parent.rglob(filename_hint))
        if matches:
            return matches[0].resolve()

    if auto_download and download_url is not None:
        return _download_file(download_url, default_path, overwrite=overwrite)

    raise FileNotFoundError(f"Missing dataset file.\nExpected file: {default_path}\nPlease run python datasets/download.py")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download large datasets for QKernel Benchmark")
    parser.add_argument("--higgs", action="store_true", help="Download the HIGGS dataset (~2.8GB)")
    parser.add_argument("--hepmass", type=str, choices=["1000", "not1000", "all"], help="Download a specific HEPMASS variant")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    
    args = parser.parse_args()

    if not any([args.higgs, args.hepmass, args.all]):
        parser.print_help()
        return

    if args.higgs or args.all:
        print("\n--- Downloading HIGGS Dataset ---")
        _resolve_or_download(
            None, 
            default_relative="HIGGS/HIGGS.csv.gz", 
            download_url=HIGGS_URL, 
            auto_download=True, 
            overwrite=args.overwrite
        )

    hepmass_variant = args.hepmass if args.hepmass else ("all" if args.all else None)
    if hepmass_variant:
        print(f"\n--- Downloading HEPMASS ({hepmass_variant}) Dataset ---")
        train_filename = f"{hepmass_variant}_train.csv"
        test_filename = f"{hepmass_variant}_test.csv"
        
        train_url = f"{HEPMASS_BASE_URL}/{train_filename}"
        test_url = f"{HEPMASS_BASE_URL}/{test_filename}"
        
        _resolve_or_download(
            None,
            default_relative=f"HEPMASS/{train_filename}",
            download_url=train_url,
            auto_download=True,
            overwrite=args.overwrite
        )
        
        _resolve_or_download(
            None,
            default_relative=f"HEPMASS/{test_filename}",
            download_url=test_url,
            auto_download=True,
            overwrite=args.overwrite
        )
        
    print("\nDownloads Complete.")

if __name__ == "__main__":
    main()
