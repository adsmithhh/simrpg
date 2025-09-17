"""
yaml_receiver.py — Clipboard watcher that auto-saves YAML you copy.
- Watches your clipboard for YAML-like text
- Validates by attempting yaml.safe_load (no execution)
- Saves to ~/Documents/yaml_logs as timestamped files
- Deduplicates via SHA-256 content hash
- Never uploads anything; fully local

Usage:
  python yaml_receiver.py
  python yaml_receiver.py --interval 1.0 --dir "D:\\YAMLsink" --prefix "chatgpt_"

Stop with Ctrl+C.

Requires: pyyaml, pyperclip, portalocker
  pip install pyyaml pyperclip portalocker
"""
import argparse
import os
import time
import hashlib
import datetime
from pathlib import Path

import portalocker
import pyperclip
import yaml

DEFAULT_INTERVAL = 1.15  # seconds

def is_probable_yaml(txt: str) -> bool:
    if not txt:
        return False
    # Quick heuristics: typical YAML markers
    if txt.lstrip().startswith(('---', '#', '-', '{', '[')) or ':' in txt:
        # Try safe parse to confirm
        try:
            yaml.safe_load(txt)
            return True
        except Exception:
            return False
    return False

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8', errors='ignore')).hexdigest()

def save_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with portalocker.Lock(str(tmp), 'w', timeout=5) as f:
        f.write(content)
    os.replace(tmp, path)

def next_filename(dir_path: Path, prefix: str) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return dir_path / f"{prefix}{ts}.yaml"

def main():
    parser = argparse.ArgumentParser(description="Auto-save copied YAML from clipboard.")
    parser.add_argument("--dir", default=str(Path.home() / "Documents" / "yaml_logs"),
                        help="Output directory")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL,
                        help="Polling interval seconds")
    parser.add_argument("--prefix", default="yaml_",
                        help="Filename prefix")
    parser.add_argument("--echo", action="store_true",
                        help="Print save events")
    parser.add_argument("--minlen", type=int, default=16,
                        help="Minimum text length to consider")
    args = parser.parse_args()

    outdir = Path(args.dir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    last_hash = None
    last_text = ""

    print(f"[yaml-receiver] Watching clipboard every {args.interval}s. Output → {outdir}")
    print("[yaml-receiver] Press Ctrl+C to stop.")

    try:
        while True:
            try:
                clip = pyperclip.paste()
            except Exception as e:
                clip = ""
            if clip and len(clip) >= args.minlen:
                if clip != last_text:
                    if is_probable_yaml(clip):
                        h = sha256(clip)
                        if h != last_hash:
                            fn = next_filename(outdir, args.prefix)
                            save_atomic(fn, clip)
                            if args.echo:
                                print(f"[saved] {fn.name} ({len(clip)} chars)")
                            last_hash = h
                            last_text = clip
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[yaml-receiver] Stopped.")

if __name__ == "__main__":
    main()
