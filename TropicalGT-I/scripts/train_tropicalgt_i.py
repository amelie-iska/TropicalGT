#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from tropicalgt.run import train

def main() -> None:
    parser = argparse.ArgumentParser(description="Train TropicalGT-I")
    parser.add_argument("--config", default=str(ROOT / "configs" / "smoke.json"))
    args = parser.parse_args()
    report = train(args.config)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
