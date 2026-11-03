"""Export a Go1 or G1 manifest-backed checkpoint to ONNX."""
import argparse
from pathlib import Path
from gobot.rl.export import export_policy

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export_policy(args.checkpoint, args.output)

if __name__ == '__main__':
    main()
