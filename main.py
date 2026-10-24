"""Neural Observatory entrypoint. Original program is preserved in legacy/."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Neural Observatory: manual neural networks and measured compression")
    parser.add_argument("action", choices=["train", "pure", "build", "infer", "test"], nargs="?")
    args, rest = parser.parse_known_args()
    root = Path(__file__).resolve().parent
    if args.action is None:
        parser.print_help()
        return
    if args.action == "test":
        command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v", *rest]
    else:
        script = {"train":"train_suite.py", "pure":"train_pure.py", "build":"build_observatory.py", "infer":"infer.py"}[args.action]
        command = [sys.executable, str(root / "scripts" / script), *rest]
    raise SystemExit(subprocess.call(command, cwd=root))


if __name__ == "__main__":
    main()
