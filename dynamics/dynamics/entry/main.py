import argparse
from .scratch import run_scratch
from .pretrained import run_pretrained


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamics training entrypoints")
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("scratch", help="Use scratch workflow")
    sub.add_parser("pretrained", help="Use pretrained workflow")
    args, remaining = parser.parse_known_args()

    if args.mode == "scratch":
        run_scratch(remaining)
    else:
        run_pretrained(remaining)
