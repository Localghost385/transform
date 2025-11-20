from __future__ import annotations
import argparse
import logging
import sys
from typing import List
from importlib.metadata import version, PackageNotFoundError

"""Entry point for the midi package.

Usage:
    python -m midi [--version] [-v|--verbose]
"""


try:
    # Python 3.8+
except Exception:
    try:
        # backport for older interpreters
        from importlib_metadata import version, PackageNotFoundError  # type: ignore
    except Exception:  # pragma: no cover - fallback stub
        def version(pkg: str) -> str:
            return "0.0.0"
        class PackageNotFoundError(Exception):
            pass

PACKAGE_NAME = "midi"


def get_version() -> str:
    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError:
        return "0.0.0"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=PACKAGE_NAME, description="midi - command line interface")
    p.add_argument("-v", "--verbose", action="store_true", help="enable verbose logging")
    p.add_argument("--version", action="store_true", help="print version and exit")
    return p


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def run(argv: List[str]) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv[1:])

    if args.version:
        print(get_version())
        return 0

    configure_logging(args.verbose)
    logging.info("Starting midi package")
    # TODO: replace the block below with actual application logic
    try:
        logging.debug("Running main task (placeholder)")
        # ... actual work here ...
        return 0
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        return 2
    except Exception:
        logging.exception("Unhandled exception")
        return 1


def main() -> None:
    raise SystemExit(run(sys.argv))


if __name__ == "__main__":  
    main()