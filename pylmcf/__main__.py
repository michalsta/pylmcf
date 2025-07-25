import argparse

from .__version__ import __version__, include


def main():
    parser = argparse.ArgumentParser(
        description="Python wrapper for the LMCF C++ library",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the version of the library",
    )
    parser.add_argument(
        "--include",
        action="store_true",
        help="Print the include path for the C++ library",
    )
    args = parser.parse_args()

    if args.include:
        print(include())
    elif args.version:
        print(f"Version: {__version__}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()