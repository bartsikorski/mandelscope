"""PyInstaller entry: console mandelscope-cli.exe (render / video / bench / info)."""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from mandelscope.cli import main
    sys.exit(main())
