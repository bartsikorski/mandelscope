"""PyInstaller entry: windowed Mandelscope.exe."""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from mandelscope.launcher import main
    sys.exit(main())
