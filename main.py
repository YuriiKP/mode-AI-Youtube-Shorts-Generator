"""Thin entry point — delegates to the unified CLI.

All commands live in :mod:`shorts_generator.cli`; this file exists so the
project can be run the familiar way:

    python main.py clip
    python main.py transcribe
    python main.py music
    python main.py subtitles
    python main.py preview
    python main.py all
"""

import sys

from shorts_generator.cli import main

if __name__ == "__main__":
    sys.exit(main())
