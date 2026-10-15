#!/usr/bin/env python3
"""Convenience launcher: `python run.py` (no need to remember uvicorn flags)."""

import os
import sys

if sys.version_info < (3, 10):
    sys.exit(
        "The Fruit Detection backend requires Python 3.10 or newer, but you are "
        f"running Python {sys.version.split()[0]}.\n"
        "Use the project virtual environment instead, for example:\n"
        "    .venv/bin/python run.py\n"
        "(On Windows: .venv\\Scripts\\python.exe run.py)"
    )

import uvicorn


def main() -> None:
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "1") == "1",
    )


if __name__ == "__main__":
    main()
