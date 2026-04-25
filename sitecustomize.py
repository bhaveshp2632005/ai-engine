"""
sitecustomize.py — Ensures ai-engine dir is always on sys.path.
Python imports this automatically at startup, including uvicorn subprocesses.
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
