"""Compatibility import for historical validation scripts."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mothra_runtime import DeduplicatingAdam
