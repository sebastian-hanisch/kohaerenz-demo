"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mini_xy():
    """x = (1, 2, 3, 4), y = (2, 1, 3, 5), Segmente der Länge 2 ohne Überlappung, Rechteckfenster, rfft-Bins 0 und 1 (unnormiert, mal 2):
    X-Segmente (1, 2) -> (3, -1), (3, 4) -> (7, -1); Y-Segmente (2, 1) -> (3, 1), (3, 5) -> (8, -2).
    S_xx = ((9 + 49)/2, (1 + 1)/2) = (29, 1); S_yy = ((9 + 64)/2, (1 + 4)/2) = (36,5; 2,5); S_xy = ((3·3 + 7·8)/2, ((-1)·1 + (-1)·(-2))/2) = (32,5; 0,5);
    Kohärenz = (32,5² / (29 · 36,5); 0,5² / (1 · 2,5)) = (0,99787; 0,1)."""
    return np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 1.0, 3.0, 5.0])
