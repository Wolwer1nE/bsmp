#!/usr/bin/env python
"""Tools for building project with CMake on any platform."""
import sys

def is_windows() -> bool:
    return sys.platform.startswith("win")

def is_linux() -> bool:
    return sys.platform.startswith("linux")