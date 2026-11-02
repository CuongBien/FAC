"""Setuptools hook; all project metadata remains in pyproject.toml."""
from pathlib import Path
import runpy
from setuptools import setup

setup(cmdclass={'build_py': runpy.run_path(str(Path(__file__).with_name('build_support.py')))['BuildPy']})
