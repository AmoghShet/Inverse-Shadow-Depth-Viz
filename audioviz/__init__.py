"""ASCII Ink Visualizer: an ASCII/fluid-based, audio-reactive visualizer (package `audioviz`).

Particles follow a curl-noise flow field, leave an advected trail buffer, and the result is drawn
as ASCII glyphs on black. The song drives the parameters; a seed makes each run different.
"""
import os

__version__ = "0.1.0"

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(PKG_DIR)
SHADER_DIR = os.path.join(PKG_DIR, "shaders")
PRESET_DIR = os.path.join(PKG_DIR, "presets")
CACHE_DIR = os.path.join(ROOT_DIR, ".cache")
OUT_DIR = os.path.join(ROOT_DIR, "out")
