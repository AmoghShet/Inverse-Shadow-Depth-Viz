"""Depth Fog (package `depthviz`): the webcam's monocular depth map drawn as a grainy fog photograph.

A depth model (Depth Anything V2 small or MiDaS v2.1 small, through onnxruntime) turns each camera
frame into relative depth; near things go dark and the far room dissolves into white haze (or the
inverse). The song drives fog density, exposure, light, grain and polarity flips; a seed makes every
run different. Separate from the ASCII Ink Visualizer (`audioviz`), whose unchanged infrastructure
(GL context, audio analysis/player, HUD text atlas, live recorder) it imports.
"""
import os

__version__ = "0.1.0"

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(PKG_DIR)
SHADER_DIR = os.path.join(PKG_DIR, "shaders")
PRESET_DIR = os.path.join(PKG_DIR, "presets")
MODEL_DIR = os.path.join(ROOT_DIR, "models", "depth")
OUT_DIR = os.path.join(ROOT_DIR, "out", "depthviz")
CACHE_DIR = os.path.join(ROOT_DIR, ".cache")
TITLE = "Depth Fog"
