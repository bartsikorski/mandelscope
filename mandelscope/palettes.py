"""Colour gradients (cyclic) and curated "looks" (palette + mapping + shading).

Gradients are interpolated in OKLab space, which gives smooth, even transitions
without the muddy mid-tones of plain RGB interpolation.
"""
from __future__ import annotations

import colorsys
import math
from functools import lru_cache

import numpy as np


def _hex(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


# name -> list of (position, hex colour); positions in [0,1), cyclic
PALETTES: dict[str, list[tuple[float, str]] | None] = {
    "Gilded": [(0.00, "#03050f"), (0.12, "#0b1f45"), (0.26, "#1f5a99"), (0.38, "#8fc3e8"),
               (0.46, "#fbf3d8"), (0.55, "#f2c14e"), (0.68, "#b3651b"), (0.82, "#4a1d08"),
               (0.93, "#120705")],
    "Nebula": [(0.00, "#05020f"), (0.14, "#2a0b4a"), (0.30, "#7a1f7d"), (0.44, "#e0457b"),
               (0.54, "#ffb38a"), (0.62, "#fff4e6"), (0.72, "#7fd6e6"), (0.84, "#2b5ea8"),
               (0.94, "#0c1438")],
    "Porcelain": [(0.00, "#fbf8f2"), (0.25, "#e8eef5"), (0.45, "#9fb9d8"), (0.60, "#2f5f9e"),
                  (0.72, "#0d2347"), (0.85, "#5e86b8"), (0.95, "#dbe5ef")],
    "Ember": [(0.00, "#050000"), (0.15, "#3b0505"), (0.32, "#a3200b"), (0.48, "#f26a1b"),
              (0.60, "#ffc65c"), (0.68, "#fff3c4"), (0.80, "#d9822b"), (0.92, "#4a0d03")],
    "Aurora": [(0.00, "#020d12"), (0.18, "#08424a"), (0.36, "#13a38a"), (0.50, "#b6f28a"),
               (0.60, "#f4ffd9"), (0.72, "#ef8fd0"), (0.86, "#5b2a86"), (0.95, "#140a2b")],
    "Glacier": [(0.00, "#010812"), (0.20, "#0a2f57"), (0.40, "#3b8cc9"), (0.55, "#bfe6ff"),
                (0.62, "#ffffff"), (0.75, "#7fb2d9"), (0.90, "#123b63")],
    "Copper & Teal": [(0.00, "#04161a"), (0.20, "#0f4f56"), (0.38, "#3fa39d"), (0.50, "#e9f1e3"),
                      (0.60, "#f3c79a"), (0.74, "#c2652c"), (0.88, "#4c1a0b")],
    # v3.8: warmer, so it no longer looks like Nebula
    "Sunset": [(0.00, "#160812"), (0.14, "#4f1530"), (0.30, "#a8322e"), (0.44, "#ec7a2c"),
               (0.55, "#ffc96b"), (0.62, "#fff1d2"), (0.74, "#c66a7c"), (0.87, "#5a2a5e"),
               (0.95, "#22102c")],
    "Ink & Gold": [(0.00, "#000000"), (0.30, "#1c1c1c"), (0.48, "#e9e4d8"), (0.56, "#ffffff"),
                   (0.66, "#c9a14a"), (0.80, "#3a2c10")],
    "Neon": [(0.00, "#03010d"), (0.22, "#0d0838"), (0.40, "#2a0f6e"), (0.50, "#00d5ff"),
             (0.56, "#e6fdff"), (0.63, "#ff38c8"), (0.76, "#3a0a5c"), (0.90, "#06031a")],
    "Sand": [(0.00, "#f4ead8"), (0.22, "#dcc29a"), (0.42, "#a87a4c"), (0.55, "#5b3a22"),
             (0.68, "#c49a6c"), (0.82, "#f3e7d0"), (0.93, "#e6d2b0")],
    # v3.8
    "Silk": [(0.00, "#1a1030"), (0.20, "#4a2a6a"), (0.40, "#b0607a"), (0.55, "#f2c6a0"),
             (0.68, "#fff4e4"), (0.82, "#6aa6b0"), (0.93, "#1c3a4a")],
    "Blueprint": [(0.00, "#061738"), (0.30, "#0c2c66"), (0.55, "#1d4f9e"), (0.80, "#5d8fd6"),
                  (0.92, "#dbe9ff")],
    "Graphite": [(0.00, "#0a0a0b"), (0.25, "#3a3a3c"), (0.50, "#b8b8b4"), (0.62, "#f4f3ef"),
                 (0.80, "#6e6e6c"), (0.93, "#1c1c1d")],
    "Midnight": [(0.00, "#02040c"), (0.30, "#081430"), (0.55, "#12284f"), (0.75, "#0a1838"),
                 (0.92, "#040917")],
    "Opal": [(0.00, "#f8f3ff"), (0.17, "#cfe6f7"), (0.33, "#bdf0dc"), (0.50, "#f8e6c4"),
             (0.67, "#f5c9df"), (0.83, "#d9cff9")],
    "Classic": [(0.0, "#000764"), (0.16, "#206bcb"), (0.42, "#edffff"),
                (0.6425, "#ffaa00"), (0.8575, "#000200")],
    "Twilight": [(0.0, "#e2d9e2"), (0.25, "#6a7cc6"), (0.5, "#2f1436"),
                 (0.75, "#b24f45"), (0.95, "#e0cfc6")],
    "Psychedelic": None,   # generated (HSV rainbow)
    "Grayscale": [(0.0, "#000000"), (0.5, "#ffffff")],
    # kept for compatibility with saved settings of version 1
    "Gold & Blue": [(0.0, "#04102a"), (0.2, "#1c4f99"), (0.38, "#8fd3ff"),
                    (0.5, "#fff6d5"), (0.62, "#f2b134"), (0.8, "#7a2e05"), (0.92, "#120400")],
    "Fire": [(0.0, "#000000"), (0.25, "#7a0000"), (0.5, "#ff5a00"),
             (0.75, "#ffe066"), (0.9, "#ffffff")],
    "Ocean": [(0.0, "#001018"), (0.3, "#00507a"), (0.55, "#3fb8c8"),
              (0.75, "#e8fbff"), (0.9, "#0a2a40")],
    "Emerald": [(0.0, "#001a0f"), (0.3, "#0f7a4a"), (0.55, "#c8f7c5"),
                (0.75, "#f5e6a8"), (0.9, "#2b1a05")],
}

# A look = palette + colour mapping + shading.  mapping 3 = auto-contrast (histogram).
LOOKS: dict[str, dict] = {
    "Gilded": dict(palette="Gilded", mapping=3, auto_cycles=5.0, density=0.03, de_strength=0.55,
                   de_scale=1.0, slope_strength=0.35, light_angle=135.0, light_height=1.6,
                   interior=(0.01, 0.01, 0.03), glow=0.0),
    # v3.8: stripe-average texture - feathery, silky bands in jewel tones
    "Silk": dict(palette="Silk", mapping=3, auto_cycles=2.0, density=0.03, stripes=1.0, stripe_freq=5.0,
                 de_strength=0.35, de_scale=1.0, slope_strength=0.25, light_angle=120.0, light_height=1.8,
                 interior=(0.03, 0.02, 0.05), glow=0.0),
    "Nebula": dict(palette="Nebula", mapping=3, auto_cycles=4.0, density=0.03, de_strength=0.35,
                   de_scale=1.0, slope_strength=0.0, interior=(0.0, 0.0, 0.0), glow=0.35,
                   glow_color=(1.0, 0.88, 0.95)),
    "Sunset": dict(palette="Sunset", mapping=3, auto_cycles=4.0, density=0.03, de_strength=0.5,
                   de_scale=1.0, slope_strength=0.3, light_angle=135.0, light_height=1.5,
                   interior=(0.03, 0.01, 0.03), glow=0.0),
    "Ember": dict(palette="Ember", mapping=3, auto_cycles=6.0, density=0.03, de_strength=0.6,
                  de_scale=1.0, slope_strength=0.4, light_angle=135.0, light_height=1.5,
                  interior=(0.0, 0.0, 0.0), glow=0.0),
    "Copper": dict(palette="Copper & Teal", mapping=3, auto_cycles=5.0, density=0.03, de_strength=0.5,
                   de_scale=1.0, slope_strength=0.5, light_angle=135.0, light_height=1.5,
                   interior=(0.01, 0.02, 0.02), glow=0.0),
    "Aurora": dict(palette="Aurora", mapping=3, auto_cycles=4.0, density=0.03, de_strength=0.5,
                   de_scale=1.0, slope_strength=0.2, light_angle=60.0, light_height=1.8,
                   interior=(0.0, 0.02, 0.03), glow=0.0),
    "Glacier": dict(palette="Glacier", mapping=3, auto_cycles=5.0, density=0.03, de_strength=0.6,
                    de_scale=1.0, slope_strength=0.45, light_angle=60.0, light_height=1.4,
                    interior=(0.0, 0.01, 0.03), glow=0.0),
    # v3.8: dark navy ground, the filaments glow gold
    "Midnight": dict(palette="Midnight", mapping=3, auto_cycles=2.0, density=0.03, de_strength=0.0,
                     de_scale=0.5, slope_strength=0.25, light_angle=135.0, light_height=1.5,
                     glow=0.85, glow_color=(1.0, 0.80, 0.42), interior=(0.0, 0.0, 0.0)),
    # v3.8: colour by distance to the set (contours hug it) with a faint drafting grid
    "Blueprint": dict(palette="Blueprint", method=1, de_density=0.3, mapping=3, auto_cycles=3.0,
                      de_strength=0.0, de_scale=0.7, slope_strength=0.0, pattern=3, pattern_strength=0.35,
                      pattern_freq=4.0, glow=0.5, glow_color=(0.86, 0.93, 1.0), interior=(0.02, 0.07, 0.17)),
    "Porcelain": dict(palette="Porcelain", mapping=3, auto_cycles=3.0, density=0.03, de_strength=0.85,
                      de_scale=0.6, slope_strength=0.25, light_angle=120.0, light_height=2.0,
                      interior=(0.05, 0.12, 0.25), glow=0.0),
    # v3.8: pastel mother-of-pearl with faint field lines
    "Opal": dict(palette="Opal", mapping=3, auto_cycles=2.5, density=0.03, de_strength=0.6, de_scale=0.8,
                 slope_strength=0.35, light_angle=110.0, light_height=1.6, pattern=2, pattern_strength=0.15,
                 pattern_freq=2.0, interior=(0.05, 0.04, 0.09), glow=0.0),
    "Relief": dict(palette="Sand", mapping=3, auto_cycles=2.0, density=0.03, de_strength=0.25,
                   de_scale=1.0, slope_strength=0.9, light_angle=135.0, light_height=0.9,
                   interior=(0.12, 0.09, 0.06), glow=0.0),
    # v3.8: black-and-white print with strong relief light
    "Graphite": dict(palette="Graphite", mapping=3, auto_cycles=3.0, density=0.03, de_strength=0.6,
                     de_scale=1.0, slope_strength=0.7, light_angle=135.0, light_height=1.2,
                     interior=(0.02, 0.02, 0.02), glow=0.0),
    "Ink": dict(palette="Ink & Gold", mapping=3, auto_cycles=6.0, density=0.03, de_strength=0.7,
                de_scale=1.0, slope_strength=0.0, interior=(0.0, 0.0, 0.0), glow=0.0),
    "Neon wire": dict(palette="Neon", mapping=3, auto_cycles=3.0, density=0.03, de_strength=0.8,
                      de_scale=0.6, slope_strength=0.0, interior=(0.0, 0.0, 0.0), glow=0.28,
                      glow_color=(0.35, 0.95, 1.0)),
    "Classic": dict(palette="Classic", mapping=3, auto_cycles=6.0, density=0.05, de_strength=0.0,
                    de_scale=1.0, slope_strength=0.0, interior=(0.0, 0.0, 0.0), glow=0.0),
}
DEFAULT_LOOK = "Gilded"


def palette_names() -> list[str]:
    return [n for n in PALETTES if n not in ("Gold & Blue", "Fire", "Ocean", "Emerald")]


def look_names() -> list[str]:
    return list(LOOKS.keys())


# ----------------------------------------------------------------- OKLab
def _srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def _rgb_to_oklab(rgb):
    lin = _srgb_to_linear(rgb)
    M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                   [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    lms = np.cbrt(lin @ M1.T)
    M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                   [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    return lms @ M2.T


def _oklab_to_rgb(lab):
    M2i = np.array([[1.0, 0.3963377774, 0.2158037573],
                    [1.0, -0.1055613458, -0.0638541728],
                    [1.0, -0.0894841775, -1.2914855480]])
    lms = (lab @ M2i.T) ** 3
    M1i = np.array([[4.0767416621, -3.3077115913, 0.2309699292],
                    [-1.2684380046, 2.6097574011, -0.3413193965],
                    [-0.0041960863, -0.7034186147, 1.7076147010]])
    return _linear_to_srgb(lms @ M1i.T)


# ----------------------------------------------------------------- user gradients
# name -> list of (position, hex); edited in the gradient editor, saved in the user config
CUSTOM: dict[str, list[tuple[float, str]]] = {}
_VERSIONS: dict[str, int] = {}


def palette_version(name: str) -> int:
    return _VERSIONS.get(name, 0)


def palette_stops(name: str) -> list[tuple[float, str]] | None:
    if name in CUSTOM:
        return CUSTOM[name]
    return PALETTES.get(name, PALETTES["Gilded"])


def set_custom(name: str, stops: list[tuple[float, str]]):
    CUSTOM[name] = sorted([(float(p) % 1.0, str(c)) for p, c in stops], key=lambda t: t[0])
    _VERSIONS[name] = _VERSIONS.get(name, 0) + 1


def delete_custom(name: str):
    CUSTOM.pop(name, None)
    _VERSIONS[name] = _VERSIONS.get(name, 0) + 1


def all_palette_names() -> list[str]:
    return palette_names() + [n for n in CUSTOM if n not in PALETTES]


def to_hex(rgb) -> str:
    r, g, b = (int(round(max(0.0, min(1.0, float(v))) * 255)) for v in rgb[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def from_hex(h: str):
    return _hex(h)


def random_stops(seed: int | None = None, n: int | None = None) -> list[tuple[float, str]]:
    """A pleasant random cyclic gradient: a dark anchor, one bright highlight and
    hues spread around a random base hue (built in OKLCh for even lightness steps)."""
    rng = np.random.default_rng(seed)
    n = int(n or rng.integers(5, 8))
    base = rng.uniform(0, 360)
    spread = rng.choice([40, 90, 150, 200])
    pos = (np.arange(n) + rng.uniform(-0.3, 0.3, n)) / n      # evenly spread, jittered
    pos[0] = 0.0
    pos = np.clip(pos, 0.0, 0.999)
    stops = []
    dark = int(rng.integers(0, n))
    light = (dark + n // 2) % n
    for i in range(n):
        if i == dark:
            L, C = rng.uniform(0.06, 0.16), rng.uniform(0.02, 0.06)
        elif i == light:
            L, C = rng.uniform(0.9, 0.98), rng.uniform(0.02, 0.06)
        else:
            L, C = rng.uniform(0.35, 0.8), rng.uniform(0.08, 0.17)
        h = math.radians(base + spread * (i / max(n - 1, 1) - 0.5) + rng.normal(0, 12))
        lab = np.array([[L, C * math.cos(h), C * math.sin(h)]])
        stops.append((float(pos[i]), to_hex(_oklab_to_rgb(lab)[0])))
    return stops


@lru_cache(maxsize=64)
def _make_palette_cached(name: str, n: int, version: int) -> bytes:
    stops = palette_stops(name)
    t = np.arange(n, dtype=np.float64) / n
    if stops is None:
        rgb = np.array([colorsys.hsv_to_rgb(x, 0.85, 1.0) for x in t])
        return rgb.astype(np.float32).tobytes()
    pos = np.array([p for p, _ in stops] + [stops[0][0] + 1.0])
    cols = np.array([_hex(c) for _, c in stops] + [_hex(stops[0][1])])
    lab = _rgb_to_oklab(cols)
    tt = np.where(t < pos[0], t + 1.0, t)
    out = np.zeros((n, 3))
    for ch in range(3):
        out[:, ch] = np.interp(tt, pos, lab[:, ch])
    rgb = _oklab_to_rgb(out)
    return np.clip(rgb, 0, 1).astype(np.float32).tobytes()


def make_palette(name: str, n: int = 1024) -> np.ndarray:
    """Return float32 array (n, 3) for a cyclic gradient."""
    return np.frombuffer(_make_palette_cached(name, n, palette_version(name)),
                         dtype=np.float32).reshape(n, 3).copy()
