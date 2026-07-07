"""Recolor the RegAI RAP logo into the app's dark-brown + orange palette.

Palette target:
    * Background:   deep chocolate brown (#3d2513) so the logo becomes its own
      rounded badge sitting on the hero gradient.
    * Icon + "RegAI" wordmark:  originally deep navy -- lifted to a bright
      amber orange (~#ff9a4a) so it reads on the dark background.
    * "RAP", magnifying glass, chart bars, underline accent: originally
      purple/magenta -- collapsed to bright orange (~#ff8033) for contrast.
    * Tagline body text:  originally near-black gray -- swapped to a bright
      amber orange (~#f2ac52) so every character in the logo shares the
      same orange palette, matching the wordmark for a unified look.
    * Tagline accent words ("Regulatory", "Readiness Assessment"): originally
      red-orange -- left in the warm range, only lightly boosted.

Algorithm (all vectorised in NumPy):

1. Load ``assets/regai_logo_original.png`` and normalise to ``[0, 1]``.
2. Alpha-decompose every pixel against the original *white* background:
   for a pixel ``P``, we solve ``P = a * C + (1 - a) * White`` where ``a`` is
   the pixel's "design coverage" and ``C`` is the pure design colour. This
   is what lets anti-aliased edges recomposite cleanly against a NEW
   background without leaving a white halo.
3. Recolour ``C`` in HSV space:
     - Blue hues -> mid orange (25 deg).
     - Purple/magenta hues -> burnt orange (18 deg).
     - Dark neutrals (tagline gray text) -> bright amber orange so every
       character in the logo shares the same warm palette.
     - Bright colours: mild saturation boost.
     - Anything dark AND colourful: lift value so it reads on the dark BG.
4. Composite the transformed ``C_new`` back over the new dark-brown BG
   using the same coverage ``a`` (so edges are anti-aliased against dark
   brown, not white).
5. Save to ``assets/regai_logo.png`` (backup at
   ``assets/regai_logo_original.png`` is untouched, so this is reversible).

Run:
    python scripts/recolor_logo.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "assets" / "regai_logo_original.png"
DST = PROJECT_ROOT / "assets" / "regai_logo.png"

# --- Target palette ---------------------------------------------------------
# Deep chocolate brown for the badge background. Sits nicely between the
# hero's dark end (#2d2d2d) and its orange end (#d04a02).
BG_COLOR = np.array([0x3d, 0x25, 0x13], dtype=np.float32) / 255.0  # #3d2513

# Target hues used inside the design-colour HSV remap.
HUE_BLUE_TO = 25.0 / 360.0     # bright/mid orange for former navy elements
HUE_PURPLE_TO = 18.0 / 360.0   # burnt orange for former purple/magenta
HUE_TAGLINE = 25.0 / 360.0     # warm hue -- pushed almost desaturated to
                               # give tiny tagline text the strongest
                               # possible contrast against the dark BG.
SAT_TAGLINE = 0.16             # near-white warm cream, not peach
VAL_TAGLINE = 1.00             # maximum lightness for tiny tagline text

# A pixel counts as "coloured" (as opposed to neutral text) if its
# saturation in the pure-colour space exceeds this threshold.
SAT_COLOURED = 0.18

# Every coloured pixel is normalised into a bright, moderately-saturated
# range so text stays highly visible on the dark chocolate BG. This maps
# the original value ``v`` into ``[VAL_LIFT_FLOOR, VAL_LIFT_CEIL]`` and
# clamps saturation into ``[SAT_FLOOR, SAT_CEIL]``.
VAL_LIFT_FLOOR = 0.94
VAL_LIFT_CEIL = 1.00
SAT_FLOOR = 0.28
SAT_CEIL = 0.55


# --- HSV helpers (vectorised) ----------------------------------------------


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    delta = maxc - minc
    s = np.where(maxc > 0, delta / np.where(maxc == 0, 1.0, maxc), 0.0)

    safe_delta = np.where(delta == 0, 1.0, delta)
    rc = (maxc - r) / safe_delta
    gc = (maxc - g) / safe_delta
    bc = (maxc - b) / safe_delta

    h = np.zeros_like(maxc)
    h = np.where((maxc == r) & (delta != 0), bc - gc, h)
    h = np.where((maxc == g) & (delta != 0), 2.0 + rc - bc, h)
    h = np.where((maxc == b) & (delta != 0), 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    h = np.where(delta == 0, 0.0, h)
    return np.stack([h, s, v], axis=-1)


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = np.floor(h * 6.0).astype(int)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i_mod = i % 6

    r = np.choose(i_mod, [v, q, p, p, t, v])
    g = np.choose(i_mod, [t, v, v, q, p, p])
    b = np.choose(i_mod, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


# --- Core transform ---------------------------------------------------------


def _transform(rgb: np.ndarray) -> np.ndarray:
    """Turn the design colour ``C`` (pre-composited) into its dark-BG form."""
    hsv = _rgb_to_hsv(rgb)
    h360 = hsv[..., 0] * 360.0
    s = hsv[..., 1]
    v = hsv[..., 2]

    coloured = s > SAT_COLOURED

    blue = coloured & (h360 >= 200.0) & (h360 <= 270.0)
    purple = coloured & (h360 > 270.0) & (h360 <= 340.0)
    warm = coloured & ~(blue | purple)  # already red/orange -- keep hue

    new_h = hsv[..., 0].copy()
    new_h[blue] = HUE_BLUE_TO
    new_h[purple] = HUE_PURPLE_TO
    # warm hues keep their original hue

    # Normalise saturation into a "light, punchy" band. Uncapped saturation
    # produces deep burnt oranges that read as dark on the chocolate BG --
    # clamping into [SAT_FLOOR, SAT_CEIL] gives a consistent peachy-amber.
    clamped_s = np.clip(s, SAT_FLOOR, SAT_CEIL)
    new_s = np.where(coloured, clamped_s, s)

    # Lift value on EVERY coloured pixel (not only the dark ones) so the
    # whole logo reads as bright/light orange rather than a mix of dark
    # burnt orange and mid orange. Linear map v -> [FLOOR, CEIL].
    lifted_v = VAL_LIFT_FLOOR + v * (VAL_LIFT_CEIL - VAL_LIFT_FLOOR)
    new_v = np.where(coloured, lifted_v, v)

    # Neutral dark pixels (tagline body text) -> bright amber orange so the
    # whole logo shares the same warm palette as the wordmark. Detected as
    # low-saturation AND dark; whites/near-whites are excluded via
    # the value ceiling.
    neutral_dark = (~coloured) & (v < 0.55)
    new_h = np.where(neutral_dark, HUE_TAGLINE, new_h)
    new_s = np.where(neutral_dark, SAT_TAGLINE, new_s)
    new_v = np.where(neutral_dark, VAL_TAGLINE, new_v)

    hsv_out = np.stack([new_h, new_s, new_v], axis=-1)
    return _hsv_to_rgb(hsv_out)


def main() -> None:
    src_img = Image.open(SRC).convert("RGBA")
    arr = np.asarray(src_img, dtype=np.float32) / 255.0
    rgb = arr[..., :3]
    alpha_orig = arr[..., 3:4]  # keep original transparency (usually 1.0)

    # Alpha-decompose against WHITE: coverage a = 1 - min(R, G, B).
    # Pure white -> a = 0 (no design coverage), pure ink -> a = 1.
    coverage = 1.0 - rgb.min(axis=-1, keepdims=True)
    safe_cov = np.where(coverage < 1e-4, 1.0, coverage)
    design_c = (rgb - (1.0 - coverage)) / safe_cov
    design_c = np.clip(design_c, 0.0, 1.0)

    # Recolour the pure design colour, then composite over the new dark BG
    # using the ORIGINAL coverage so edge anti-aliasing stays correct.
    new_c = _transform(design_c)
    composited = coverage * new_c + (1.0 - coverage) * BG_COLOR

    out = np.concatenate([composited, alpha_orig], axis=-1)
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(out, mode="RGBA").save(DST, format="PNG", optimize=True)
    print(f"Wrote {DST} ({DST.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
