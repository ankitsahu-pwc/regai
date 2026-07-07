"""Capture a preview showing the new RegAI RAP logo in the header of every page.

Assumes ``streamlit run app.py`` is already serving on ``http://127.0.0.1:8501``.
Renders each of the 5 workflow pages in a headless Chromium, crops the top of
each page (the shared hero block), stacks them into a single labeled image and
writes it to ``_previews/header_logo_all_pages.png``.

Run:
    python scripts/preview_header_logo.py
"""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREVIEW_DIR = PROJECT_ROOT / "_previews"
PREVIEW_DIR.mkdir(exist_ok=True)

APP_URL = "http://127.0.0.1:8501"
PAGES = [
    "1. Setup",
    "2. Generate BRD / FRD",
    "3. Questionnaire",
    "4. Dashboard",
    "5. Export",
]

# Height of each per-page slice (in device px). Large enough to always contain
# the hero block plus a hint of the page content underneath it.
CROP_HEIGHT = 360
LABEL_HEIGHT = 34
VIEWPORT = {"width": 1440, "height": 900}


def _wait_for_hero(page) -> None:
    page.wait_for_selector(".pwc-hero", state="visible", timeout=20000)
    page.wait_for_selector(".pwc-hero-logo img", state="visible", timeout=20000)
    page.wait_for_load_state("networkidle")


def _pick_page(page, label: str) -> None:
    # The sidebar nav is a Streamlit radio. Clicking the label text switches
    # ``st.session_state["page"]`` which triggers a rerun.
    locator = page.get_by_text(label, exact=True).first
    locator.scroll_into_view_if_needed()
    locator.click()
    page.wait_for_load_state("networkidle")
    _wait_for_hero(page)


def _screenshot_top(page) -> Image.Image:
    png = page.screenshot(
        clip={"x": 0, "y": 0, "width": VIEWPORT["width"], "height": CROP_HEIGHT},
        type="png",
    )
    return Image.open(io.BytesIO(png)).convert("RGB")


def _label_font(size: int = 20) -> ImageFont.ImageFont:
    for candidate in ("segoeuib.ttf", "arialbd.ttf", "seguisb.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    slices: list[tuple[str, Image.Image]] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            context = browser.new_context(viewport=VIEWPORT, device_scale_factor=1)
            page = context.new_page()
            page.goto(APP_URL, wait_until="networkidle")
            _wait_for_hero(page)

            for label in PAGES:
                _pick_page(page, label)
                slices.append((label, _screenshot_top(page)))
        finally:
            browser.close()

    font = _label_font()
    small_font = _label_font(size=14)

    per_slice_h = CROP_HEIGHT + LABEL_HEIGHT
    header_h = 90
    total_h = header_h + per_slice_h * len(slices)
    canvas = Image.new("RGB", (VIEWPORT["width"], total_h), (250, 246, 240))
    draw = ImageDraw.Draw(canvas)

    draw.rectangle(
        [(0, 0), (VIEWPORT["width"], header_h)],
        fill=(45, 45, 45),
    )
    draw.text(
        (28, 20),
        "RegAI RAP - new header logo, live on all five workflow pages",
        fill=(255, 255, 255),
        font=_label_font(size=24),
    )
    draw.text(
        (28, 56),
        "Captured from http://127.0.0.1:8501 via headless Chromium",
        fill=(215, 200, 185),
        font=small_font,
    )

    y = header_h
    for label, img in slices:
        draw.rectangle(
            [(0, y), (VIEWPORT["width"], y + LABEL_HEIGHT)],
            fill=(208, 74, 2),
        )
        draw.text((20, y + 7), label, fill=(255, 255, 255), font=font)
        canvas.paste(img, (0, y + LABEL_HEIGHT))
        y += per_slice_h

    out_path = PREVIEW_DIR / "header_logo_all_pages.png"
    canvas.save(out_path, format="PNG", optimize=True)
    print(f"Wrote preview: {out_path}")

    top_only = PREVIEW_DIR / "header_logo_page1_top.png"
    slices[0][1].save(top_only, format="PNG", optimize=True)
    print(f"Wrote preview: {top_only}")


if __name__ == "__main__":
    main()
