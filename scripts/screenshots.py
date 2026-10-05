"""Capture README screenshots of the running app (needs: pip install playwright, Google Chrome).

    streamlit run app.py &            # then
    python scripts/screenshots.py [http://localhost:8501]
"""
import glob
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = next((a for a in sys.argv[1:] if a.startswith('http')), 'http://localhost:8501')
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "screens"
FIX = ROOT / "tests" / "fixtures"
OUT.mkdir(parents=True, exist_ok=True)


def open_app(p, width=1500, height=2300):
    browser = p.chromium.launch(channel="chrome", headless=True)
    page = browser.new_page(viewport={"width": width, "height": height})
    page.goto(URL)
    page.wait_for_selector("text=Late-discovery screening aid for cryptic pregnancy", timeout=60000)
    return browser, page


def upload(page, paths, wait_text):
    page.set_input_files("input[type=file]", [str(x) for x in paths])
    page.wait_for_selector(f"text={wait_text}", timeout=600000)
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(4000)


def shot(page, name):
    page.screenshot(path=str(OUT / name), full_page=True)
    print("saved", name)


with sync_playwright() as p:
    # 1 empty state
    b, page = open_app(p)
    page.wait_for_selector("[data-testid=stFileUploaderDropzone]", timeout=60000)
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(2500)
    shot(page, "1_empty_dark.png")
    # 2 good scan (late stage, answer No -> supportive message)
    page.get_by_text("No", exact=True).first.click()
    upload(page, [FIX / "valid_head_late.png"], "Estimated gestational age")
    shot(page, "2_good_scan_dark.png")
    # 3 light theme
    page.get_by_text("Settings").first.click()
    page.get_by_text("Light theme").click()
    page.wait_for_timeout(2500)
    shot(page, "3_good_scan_light.png")
    b.close()
    # 4 poor scan
    b, page = open_app(p)
    upload(page, [FIX / "week12_user_crop.png"], "Estimated gestational age")
    shot(page, "4_poor_scan_dark.png")
    b.close()
    # 5 batch of 12
    b, page = open_app(p)
    files = sorted(glob.glob(str(ROOT.parent / "data" / "test_set" / "*.png")))[:12]
    upload(page, files, "Open ›")
    shot(page, "5_batch_dark.png")
    b.close()
    # 5b the full test_set (335 scans)
    if "--all" in sys.argv:
        b, page = open_app(p, 1500, 1800)
        files = sorted(glob.glob(str(ROOT.parent / "data" / "test_set" / "*.png")))
        upload(page, files, "Open ›")
        shot(page, "7_batch_335_dark.png")
        b.close()
    # 6 mobile width
    b, page = open_app(p, 400, 3600)
    upload(page, [FIX / "valid_head_late.png"], "Estimated gestational age")
    shot(page, "6_mobile_dark.png")
    b.close()
