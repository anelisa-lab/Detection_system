"""Screenshots of the real app for chosen FETAL_PLANES images (needs the app running and playwright + Chromium).

    streamlit run app.py --server.port 8599 --server.headless true &
    python scripts/eval_fetal_planes_screens.py --out eval_fetal_planes/examples NAME=PATH[:ANSWER] ...
"""
import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8599")
    ap.add_argument("--out", default="eval_fetal_planes/examples")
    ap.add_argument("--chrome", default=None, help="path to a Chromium/Chrome binary")
    ap.add_argument("items", nargs="+", help="NAME=IMAGE_PATH[:ANSWER], ANSWER is Yes, No or Not sure")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.chrome, headless=True)
        for item in args.items:
            name, rest = item.split("=", 1)
            path, _, answer = rest.partition(":")
            page = browser.new_page(viewport={"width": 1500, "height": 1000})
            page.goto(args.url)
            page.wait_for_selector("text=Did you know you were pregnant", timeout=90000)
            if answer:
                page.get_by_text(answer, exact=True).first.click()
            page.set_input_files("input[type=file]", path)
            page.wait_for_selector("text=Cryptic pregnancy screening result", timeout=180000)
            page.wait_for_timeout(1500)
            page.screenshot(path=str(Path(args.out) / f"{name}.png"))
            page.close()
        browser.close()


if __name__ == "__main__":
    main()
