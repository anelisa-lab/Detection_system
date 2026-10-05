"""PDF report with the images, the estimate, the scan summary and the disclaimer."""
from datetime import date

import cv2
import numpy as np
from fpdf import FPDF

from .content import DISCLAIMER


def _txt(s: str) -> str:
    """The built-in PDF fonts are latin-1 only."""
    table = {"–": "-", "—": "-", "±": "+/-", "≈": "~", "’": "'", "→": "->"}
    for k, v in table.items():
        s = s.replace(k, v)
    return s.encode("latin-1", "replace").decode("latin-1")


def _png(rgb_or_gray: np.ndarray) -> bytes:
    img = rgb_or_gray if rgb_or_gray.ndim == 2 else cv2.cvtColor(rgb_or_gray, cv2.COLOR_RGB2BGR)
    return cv2.imencode(".png", img)[1].tobytes()


def build_report(items: list[dict], overall: str | None = None) -> bytes:
    """items: dicts with name, gray, heat (RGB or None), headline, range_text, trimester_text, due_text,
    screen_badge, screen_label, screen_lines (sentence, why, answer, notes), summary (list of paragraphs)."""
    import io

    pdf = FPDF()
    pdf.set_auto_page_break(True, 15)
    for it in items:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.multi_cell(0, 8, _txt("Ultrasound age estimate - research prototype"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 5, _txt(f"File: {it['name']}   Date: {date.today().isoformat()}"), new_x="LMARGIN", new_y="NEXT")
        y = pdf.get_y() + 3
        pdf.image(io.BytesIO(_png(it["gray"])), x=10, y=y, w=60)
        if it.get("heat") is not None:
            pdf.image(io.BytesIO(_png(it["heat"])), x=75, y=y, w=60)
        pdf.set_y(y + 64)
        pdf.set_font("Helvetica", "B", 20)
        pdf.cell(0, 10, _txt(it["headline"]), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        for line in (it.get("range_text"), it.get("trimester_text"), it.get("due_text")):
            if line:
                pdf.multi_cell(0, 5, _txt(line), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 6, _txt("Cryptic pregnancy screening result"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "B", 11)
        pdf.multi_cell(0, 5.5, _txt(f"{it['screen_badge']}: {it['screen_label']}"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        for line in it["screen_lines"]:
            pdf.multi_cell(0, 5, _txt(line), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 6, _txt("What this scan suggests"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        for para in it["summary"]:
            pdf.multi_cell(0, 5, _txt(para), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
        pdf.ln(3)
        pdf.set_font("Helvetica", "I", 9)
        pdf.multi_cell(0, 4.5, _txt(DISCLAIMER), new_x="LMARGIN", new_y="NEXT")
        if overall:
            pdf.multi_cell(0, 4.5, _txt(overall), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
