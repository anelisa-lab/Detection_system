"""Presentation layer for the workstation UI: palettes, CSS loading and reusable HTML cards.

Everything here returns HTML strings or loads CSS; app.py decides what to show. Status is never
shown by colour alone: every badge carries an icon and text.
"""
import base64
import html
from pathlib import Path

import cv2
import numpy as np

from .growth import fmt_weeks_days, hadlock_hc_mm

# --- palettes (the contrast test in tests/test_ui_contrast.py checks these pairs) -------------------

DARK = {
    "bg": "#0E1116", "surface": "#161B22", "surface2": "#1C232C", "border": "#2A323D",
    "text": "#E6EDF3", "muted": "#9AA7B4", "accent": "#2FA58B", "on_accent": "#04120E",
    "good": "#2FA58B", "limited": "#E3A93B", "poor": "#F0645C", "grey": "#9AA7B4", "info": "#6EA8FF",
    "canvas": "#05070A", "shadow": "none",
}
LIGHT = {
    "bg": "#F3F6FA", "surface": "#FFFFFF", "surface2": "#F6F8FB", "border": "#DDE3EA",
    "text": "#1B2430", "muted": "#566374", "accent": "#2563EB", "on_accent": "#FFFFFF",
    "good": "#146B58", "limited": "#8A5A00", "poor": "#C62828", "grey": "#566374", "info": "#1D4ED8",
    "canvas": "#0B0F14", "shadow": "0 1px 3px rgba(16,24,40,.10), 0 1px 2px rgba(16,24,40,.06)",
}

_CSS = Path(__file__).with_name("ui.css")

# image-check badge -> (css class, icon name, text)
BADGES = {
    "Good": ("good", "check", "Good"),
    "Limited": ("limited", "warn", "Limited"),
    "Poor": ("poor", "cross", "Poor"),
    "Rejected": ("grey", "ring", "Rejected"),
    "Unreadable": ("grey", "ring", "Unreadable"),
}
# screening badge -> (css class, icon name)
VERDICTS = {
    "Cryptic": ("limited", "warn"),
    "Possibly cryptic": ("limited", "ring"),
    "Not cryptic": ("info", "check"),
    "Cannot assess": ("grey", "ring"),
}

_ICONS = {
    "check": '<path d="M3.5 8.5l3 3 6-7" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
    "warn": '<path d="M8 2.5l6 10.5H2z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/><path d="M8 6.8v3M8 11.6v.1" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>',
    "cross": '<path d="M4 4l8 8M12 4l-8 8" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>',
    "ring": '<circle cx="8" cy="8" r="5" fill="none" stroke="currentColor" stroke-width="1.8"/>',
}


def icon(name: str) -> str:
    return f'<svg class="ico" viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">{_ICONS[name]}</svg>'


def theme_css(light: bool) -> str:
    pal = LIGHT if light else DARK
    root = ";".join(f"--{k.replace('_', '-')}:{v}" for k, v in pal.items())
    return f"<style>:root{{{root}}}\n{_CSS.read_text(encoding='utf8')}</style>"


# --- images ------------------------------------------------------------------------------------------


def data_uri(arr: np.ndarray, fmt: str = ".png") -> str:
    img = arr if arr.ndim == 2 else cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(fmt, img)
    mime = "image/png" if fmt == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(buf.tobytes()).decode()}"


def upscale(arr: np.ndarray, size: int = 560) -> np.ndarray:
    return cv2.resize(arr, (size, size), interpolation=cv2.INTER_CUBIC)


def img_tag(arr: np.ndarray, alt: str, cls: str = "") -> str:
    return f'<img class="{cls}" src="{data_uri(arr)}" alt="{html.escape(alt, quote=True)}">'


def thumb_tag(gray: np.ndarray, alt: str) -> str:
    small = cv2.resize(gray, (56, 56), interpolation=cv2.INTER_AREA)
    return f'<img class="thumb" src="{data_uri(small)}" alt="{html.escape(alt, quote=True)}">'


# --- small pieces -----------------------------------------------------------------------------------


def status_badge(kind: str) -> str:
    cls, ico, text = BADGES[kind]
    return f'<span class="badge {cls}">{icon(ico)}<span>{text}</span></span>'


def verdict_badge(kind: str) -> str:
    cls, ico = VERDICTS[kind]
    return f'<span class="badge big {cls}">{icon(ico)}<span>{html.escape(kind)}</span></span>'


def chip(label: str, value: str, tip: str = "") -> str:
    return (f'<span class="chip" title="{html.escape(tip, quote=True)}"><span class="chip-k">{html.escape(label)}</span>'
            f'<span class="num">{html.escape(value)}</span></span>')


def card(inner: str, cls: str = "") -> str:
    return f'<section class="ws-card {cls}">{inner}</section>'


def eyebrow(text: str) -> str:
    return f'<div class="eyebrow">{html.escape(text)}</div>'


# --- cards ------------------------------------------------------------------------------------------


def viewer_bar(name: str, size: tuple[int, int], badge: str, extra: str = "") -> str:
    dims = f"{size[0]} × {size[1]} px" if size[0] else ""
    return (f'<div class="viewer-bar"><span class="vb-name" title="{html.escape(name, quote=True)}">'
            f'{html.escape(name)}</span><span class="vb-meta num">{dims}</span>{status_badge(badge)}{extra}</div>')


def canvas(inner: str) -> str:
    return f'<div class="canvas">{inner}</div>'


def primary_card(ga, lo, hi, tri, due, badge, note="", src="model") -> str:
    chips = (chip("Trimester", str(tri), "Trimesters: 1 = to 13w6d, 2 = 14w0d-27w6d, 3 = 28 weeks on")
             + chip("Due", f"{due:%d %b %Y}", "Today plus (40 weeks minus the estimate)")
             + f'<span class="chip" title="Image check combines how typical the image is, whether the heatmap '
               f'is on the skull and how close the age is to the edge of the training data.">'
               f'<span class="chip-k">Image check</span>{status_badge(badge)}</span>')
    return card(eyebrow("Estimated gestational age")
                + f'<div class="hero num" aria-label="{fmt_weeks_days(ga)}">{fmt_weeks_days(ga)}</div>'
                + f'<div class="range num">Likely range {fmt_weeks_days(lo)} to {fmt_weeks_days(hi)}</div>'
                + f'<div class="chips">{chips}</div>'
                + (f'<div class="note">{note}</div>' if note else ""), "primary")


def no_estimate_card(message: str) -> str:
    return card(eyebrow("Estimated gestational age") + '<div class="hero muted">No estimate</div>'
                + f'<div class="range">{html.escape(message)}</div>', "primary")


def verdict_card(sc, supportive: str = "") -> str:
    readings = "".join(f"<li><b>{html.escape(c)}:</b> {html.escape(l)}</li>" for c, l in sc.readings)
    notes = "".join(f'<div class="note">{html.escape(n)}</div>' for n in sc.notes)
    action = ""
    if supportive:
        action = (f'<div class="callout"><div class="callout-t">{icon("warn")} See a doctor or midwife</div>'
                  f'<div>{html.escape(supportive)}</div></div>')
    return card(eyebrow("Cryptic pregnancy screening result") + verdict_badge(sc.badge)
                + f'<div class="v-title">{html.escape(sc.label)}</div>'
                + f'<div class="v-sent">{html.escape(sc.sentence)}</div>'
                + (f"<ul>{readings}</ul>" if readings else "")
                + f'<div class="why"><b>{html.escape(sc.why)}</b></div>{notes}{action}'
                + '<div class="fine">A screening aid based on the scan and your answer. It suggests, it does '
                  'not diagnose, and a doctor or midwife must confirm it.</div>', "verdict")


def gauge(frac: float, marks: list[tuple[float, str]] | None = None, label: str = "") -> str:
    frac = float(np.clip(frac, 0, 1))
    ticks = "".join(f'<i class="tick" style="left:{m * 100:.1f}%" title="{html.escape(t, quote=True)}"></i>'
                    for m, t in (marks or []))
    return (f'<div class="gauge" role="img" aria-label="{html.escape(label, quote=True)}">'
            f'<div class="gauge-fill" style="width:{frac * 100:.1f}%"></div>{ticks}</div>')


def tile(label: str, value: str, sub: str, gauge_html: str = "") -> str:
    return (f'<div class="tile"><div class="tile-k">{html.escape(label)}</div>'
            f'<div class="tile-v num">{html.escape(value)}</div>{gauge_html}'
            f'<div class="tile-s">{html.escape(sub)}</div></div>')


def tiles(res, half_days: float, ref, heat_min: float, rejected: bool, base_half: float) -> str:
    if rejected:
        dash = [tile("Head circumference", "—", "not measured"), tile("Heatmap on skull", "—", "no heatmap"),
                tile("Image-check score", f"{res.distance:.3f}", f"refused above {ref.reject:.3f} (lower is better)",
                     gauge(res.distance / (ref.reject * 1.4), [(1 / 1.4, "refusal limit")], "image check score")),
                tile("Range width", "—", "no estimate")]
        return f'<div class="tiles">{"".join(dash)}</div>'
    hc_lo, hc_hi = 44.0, 346.0
    ov = res.overlap or 0.0
    out = [
        tile("Head circumference", f"{res.hc_mm:.0f} mm", "model estimate; training range 44–346 mm",
             gauge((res.hc_mm - hc_lo) / (hc_hi - hc_lo), label="head circumference within the training range")),
        tile("Heatmap on skull", f"{ov:.0%}", f"checked against the {heat_min:.0%} minimum",
             gauge(ov, [(heat_min, "minimum")], "share of the heatmap on the skull")),
        tile("Image-check score", f"{res.distance:.3f}",
             f"unusual above {ref.p90:.3f}, refused above {ref.reject:.3f}",
             gauge(res.distance / (ref.reject * 1.4), [(ref.p90 / (ref.reject * 1.4), "90th percentile"),
                                                       (1 / 1.4, "refusal limit")], "image check score")),
        tile("Range width", f"±{half_days:.0f} d", f"validation error ±{base_half:.0f} d, widened if unusual",
             gauge(half_days / 42, [(base_half / 42, "validation error")], "range width in days")),
    ]
    return f'<div class="tiles">{"".join(out)}</div>'


def timeline(ga: float, lo: float, hi: float) -> str:
    """40-week bar with trimester segments, the 20-week mark, the range and the estimate."""
    def pos(w):
        return float(np.clip(w, 0, 40)) / 40 * 100
    seg = "".join(f'<div class="seg s{i}" style="width:{w / 40 * 100:.1f}%" title="Trimester {i}"><span>{t}</span></div>'
                  for i, (t, w) in enumerate([("T1", 14), ("T2", 14), ("T3", 12)], 1))
    ticks = "".join(f'<span class="wk" style="left:{pos(w):.1f}%">{w}</span>' for w in (0, 10, 20, 30, 40))
    label = f"Estimate {fmt_weeks_days(ga)} on a 40-week timeline, range {fmt_weeks_days(lo)} to {fmt_weeks_days(hi)}"
    return card(eyebrow("Pregnancy timeline (weeks)")
                + f'<div class="timeline" role="img" aria-label="{html.escape(label, quote=True)}">'
                  f'<div class="segs">{seg}</div>'
                  f'<div class="band" style="left:{pos(lo):.1f}%;width:{max(pos(hi) - pos(lo), 0.8):.1f}%"></div>'
                  f'<div class="cut" style="left:{pos(20):.1f}%" title="20 weeks: usual cryptic cut-off"></div>'
                  f'<div class="marker" style="left:{pos(ga):.1f}%"><span class="num">{ga:.1f} w</span></div>'
                  f'<div class="wks">{ticks}</div></div>'
                  '<div class="fine">T1 to 13w6d · T2 14w0d–27w6d · T3 from 28w. The dashed line marks 20 weeks, the usual cut-off for a cryptic pregnancy.</div>')


def growth_chart_uri(ga: float, half_days: float, light: bool) -> str:
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pal = LIGHT if light else DARK
    weeks = np.linspace(12, 40, 100)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.plot(weeks, hadlock_hc_mm(weeks) / 10, color=pal["muted"], lw=2, label="Hadlock mean curve")
    ax.axvspan(ga - half_days / 7, ga + half_days / 7, color=pal["accent"], alpha=0.25, label="Estimate range")
    ax.scatter([ga], [hadlock_hc_mm(ga) / 10], color=pal["accent"], s=60, zorder=3, label="This scan")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(pal["border"])
    ax.set_xlabel("Gestational age (weeks)", color=pal["text"])
    ax.set_ylabel("Head circumference (cm)", color=pal["text"])
    ax.tick_params(colors=pal["text"])
    ax.legend(fontsize=7, frameon=False, labelcolor=pal["text"], loc="upper left")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def growth_card(ga: float, half_days: float, light: bool) -> str:
    alt = (f"Head circumference against gestational age on the Hadlock curve, with this scan at "
           f"{fmt_weeks_days(ga)} and the range shaded")
    return card(eyebrow("Growth chart")
                + f'<img class="chart" src="{growth_chart_uri(ga, half_days, light)}" alt="{html.escape(alt)}">',
                "chart")


def footer(text: str) -> str:
    return f'<div class="ws-footer" role="contentinfo">{icon("warn")}<span>{html.escape(text)}</span></div>'


def logo() -> str:
    return ('<div class="brand"><svg viewBox="0 0 32 32" width="30" height="30" role="img" aria-label="Logo">'
            '<rect width="32" height="32" rx="8" fill="var(--accent)"/>'
            '<path d="M16 6a10 10 0 0 1 10 10H6A10 10 0 0 1 16 6z" fill="none" stroke="var(--on-accent)" '
            'stroke-width="2"/><path d="M16 16v9" stroke="var(--on-accent)" stroke-width="2" stroke-linecap="round"/>'
            '<circle cx="16" cy="26" r="1.6" fill="var(--on-accent)"/></svg>'
            '<div><div class="brand-n">Cryptic pregnancy</div><div class="brand-s">Ultrasound prototype</div></div></div>')


def summary_strip(counts: dict[str, int], total: int) -> str:
    items = "".join(f'<span class="strip-i">{status_badge(k)}<span class="num strip-n">{v}</span></span>'
                    for k, v in counts.items() if v)
    return f'<div class="strip"><span class="strip-i"><b>Scans</b><span class="num strip-n">{total}</span></span>{items}</div>'


def row_html(thumb: str, name: str, estimate: str, badge: str, verdict: str, vbadge: str) -> list[str]:
    """Cells for one batch-table row."""
    return [thumb, f'<div class="cell-name" title="{html.escape(name, quote=True)}">{html.escape(name)}</div>',
            f'<div class="num cell-est">{html.escape(estimate)}</div>', status_badge(badge),
            f'<div class="cell-v">{verdict_badge(vbadge)}</div>']
