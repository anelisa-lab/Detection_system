"""WCAG contrast of the UI palettes: text pairs need at least 4.5:1 (large text 3:1)."""
import pytest

from hcml.ui import DARK, LIGHT


def _lin(c):
    c /= 255
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def _rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _lum(rgb):
    r, g, b = (_lin(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def ratio(a, b):
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def mix(fg, bg, pct):
    """color-mix(in srgb, fg pct%, bg)"""
    return tuple(round(f * pct + b * (1 - pct)) for f, b in zip(_rgb(fg), _rgb(bg)))


@pytest.mark.parametrize("name,p", [("dark", DARK), ("light", LIGHT)])
def test_text_and_status_contrast(name, p):
    checks = {
        "text on bg": (_rgb(p["text"]), _rgb(p["bg"])),
        "text on surface": (_rgb(p["text"]), _rgb(p["surface"])),
        "text on surface2": (_rgb(p["text"]), _rgb(p["surface2"])),
        "muted on surface": (_rgb(p["muted"]), _rgb(p["surface"])),
        "muted on bg": (_rgb(p["muted"]), _rgb(p["bg"])),
        "muted on surface2": (_rgb(p["muted"]), _rgb(p["surface2"])),
        "on-accent on accent": (_rgb(p["on_accent"]), _rgb(p["accent"])),
        "accent on surface": (_rgb(p["accent"]), _rgb(p["surface"])),
    }
    for k in ("good", "limited", "poor", "grey", "info"):          # badge text over its 14% tint
        checks[f"{k} badge"] = (_rgb(p[k]), mix(p[k], p["surface"], 0.14))
        checks[f"{k} on surface"] = (_rgb(p[k]), _rgb(p["surface"]))
    checks["callout title"] = (_rgb(p["limited"]), mix(p["limited"], p["surface"], 0.10))
    failures = {k: round(ratio(*v), 2) for k, v in checks.items() if ratio(*v) < 4.5}
    assert not failures, f"{name} theme below 4.5:1: {failures}"
