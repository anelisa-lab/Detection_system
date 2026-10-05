"""Streamlit app: a clinical-ultrasound-workstation view of the gestational-age estimate and the cryptic
pregnancy screening result. Research prototype.

    streamlit run app.py            # expects artifacts/model.keras from train.py
    MODEL_DIR=other_dir streamlit run app.py

Layout and styling live in hcml/ui.py and hcml/ui.css; the analysis logic is in the hcml package.
"""
import html
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from hcml import content, ui
from hcml.growth import CRL_CITATION, CRL_RANGE_MM, due_date, fmt_weeks_days, robinson_ga_weeks, trimester
from hcml.model import overlay
from hcml.pipeline import LIMITED_WARNING, Predictor
from hcml.report import build_report
from hcml.screening import ANSWERS, screen

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "artifacts"))
PAGE_SIZE = 20
MAX_PDF_SCANS = 100
BADGE_ORDER = ["Good", "Limited", "Poor", "Rejected", "Unreadable"]

st.set_page_config(page_title="Ultrasound age estimate", page_icon="🩺", layout="wide")
light = bool(st.session_state.get("light", False))
st.markdown(ui.theme_css(light), unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model...")
def load_predictor(model_dir: str) -> Predictor:
    return Predictor(model_dir)


@st.cache_data(show_spinner=False, max_entries=1024)
def analyse(data: bytes, name: str, model_dir: str, strictness: float):
    """Preprocess one upload, check it is a head-circumference view, and estimate the age."""
    return load_predictor(model_dir).predict(data, name, strictness)


# --- model ---------------------------------------------------------------------------------------

if not (MODEL_DIR / "model.keras").exists():
    st.error(f"No trained model in `{MODEL_DIR}/`. Run `python train.py --data <HC18 training_set> "
             f"--out {MODEL_DIR}` first, or set MODEL_DIR.")
    st.stop()
try:
    predictor = load_predictor(str(MODEL_DIR))
except Exception as e:
    st.error(f"The model in `{MODEL_DIR}/` could not be loaded ({e}). Retrain with `python train.py` "
             "(older models lack the input-validity reference).")
    st.stop()
meta, reg = predictor.meta, predictor.reg
half_days = float(reg["interval_days"])

# --- sidebar: brand, navigation slot, settings, performance ----------------------------------------

with st.sidebar:
    st.markdown(ui.logo(), unsafe_allow_html=True)
    nav_slot = st.container()
    with st.expander("Settings"):
        strictness = st.slider(
            "Image check strictness", 0.5, 1.0, 1.0, 0.05,
            help="1.0 is the calibrated setting: about 0.5% of valid head scans are wrongly refused. "
                 "Lower values refuse more unusual images.")
        st.toggle("Light theme", key="light", help="Brighter cards for bright rooms.")
    with st.expander("Model performance"):
        t = reg["test"]
        st.metric("Age error (MAE)", f"{t['mae_days']:.1f} days",
                  help="Mean absolute error of the gestational-age estimate on held-out HC18 scans.")
        st.caption(f"Typical range shown: ±{half_days:.0f} days (80% of validation errors were within it). "
                   f"Within 14 days: {t['within_14_days']:.0%}.")
        m = meta.get("metrics", {})
        st.table(pd.DataFrame({
            "Stage classifier": ["Accuracy (= weighted recall)", "Precision", "F1", "Specificity"],
            "Score": [m.get(k) for k in ("accuracy", "precision", "f1", "specificity")],
        }).round(3))
        pc = meta.get("per_class", {})
        if pc:
            base = max(v["support"] for v in pc.values()) / sum(v["support"] for v in pc.values())
            st.caption(f"Stage accuracy should be read against the always-\"early\" baseline of {base:.1%}.")
        st.caption(f"Age formula: {reg['formula']}.")

# --- top bar, question and upload ---------------------------------------------------------------------

st.markdown('<div class="topbar"><h1>Late-discovery screening aid for cryptic pregnancy</h1>'
            '<span class="sub">Ultrasound prototype · gestational age from the fetal head</span></div>',
            unsafe_allow_html=True)

with st.expander("Scans and question" + (f"  ·  answer: {st.session_state.get('answer', 'Not sure')}"
                                         if st.session_state.get("uploader") else ""),
                 expanded=not st.session_state.get("uploader")):
    answer = st.radio("Did you know you were pregnant before this scan?", ANSWERS, index=2, horizontal=True, key="answer",
                      help="This is the only personal question. The scan shows how far along the pregnancy "
                           "looks; only you can say whether you knew about it.")
    weeks_found = None
    if answer == "Yes":
        weeks_found = st.number_input("If yes, at roughly how many weeks did you find out? (0 = not sure)",
                                      0, 42, 0, help="Optional.") or None
    st.markdown('<div class="hint">Expected view: a <b>standard axial head circumference plane</b> '
                '(2D fetal head, skull ring visible). Whole-fetus or first-trimester CRL views are not '
                'supported. Drag and drop one scan or a whole folder of scans.</div>', unsafe_allow_html=True)
    files = st.file_uploader("Fetal head ultrasound images (PNG, JPEG, BMP or DICOM)", key="uploader",
                             type=["png", "jpg", "jpeg", "bmp", "dcm", "dicom"], accept_multiple_files=True,
                             help="Use a 2D image of the fetal head. Several images are fine.")

# --- analysis (cached per file list) --------------------------------------------------------------------


def ensure_results(files, strictness):
    sig = (tuple((f.name, len(f.getvalue())) for f in files), round(strictness, 3), str(MODEL_DIR))
    held = st.session_state.get("analysis")
    if held and held["sig"] == sig:
        return held["entries"]
    entries, bar = [], st.progress(0.0, text=f"Analysing 0 of {len(files)} scans...")
    for k, f in enumerate(files):
        try:
            entries.append({"name": f.name, "res": analyse(f.getvalue(), f.name, str(MODEL_DIR), strictness),
                            "error": None})
        except Exception as e:
            entries.append({"name": f.name, "res": None, "error": str(e)})
        bar.progress((k + 1) / len(files), text=f"Analysing {k + 1} of {len(files)} scans...")
    bar.empty()
    for e in entries:
        e["thumb"] = ui.thumb_tag(e["res"].gray, f"Thumbnail of {e['name']}") if e["res"] is not None else ""
    st.session_state["analysis"] = {"sig": sig, "entries": entries}
    if len(files) > 5:
        st.session_state["nav"] = "Batch"          # many files: start on the table
    st.session_state["sel"] = 0
    return entries


entries = ensure_results(files, strictness) if files else []
if not files:
    st.session_state.pop("analysis", None)

NAV = ["Scan", "Batch", "Report", "About"]
with nav_slot:
    st.radio("Navigation", NAV, key="nav", label_visibility="collapsed",
             format_func=lambda x: {"Scan": "▣  Scan", "Batch": "▤  Batch", "Report": "▥  Report",
                                    "About": "ⓘ  About"}[x])
page = st.session_state.get("nav", "Scan")

# --- per-scan view model -----------------------------------------------------------------------------


def describe(entry, i):
    """Everything the page needs about one scan, including the screening result."""
    res = entry["res"]
    if res is None:
        return None
    rejected = res.ga_weeks is None
    ga, half, src = res.ga_weeks, res.half_days, "model"
    crl = st.session_state.get(f"crl_{i}")
    if rejected and crl and CRL_RANGE_MM[0] <= crl <= CRL_RANGE_MM[1]:
        ga, half, src = robinson_ga_weeks(crl), 7.0, "crl"
    limited = (not rejected) and res.badge != "Good"
    shown = "Good" if src == "crl" else res.badge
    sc = screen(ga, half, shown, answer, weeks_found, "; ".join(res.badge_reasons))
    v = {"res": res, "ga": ga, "half": half, "src": src, "rejected": rejected, "limited": limited, "sc": sc,
         "crl": crl, "badge": res.badge}
    if ga is not None:
        v.update(lo=max(ga - half / 7, 0), hi=ga + half / 7, tri=trimester(ga), due=due_date(ga),
                 headline=fmt_weeks_days(ga))
        v["paras"] = content.what_scan_suggests(ga) if (src == "crl" or not limited) else [content.UNRELIABLE_SENTENCE]
    else:
        v.update(headline="No estimate", paras=[content.UNRELIABLE_SENTENCE])
    return v


def heat_image(res, alpha, size=560):
    """Grad-CAM overlay with the detected skull outline, upscaled for display."""
    heat = ui.upscale(overlay(res.gray, res.cam, alpha), size)
    if res.ellipse is not None:
        (cx, cy), (a, b), ang = res.ellipse
        k = size / res.gray.shape[0]
        cv2.ellipse(heat, ((cx * k, cy * k), (a * k, b * k), ang), (255, 255, 255), 2, cv2.LINE_AA)
    return heat


def report_item(entry, v, alpha):
    res = v["res"]
    lines = [v["sc"].sentence, v["sc"].why, f"Your answer: {answer}"] + v["sc"].notes
    item = {"name": entry["name"], "gray": res.gray, "heat": None if v["rejected"] else heat_image(res, alpha, 224),
            "headline": v["headline"], "range_text": None, "trimester_text": None, "due_text": None,
            "screen_badge": v["sc"].badge, "screen_label": v["sc"].label, "screen_lines": lines,
            "summary": v["paras"]}
    if v["ga"] is not None:
        item["range_text"] = (f"Likely range: {fmt_weeks_days(v['lo'])} to {fmt_weeks_days(v['hi'])} "
                              f"(+/-{v['half']:.0f} days)")
        item["trimester_text"] = f"Trimester {v['tri']}. " + (
            f"Formula: {reg['formula']}" if v["src"] == "model" else f"From user-entered CRL ({CRL_CITATION})")
        item["due_text"] = f"Estimated due date: {v['due']:%d %b %Y}"
        if v["limited"]:
            item["trimester_text"] += f". Image check: {res.badge} ({'; '.join(res.badge_reasons)}). {LIMITED_WARNING}"
    return item


def table_rows():
    rows = []
    for i, e in enumerate(entries):
        v = describe(e, i)
        if v is None:
            rows.append({"i": i, "file": e["name"], "estimate": "Unreadable", "ga": np.inf, "image_check": "Unreadable",
                         "verdict": "Cannot assess", "label": e["error"] or "", "heatmap_on_skull": None, "score": None})
        else:
            r = v["res"]
            rows.append({"i": i, "file": e["name"], "estimate": v["headline"],
                         "ga": v["ga"] if v["ga"] is not None else np.inf,
                         "image_check": r.badge if v["src"] == "model" else "Good", "verdict": v["sc"].badge,
                         "label": v["sc"].label, "heatmap_on_skull": None if r.overlap is None else round(r.overlap, 2),
                         "score": round(r.distance, 3)})
    return rows


def open_scan(i):
    st.session_state["sel"] = i
    st.session_state["nav"] = "Scan"


def guidance(v):
    """Stage guide, scan summary and when-to-contact tabs."""
    g1, g2, g3 = st.tabs(["Stage guide", "What this scan suggests", "When to contact a doctor"])
    with g1:
        if v["ga"] is not None and v["sc"].badge != "Cannot assess":
            length_cm, weight_g = content.size_at(v["ga"])
            st.markdown(ui.card(
                ui.eyebrow(f"About this stage · {fmt_weeks_days(v['ga'])} · trimester {v['tri']}")
                + f"<p><b>Size.</b> Roughly {length_cm:.0f} cm and {weight_g:,.0f} g on average; healthy babies vary.</p>"
                + f"<p><b>Development.</b> {html.escape(content.development_at(v['ga']))}</p>"
                + "<p><b>What care is due.</b></p><ul>"
                + "".join(f"<li>{html.escape(s)}</li>" for s in content.CHECKUPS[v["tri"]]) + "</ul>"
                + "<p><b>General health.</b></p><ul>"
                + "".join(f"<li>{html.escape(s)}</li>" for s in content.CHECKLIST) + "</ul>"), unsafe_allow_html=True)
        else:
            st.caption("The stage guide appears when the scan gives a reliable estimate.")
    with g2:
        body = "".join(f"<p>{html.escape(t)}</p>" for t in v["paras"])
        st.markdown(ui.card(ui.eyebrow("What this scan suggests") + body), unsafe_allow_html=True)
    with g3:
        st.markdown(ui.card(ui.eyebrow("Contact a doctor or midwife promptly for")
                            + "<ul>" + "".join(f"<li>{html.escape(s)}</li>" for s in content.SEE_DOCTOR) + "</ul>"),
                    unsafe_allow_html=True)


# --- pages ------------------------------------------------------------------------------------------


def page_scan():
    n = len(entries)
    if n == 0:
        st.markdown(ui.card(ui.eyebrow("Ready for a scan")
                            + '<div class="v-title">Drop an ultrasound image above to begin</div>'
                              '<div class="range">The app estimates gestational age from the fetal head and checks '
                              'that the image is a standard head circumference view.</div>'), unsafe_allow_html=True)
        return
    if n > 1:
        st.selectbox("Scan", list(range(n)), key="sel", format_func=lambda k: f"{k + 1}/{n}  ·  {entries[k]['name']}")
    i = min(st.session_state.get("sel", 0), n - 1)
    entry = entries[i]
    v = describe(entry, i)
    if v is None:
        st.error(f"**{entry['name']}** could not be read ({entry['error']}). Check that it is a valid PNG, JPEG, "
                 "BMP or DICOM image, then upload it again.")
        return
    res, alpha = v["res"], float(st.session_state.get("alpha", 0.4))
    left, right = st.columns([1.55, 1], gap="medium")

    with left:
        bar = ui.viewer_bar(entry["name"], res.orig_size, v["badge"])
        orig = res.original
        if st.session_state.get("compare", False):
            second = (ui.img_tag(heat_image(res, alpha), f"Grad-CAM overlay of {entry['name']} with the skull outline")
                      if not v["rejected"] else ui.img_tag(ui.upscale(res.gray), f"Preprocessed {entry['name']}"))
            st.markdown(bar + ui.canvas(ui.img_tag(orig, f"Original scan {entry['name']}") + second).replace(
                'class="canvas"', 'class="canvas compare"'), unsafe_allow_html=True)
        else:
            t1, t2, t3 = st.tabs(["Original", "Preprocessed", "Grad-CAM + skull outline"])
            t1.markdown(bar + ui.canvas(ui.img_tag(orig, f"Original scan {entry['name']}")), unsafe_allow_html=True)
            t2.markdown(bar + ui.canvas(ui.img_tag(ui.upscale(res.gray),
                                                    f"Preprocessed 224 by 224 scan {entry['name']}")),
                        unsafe_allow_html=True)
            if v["rejected"]:
                t3.markdown(bar + ui.canvas('<div class="empty">Heatmap hidden: the image did not pass the check for '
                                            'a standard head circumference view.</div>'), unsafe_allow_html=True)
            else:
                t3.markdown(bar + ui.canvas(ui.img_tag(
                    heat_image(res, alpha), f"Grad-CAM heatmap over {entry['name']} with the skull outlined in white")),
                    unsafe_allow_html=True)
        c1, c2 = st.columns([2, 1])
        c1.slider("Heatmap opacity", 0.0, 1.0, 0.4, 0.05, key="alpha")
        c2.toggle("Side-by-side compare", key="compare", help="Show the original beside the Grad-CAM overlay.")
        if not v["rejected"]:
            on = (f"{res.overlap:.0%} of the heatmap is on the detected skull (white outline)"
                  if res.ellipse is not None else "no skull-like region was detected")
            st.caption(f"Heatmap check: {on}. Attention on corners, text or calipers rather than the fetal skull "
                       "means the estimate should not be trusted.")
        if v["ga"] is not None:
            st.markdown(ui.timeline(v["ga"], v["lo"], v["hi"]), unsafe_allow_html=True)

    with right:
        if v["limited"]:
            st.warning(LIMITED_WARNING)
        if v["ga"] is None:
            st.markdown(ui.no_estimate_card(res.notes[0]), unsafe_allow_html=True)
            st.markdown('<div class="hint">The model was trained on standard head views from about 12 to 40 weeks. '
                        'First-trimester whole-fetus (CRL) views, photos and unusual frames are outside that, and '
                        'image-based first-trimester scans are not supported yet.</div>', unsafe_allow_html=True)
        if v["rejected"]:
            crl = st.number_input("First trimester? Enter the crown-rump length (CRL) in mm, if the scan shows one",
                                  0.0, 120.0, 0.0, 0.1, key=f"crl_{i}",
                                  help=f"Uses the formula of {CRL_CITATION}. Valid for about {CRL_RANGE_MM[0]:.0f}-"
                                       f"{CRL_RANGE_MM[1]:.0f} mm. This is your measurement, not read from the image.")
            if crl and not CRL_RANGE_MM[0] <= crl <= CRL_RANGE_MM[1]:
                st.warning(f"A CRL of {crl:.0f} mm is outside the {CRL_RANGE_MM[0]:.0f}-{CRL_RANGE_MM[1]:.0f} mm "
                           "range of the formula, so no estimate was made.")
        if v["ga"] is not None:
            if v["src"] == "model":
                why = ("; ".join(res.badge_reasons)
                       or "the scan looks like a standard head view and the heatmap is on the skull")
                note = (f"From head circumference using {html.escape(reg['formula'])}. The range is ±{v['half']:.0f} "
                        f"days (80% of validation errors are within ±{half_days:.0f} days"
                        f"{', widened here' if v['half'] > half_days + 0.5 else ''}); individual errors can be "
                        f"larger.<br><b>Why {res.badge}:</b> {html.escape(why[0].upper() + why[1:])}.")
            else:
                note = (f"From the CRL you entered ({v['crl']:.1f} mm), using {CRL_CITATION}. This is not read from "
                        "the image. The range is a rough ±1 week.")
            if v["lo"] < 14:
                note += ("<br><b>For scans before 14 weeks, crown-rump length is the standard measure. Head "
                         "circumference estimates here are less reliable.</b>")
            st.markdown(ui.primary_card(v["ga"], v["lo"], v["hi"], v["tri"], v["due"],
                                        v["badge"] if v["src"] == "model" else "Good", note, v["src"]),
                        unsafe_allow_html=True)
        supportive = content.SUPPORTIVE if v["sc"].badge in ("Cryptic", "Possibly cryptic") else ""
        st.markdown(ui.verdict_card(v["sc"], supportive), unsafe_allow_html=True)
        st.markdown(ui.tiles(res, v["half"], predictor.ref, predictor.heat["min_overlap"], v["rejected"], half_days),
                    unsafe_allow_html=True)
        if v["ga"] is not None and v["src"] == "model":
            st.markdown(ui.growth_card(v["ga"], v["half"], light), unsafe_allow_html=True)
        try:
            st.download_button("Download report (PDF)", build_report([report_item(entry, v, alpha)]),
                               f"report_{Path(entry['name']).stem}.pdf", "application/pdf",
                               help="Image, estimate, screening result and the disclaimer.")
        except Exception as e:
            st.error(f"The PDF report could not be created ({e}).")

    guidance(v)



def page_batch():
    if not entries:
        st.info("Upload scans above to see the batch table.")
        return
    rows = table_rows()
    counts = {k: sum(1 for r in rows if r["image_check"] == k) for k in BADGE_ORDER}
    st.markdown(ui.summary_strip(counts, len(rows)), unsafe_allow_html=True)
    f1, f2, f3 = st.columns([2, 2, 1.6])
    present = [k for k in BADGE_ORDER if counts[k]]
    pick = f1.multiselect("Filter by image check", present, default=present, key="f_badge")
    verdicts = ["Cryptic", "Possibly cryptic", "Not cryptic", "Cannot assess"]
    vpick = f2.multiselect("Filter by screening result", verdicts, default=verdicts, key="f_verdict")
    order = f3.selectbox("Sort by", ["File name", "Estimate, low to high", "Estimate, high to low",
                                     "Image check, best first", "Screening result"], key="sort")
    shown = [r for r in rows if r["image_check"] in pick and r["verdict"] in vpick]
    keyf = {"File name": lambda r: r["file"].lower(), "Estimate, low to high": lambda r: r["ga"],
            "Estimate, high to low": lambda r: -r["ga"] if np.isfinite(r["ga"]) else 1,
            "Image check, best first": lambda r: BADGE_ORDER.index(r["image_check"]),
            "Screening result": lambda r: r["verdict"]}[order]
    shown.sort(key=keyf)
    pages = max(1, -(-len(shown) // PAGE_SIZE))
    st.caption(f"{len(shown)} of {len(rows)} scans shown. Press Open to view a scan in the viewer.")
    page_no = st.columns([1, 5])[0].number_input("Page", 1, pages, 1, key="page") if pages > 1 else 1
    widths = [0.7, 3, 2, 1.6, 2.6, 1]
    for c, t in zip(st.columns(widths), ["", "File", "Estimate", "Image check", "Screening result", ""]):
        c.markdown(f'<div class="rowhead">{t}</div>', unsafe_allow_html=True)
    for r in shown[(page_no - 1) * PAGE_SIZE: page_no * PAGE_SIZE]:
        e = entries[r["i"]]
        cells = ui.row_html(e["thumb"] or "", r["file"], r["estimate"], r["image_check"], r["label"], r["verdict"])
        cols = st.columns(widths)
        for c, h in zip(cols, cells):
            c.markdown(h, unsafe_allow_html=True)
        cols[5].button("Open ›", key=f"open_{r['i']}", on_click=open_scan, args=(r["i"],),
                       help=f"Open {r['file']} in the viewer")
    export_controls(shown, rows)


def export_controls(shown, rows):
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("i", "ga")} for r in rows])
    d1, d2 = st.columns(2)
    d1.download_button("Export all as CSV", df.to_csv(index=False), "estimates.csv", "text/csv")
    ids = [r["i"] for r in shown if entries[r["i"]]["res"] is not None][:MAX_PDF_SCANS]
    if d2.button(f"Prepare PDF report ({len(ids)} scans)", help=f"Builds one page per scan, up to {MAX_PDF_SCANS}."):
        a = float(st.session_state.get("alpha", 0.4))
        items = [report_item(entries[i], describe(entries[i], i), a) for i in ids]
        with st.spinner("Building the PDF..."):
            st.session_state["pdf"] = build_report(items) if items else None
    if st.session_state.get("pdf"):
        d2.download_button("Download report (PDF)", st.session_state["pdf"], "ultrasound_report.pdf", "application/pdf")
    if len(shown) > MAX_PDF_SCANS:
        st.caption(f"The PDF includes the first {MAX_PDF_SCANS} scans of the current filter.")


def page_report():
    if not entries:
        st.info("Upload scans above to build a report.")
        return
    rows = table_rows()
    good = [r for r in rows if np.isfinite(r["ga"])]
    st.markdown(ui.summary_strip({k: sum(1 for r in rows if r["image_check"] == k) for k in BADGE_ORDER}, len(rows)),
                unsafe_allow_html=True)
    if len(good) > 1:
        avg = float(np.mean([r["ga"] for r in good]))
        st.markdown(ui.card(ui.eyebrow(f"Overall average of {len(good)} scans")
                            + f'<div class="hero num">{fmt_weeks_days(avg)}</div>'
                              f'<div class="range">Est. due date {due_date(avg):%d %b %Y}. Scans with no estimate are '
                              'left out; all scans are assumed to be of the same pregnancy.</div>'),
                    unsafe_allow_html=True)
    export_controls(rows, rows)


def page_about():
    st.markdown(ui.card(
        ui.eyebrow("About")
        + "<p>This research prototype estimates gestational age from a fetal head ultrasound, checks that the image "
          "is a standard head circumference view, and gives a screening aid for the cryptic pregnancy question.</p>"
        + f"<p><b>Age estimate.</b> A ResNet50 regression head predicts head circumference, converted to age with "
          f"{html.escape(reg['formula'])}. Average error on held-out scans is {reg['test']['mae_days']:.1f} days.</p>"
        + "<p><b>Image check.</b> Embedding distance to the training scans, a heatmap-on-skull check and the "
          "estimate's distance from the training data decide Good, Limited, Poor or Rejected.</p>"
        + "<p><b>Screening.</b> A cryptic pregnancy cannot be identified from a scan alone. The result combines the "
          "scan estimate with one question: did you know you were pregnant before this scan?</p>"
        + "<p><b>Limits.</b> Single-centre data; first-trimester scans are measured by crown-rump length, which the "
          "image model does not read; not validated on cryptic pregnancies.</p>"
        + f'<p class="fine">{html.escape(content.DISCLAIMER)}</p>'), unsafe_allow_html=True)


{"Scan": page_scan, "Batch": page_batch, "Report": page_report, "About": page_about}[page]()

st.markdown(ui.footer(content.FOOTER), unsafe_allow_html=True)
