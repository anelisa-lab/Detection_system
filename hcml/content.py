"""Plain-language reference text. General information only, not medical advice."""
import numpy as np

from .growth import fmt_weeks_days, trimester

# Approximate average size by week (crown-heel length from week 20, crown-rump before),
# from commonly cited reference tables. Individual babies vary widely.
_SIZE = [(12, 5.4, 14), (16, 11.6, 100), (20, 25.6, 300), (24, 30.0, 600), (28, 37.6, 1000),
         (32, 42.4, 1700), (36, 47.4, 2600), (40, 51.2, 3400)]

_DEVELOPMENT = [
    (0, "Organs are formed and starting to work. Fingers and toes are separate, and the face is taking shape."),
    (14, "The baby grows quickly and can move, suck and swallow. Movements may be felt from about 16-22 weeks."),
    (20, "Hearing is developing and the baby has sleep and wake cycles. The anatomy scan usually checks the organs."),
    (24, "Lungs are developing and the baby responds to sound. Babies born from about 24 weeks can survive with intensive care."),
    (28, "The eyes open, the brain grows fast and the baby gains fat. Movements are regular and strong."),
    (32, "Weight gain speeds up and lungs keep maturing. The baby usually settles head-down in the weeks ahead."),
    (37, "Considered early term. Lungs and brain are nearly mature, and the baby is preparing for birth."),
]

CHECKUPS = {
    1: ["Book a first antenatal visit if you have not had one.",
        "Blood tests and a dating or early scan are usually offered.",
        "Discuss folic acid, current medicines and any health conditions."],
    2: ["An anatomy scan is usually offered at about 18-22 weeks.",
        "Routine visits check blood pressure, urine and growth.",
        "A glucose tolerance test is commonly offered at about 24-28 weeks."],
    3: ["Visits become more frequent, usually every 2-4 weeks and then weekly near term.",
        "Growth, the baby's position and blood pressure are checked.",
        "Plan the birth, and ask which signs mean you should call or go in."],
}

CHECKLIST = [
    "Take folic acid and any vitamins your clinician advises.",
    "Avoid alcohol, smoking and recreational drugs.",
    "Check with a pharmacist or doctor before any medicine, including herbal ones.",
    "Eat a varied diet, drink water and avoid raw meat, unpasteurised dairy and high-mercury fish.",
    "Keep moving with gentle activity such as walking.",
    "Get enough rest and ask for support with your mental health if you feel low or anxious.",
]

SEE_DOCTOR = [
    "Heavy vaginal bleeding or fluid leaking from the vagina",
    "Severe or persistent abdominal pain, or regular painful tightening",
    "Severe headache, vision changes or sudden swelling of the face or hands",
    "Fever, or pain or burning when you pass urine",
    "Less movement than usual from the baby (from about 24 weeks)",
    "Persistent vomiting so that you cannot keep fluids down",
    "Feeling faint, short of breath or having chest pain",
]

DISCLAIMER = (
    "Research prototype, not a diagnostic device. A cryptic pregnancy cannot be identified from a scan "
    "alone: the scan can only suggest the stage of development, which a clinician then compares with the "
    "person's own history. Estimates have an error of days to weeks. Always see a doctor or midwife for "
    "confirmation and care.")

SUPPORTIVE = (
    "Please see a doctor or midwife promptly for confirmation, dating and antenatal care. A cryptic pregnancy "
    "is a recognised and manageable situation, it happens to many people, and it is not your fault. Early "
    "care helps both you and the baby, and a clinician can answer your questions and go through your history "
    "with you.")

FOOTER = ("Research prototype, not a diagnostic device and not validated on cryptic pregnancies. A cryptic "
          "pregnancy cannot be identified from a scan alone. A clinician confirms everything.")

UNRELIABLE_SENTENCE = ("This estimate is unreliable for this image. Please retake the scan or use a "
                       "standard head circumference view.")

NO_FETUS_SENTENCE = ("No fetal head was found in this image, so this tool has nothing to date. If the person is not "
                     "pregnant, that is expected. If there is any doubt about a pregnancy, a pregnancy test and a "
                     "clinician can confirm.")

CRYPTIC_NOTE = (
    "A cryptic pregnancy cannot be identified from a scan alone, because it depends on the person not "
    "knowing they are pregnant. The scan can only show the stage of development, which a clinician then "
    "compares with the person's own history.")


def size_at(ga_weeks: float) -> tuple[float, float]:
    """Approximate (length cm, weight g), interpolated and clipped to the reference range."""
    w, cm, g = zip(*_SIZE)
    return float(np.interp(ga_weeks, w, cm)), float(np.interp(ga_weeks, w, g))


def development_at(ga_weeks: float) -> str:
    text = _DEVELOPMENT[0][1]
    for start, t in _DEVELOPMENT:
        if ga_weeks >= start:
            text = t
    return text


_ORDINAL = {1: "first", 2: "second", 3: "third"}


def what_scan_suggests(ga_weeks: float) -> list[str]:
    """Scan-only paragraphs: estimate and trimester, the cryptic caveat, and a stage-based note."""
    tri = trimester(ga_weeks)
    paras = [
        f"The scan suggests a gestational age of about {fmt_weeks_days(ga_weeks)}, which is consistent with "
        f"the {_ORDINAL[tri]} trimester.",
        CRYPTIC_NOTE,
    ]
    if ga_weeks >= 20:
        paras.append("An estimate of 20 weeks or more suggests the pregnancy is far along. Please see a doctor "
                     "or midwife soon for confirmation and care.")
    else:
        paras.append("This suggests an earlier stage of pregnancy. Please still see a doctor or midwife soon for "
                     f"confirmation and care. Typical at this stage: {development_at(ga_weeks)[0].lower()}"
                     f"{development_at(ga_weeks)[1:]}")
    return paras
