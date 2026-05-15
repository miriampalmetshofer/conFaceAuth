"""Shared font setup for evaluation figures."""
import base64
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import plotly.graph_objects as go


FONT_FAMILY = "Latin Modern Roman"
PLOTLY_FONT_FAMILY = f"{FONT_FAMILY}, serif"
FONT_DIR = Path("/usr/local/texlive/2025/texmf-dist/fonts/opentype/public/lm")
FONT_FILES = {
    "regular": (FONT_DIR / "lmroman10-regular.otf", "400", "normal"),
    "bold": (FONT_DIR / "lmroman10-bold.otf", "700", "normal"),
    "italic": (FONT_DIR / "lmroman10-italic.otf", "400", "italic"),
    "bold_italic": (FONT_DIR / "lmroman10-bolditalic.otf", "700", "italic"),
}


def configure_matplotlib_font() -> None:
    for font_path, _, _ in FONT_FILES.values():
        if font_path.exists():
            font_manager.fontManager.addfont(font_path)

    plt.rcParams.update({"font.family": FONT_FAMILY})


def apply_plotly_font(fig: go.Figure) -> None:
    fig.update_layout(font_family=PLOTLY_FONT_FAMILY)

    for annotation in fig.layout.annotations or []:
        if annotation.font is None:
            annotation.font = dict(family=PLOTLY_FONT_FAMILY)
        else:
            annotation.font.family = PLOTLY_FONT_FAMILY


def plotly_font_style_tag() -> str:
    rules = [
        _font_face_rule(font_path, weight, style)
        for font_path, weight, style in FONT_FILES.values()
    ]
    rules.append(
        "body, .plotly-graph-div, .plotly-graph-div * {"
        f"font-family: '{FONT_FAMILY}', serif;"
        "}"
    )
    return f"<style>{''.join(rule for rule in rules if rule)}</style>"


def plotly_html(fig: go.Figure) -> str:
    html = fig.to_html()
    return html.replace("</head>", f"{plotly_font_style_tag()}</head>", 1)


def _font_face_rule(font_path: Path, weight: str, style: str) -> str:
    if not font_path.exists():
        return ""

    encoded = base64.b64encode(font_path.read_bytes()).decode("ascii")
    return (
        "@font-face {"
        f"font-family: '{FONT_FAMILY}';"
        f"src: url(data:font/otf;base64,{encoded}) format('opentype');"
        f"font-weight: {weight};"
        f"font-style: {style};"
        "}"
    )


configure_matplotlib_font()
