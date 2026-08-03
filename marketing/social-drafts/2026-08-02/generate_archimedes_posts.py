from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).parent
W, H = 1080, 1920

FONT_BOLD = "/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf"
FONT_REG = "/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf"
FONT_MONO = "/usr/share/fonts/opentype/urw-base35/NimbusMonoPS-Bold.otf"


def font(path, size):
    return ImageFont.truetype(path, size=size)


def wrap(draw, text, fnt, max_width):
    words = text.split()
    lines = []
    line = ""
    for word in words:
        test = f"{line} {word}".strip()
        if draw.textbbox((0, 0), test, font=fnt)[2] <= max_width:
            line = test
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def text_block(draw, xy, text, fnt, fill, max_width, line_gap=16):
    x, y = xy
    for line in wrap(draw, text, fnt, max_width):
        draw.text((x, y), line, font=fnt, fill=fill)
        y += fnt.size + line_gap
    return y


def pill(draw, xy, text, fill, stroke, fg):
    x, y = xy
    fnt = font(FONT_BOLD, 34)
    tw = draw.textbbox((0, 0), text, font=fnt)[2]
    box = (x, y, x + tw + 56, y + 62)
    draw.rounded_rectangle(box, radius=22, fill=fill, outline=stroke, width=2)
    draw.text((x + 28, y + 15), text, font=fnt, fill=fg)


def make_card(slug, eyebrow, title, problem, proof, action, accent, rows=None):
    bg = (245, 247, 250)
    ink = (16, 27, 45)
    muted = (86, 99, 119)
    navy = (12, 29, 48)
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)

    d.rectangle((0, 0, W, 270), fill=navy)
    d.text((72, 68), "ARCHIMEDES", font=font(FONT_BOLD, 64), fill=(255, 255, 255))
    d.text((74, 145), eyebrow.upper(), font=font(FONT_MONO, 30), fill=accent)

    y = 340
    y = text_block(d, (72, y), title, font(FONT_BOLD, 80), ink, 930, 18)

    d.rounded_rectangle((72, y + 45, 1008, y + 345), radius=28, fill=(255, 255, 255), outline=(210, 218, 230), width=2)
    d.text((112, y + 84), "THE OLD WAY", font=font(FONT_BOLD, 34), fill=(145, 52, 52))
    text_block(d, (112, y + 136), problem, font(FONT_REG, 43), ink, 840, 10)

    d.rounded_rectangle((72, y + 390, 1008, y + 735), radius=28, fill=(255, 255, 255), outline=(210, 218, 230), width=2)
    d.text((112, y + 430), "ARCHIMEDES OUTPUT", font=font(FONT_BOLD, 34), fill=accent)
    text_block(d, (112, y + 482), proof, font(FONT_REG, 43), ink, 840, 10)

    if rows:
        ry = y + 795
        for label, value in rows:
            d.rounded_rectangle((72, ry, 1008, ry + 116), radius=24, fill=(15, 33, 54))
            d.text((112, ry + 28), label, font=font(FONT_BOLD, 34), fill=(190, 211, 230))
            d.text((560, ry + 28), value, font=font(FONT_BOLD, 38), fill=(255, 255, 255))
            ry += 136

    d.rounded_rectangle((72, 1630, 1008, 1768), radius=28, fill=accent)
    text_block(d, (112, 1662), action, font(FONT_BOLD, 42), (255, 255, 255), 850, 8)
    d.text((72, 1816), "archimedesmd.com", font=font(FONT_BOLD, 44), fill=ink)
    d.text((72, 1872), "Turn messy exports into action reports.", font=font(FONT_REG, 31), fill=muted)

    img.save(OUT / f"{slug}.png", quality=95)


make_card(
    "01_data_organization",
    "Data organization",
    "Your business data is not useless. It is just messy.",
    "Exports, notes, CSVs, and reports spread across tools.",
    "Archimedes cleans the structure so the real problem can be analyzed.",
    "Upload messy data. Get something your team can actually use.",
    (0, 132, 150),
    [("INPUT", "messy"), ("OUTPUT", "structured")],
)

make_card(
    "02_reports",
    "Reports",
    "A chart is not a decision.",
    "Dashboards show movement, but they rarely say what to do next.",
    "Archimedes turns organized data into focused reports with risks, segments, and next steps.",
    "Stop staring at rows. Start reviewing the action report.",
    (43, 94, 170),
    [("ECOM DEMO", "$3.0M lost rev"), ("SaaS DEMO", "29.7% churn risk")],
)

make_card(
    "03_predictive_evaluations",
    "Predictive evaluations",
    "Yesterday's spreadsheet should help predict tomorrow.",
    "Most teams collect data, then leave the future buried inside it.",
    "Archimedes supports model training and predictive evaluation from cleaned business data.",
    "Use your past data to rank risk, spot patterns, and test what is likely next.",
    (92, 75, 168),
    [("TRAIN", "clean data"), ("EVALUATE", "likely outcomes")],
)

make_card(
    "04_combined_workflow",
    "Full workflow",
    "One workflow. Three jobs.",
    "Organize the mess. Build the report. Evaluate what might happen next.",
    "Archimedes connects data organization, reporting, and predictive evaluation into one operating workflow.",
    "Archimedes is not another CSV chatbot. It is the first analyst pass.",
    (23, 139, 91),
    [("1", "organize"), ("2", "report + predict")],
)
