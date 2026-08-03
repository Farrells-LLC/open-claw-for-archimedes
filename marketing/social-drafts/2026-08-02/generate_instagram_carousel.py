from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).parent
W, H = 1080, 1350
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


def text_block(draw, xy, text, fnt, fill, max_width, gap=12):
    x, y = xy
    for line in wrap(draw, text, fnt, max_width):
        draw.text((x, y), line, font=fnt, fill=fill)
        y += fnt.size + gap
    return y


def card(slug, eyebrow, headline, old_way, output, cta, accent):
    bg = (245, 247, 250)
    navy = (12, 29, 48)
    ink = (16, 27, 45)
    muted = (85, 98, 116)
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)

    d.rectangle((0, 0, W, 185), fill=navy)
    d.text((64, 48), "ARCHIMEDES", font=font(FONT_BOLD, 58), fill=(255, 255, 255))
    d.text((66, 120), eyebrow.upper(), font=font(FONT_MONO, 26), fill=accent)

    y = 250
    y = text_block(d, (64, y), headline, font(FONT_BOLD, 68), ink, 940, 12)

    y += 36
    d.rounded_rectangle((64, y, 1016, y + 215), radius=24, fill=(255, 255, 255), outline=(210, 218, 230), width=2)
    d.text((104, y + 34), "THE OLD WAY", font=font(FONT_BOLD, 30), fill=(145, 52, 52))
    text_block(d, (104, y + 82), old_way, font(FONT_REG, 38), ink, 840, 8)

    y += 255
    d.rounded_rectangle((64, y, 1016, y + 260), radius=24, fill=(255, 255, 255), outline=(210, 218, 230), width=2)
    d.text((104, y + 34), "ARCHIMEDES OUTPUT", font=font(FONT_BOLD, 30), fill=accent)
    text_block(d, (104, y + 82), output, font(FONT_REG, 38), ink, 840, 8)

    d.rounded_rectangle((64, 1110, 1016, 1222), radius=24, fill=accent)
    text_block(d, (104, 1138), cta, font(FONT_BOLD, 36), (255, 255, 255), 850, 6)

    d.text((64, 1264), "archimedesmd.com", font=font(FONT_BOLD, 38), fill=ink)
    d.text((64, 1310), "Messy exports -> action reports.", font=font(FONT_REG, 27), fill=muted)
    img.save(OUT / f"ig_{slug}.jpg", quality=92)


card(
    "01_data_organization",
    "Data organization",
    "Messy data is still valuable.",
    "Exports, notes, CSVs, and reports spread across tools.",
    "Archimedes cleans the structure so the real problem can be analyzed.",
    "Upload messy data. Get something your team can use.",
    (0, 132, 150),
)

card(
    "02_reports",
    "Reports",
    "A chart is not a decision.",
    "Dashboards show movement, but rarely say what to do next.",
    "Archimedes turns organized data into focused reports with risks, segments, and next steps.",
    "Stop staring at rows. Start reviewing the action report.",
    (43, 94, 170),
)

card(
    "03_predictive_evaluations",
    "Predictive evaluations",
    "Past data should guide the next move.",
    "Most teams collect data, then leave the future buried inside it.",
    "Archimedes supports model training and predictive evaluation from cleaned business data.",
    "Rank risk. Spot patterns. Test likely outcomes.",
    (92, 75, 168),
)

card(
    "04_combined_workflow",
    "Full workflow",
    "One workflow. Three jobs.",
    "Organize the mess. Build the report. Evaluate what might happen next.",
    "Archimedes connects data organization, reporting, and predictive evaluation into one operating workflow.",
    "Not another CSV chatbot. The first analyst pass.",
    (23, 139, 91),
)
