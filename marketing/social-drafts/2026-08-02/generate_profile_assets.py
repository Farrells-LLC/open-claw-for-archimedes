from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).parent
W = H = 1024

FONT_BOLD = "/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf"
FONT_REG = "/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf"


def font(path, size):
    return ImageFont.truetype(path, size=size)


img = Image.new("RGB", (W, H), (12, 29, 48))
d = ImageDraw.Draw(img)

accent = (0, 145, 160)
green = (31, 151, 105)
white = (255, 255, 255)
muted = (188, 208, 225)

d.rounded_rectangle((92, 92, 932, 932), radius=120, fill=(245, 247, 250))
d.rounded_rectangle((138, 138, 886, 886), radius=94, fill=(12, 29, 48))

d.ellipse((226, 206, 798, 778), outline=accent, width=28)
d.arc((260, 240, 764, 744), start=200, end=340, fill=green, width=22)
d.arc((260, 240, 764, 744), start=20, end=160, fill=accent, width=22)

mark_font = font(FONT_BOLD, 330)
text = "A"
bbox = d.textbbox((0, 0), text, font=mark_font)
d.text(((W - (bbox[2] - bbox[0])) / 2, 288), text, font=mark_font, fill=white)

name_font = font(FONT_BOLD, 64)
name = "ARCHIMEDES"
bbox = d.textbbox((0, 0), name, font=name_font)
d.text(((W - (bbox[2] - bbox[0])) / 2, 790), name, font=name_font, fill=white)

img.save(OUT / "archimedes_profile_picture.png", quality=95)
