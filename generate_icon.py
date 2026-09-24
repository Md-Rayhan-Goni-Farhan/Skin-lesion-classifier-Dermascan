"""
Run once to generate icon.ico from scratch using Pillow.
Output: assets/icon.ico
"""
from PIL import Image, ImageDraw, ImageFont
import os

SIZE = 256
img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Background circle
draw.ellipse([4, 4, SIZE - 4, SIZE - 4], fill="#00c896")

# White cross (medical)
m = SIZE // 2
arm = SIZE // 5
thick = SIZE // 8
draw.rectangle([m - thick // 2, m - arm, m + thick // 2, m + arm], fill="white")
draw.rectangle([m - arm, m - thick // 2, m + arm, m + thick // 2], fill="white")

out = os.path.join(os.path.dirname(__file__), "icon.ico")
img.save(out, format="ICO", sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print(f"Saved {out}")
