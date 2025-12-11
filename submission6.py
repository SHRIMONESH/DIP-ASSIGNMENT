import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------
# 🔥 CHANGE THIS LINE TO YOUR IMAGE FILE
IMAGE_PATH = "C:\\Users\\Arjun\\Downloads\\q5.jpg"
         # <-- put your file name here
# ---------------------------------------

# Output folder
OUT_DIR = Path("bitplane_output")
OUT_DIR.mkdir(exist_ok=True)

def to_gray(img):
    if img.mode != 'L':
        img = img.convert('L')
    return np.array(img, dtype=np.uint8)

def bitplanes(img):
    """Return bit-planes [BP8..BP1]"""
    planes = []
    for bit in range(8):
        plane = ((img >> bit) & 1) * 255
        planes.append(plane)
    return planes[::-1]

def reconstruct(planes, bits):
    """Reconstruct image using bit indices (0 = LSB)"""
    rec = np.zeros_like(planes[0], dtype=np.uint8)
    for b in bits:
        idx = 7 - b
        rec |= ((planes[idx] > 0).astype(np.uint8) << b)
    return rec

def save(name, arr):
    Image.fromarray(arr).save(OUT_DIR / name)

# ---------------------------------------
# 1. Load user image
# ---------------------------------------
print("Loading:", IMAGE_PATH)
img = Image.open(IMAGE_PATH)
orig = to_gray(img)

# Create low-light & bright-light images
low_img = ImageEnhance.Brightness(Image.fromarray(orig)).enhance(0.45)
bright_img = ImageEnhance.Brightness(Image.fromarray(orig)).enhance(1.6)

low = to_gray(low_img)
bright = to_gray(bright_img)

save("original.png", orig)
save("low.png", low)
save("bright.png", bright)

# ---------------------------------------
# 2. Compute bit-planes
# ---------------------------------------
planes = bitplanes(orig)
for i, p in enumerate(planes, start=8):
    save(f"BP{i}.png", p)

# ---------------------------------------
# 3. Reconstruct using lowest 3 bit-planes
# ---------------------------------------
lowest_bits = [0, 1, 2]

rec = reconstruct(planes, lowest_bits)
save("reconstructed_low3.png", rec)

# ---------------------------------------
# 4. Difference image
# ---------------------------------------
diff = np.abs(orig.astype(int) - rec.astype(int)).astype(np.uint8)
save("difference_low3.png", diff)

print("\n✔ Done! Files saved in folder:", OUT_DIR)
print("Generated images:")
for f in OUT_DIR.iterdir():
    print(" -", f.name)
