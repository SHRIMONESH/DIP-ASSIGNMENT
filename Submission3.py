# sampling_demo_two_images.py
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# Replace these with your actual images (car and lion)
car_path = r"C:\Users\Arjun\Desktop\wfp.png"
lion_path = r"C:\Users\Arjun\Desktop\lion.png"

out_dir = "sampling_outputs_two_images"
os.makedirs(out_dir, exist_ok=True)

factors = [2, 4, 8, 16]

def lowpass_mask(shape, keep_w, keep_h):
    H, W = shape
    mask = np.zeros((H, W), dtype=np.float32)
    cy, cx = H//2, W//2
    hy, hx = keep_h//2, keep_w//2
    y1, y2 = max(0, cy-hy), min(H, cy+hy + (1 if keep_h%2 else 0))
    x1, x2 = max(0, cx-hx), min(W, cx+hx + (1 if keep_w%2 else 0))
    mask[y1:y2, x1:x2] = 1.0
    return mask

# Load
img_car = Image.open(car_path).convert("RGB")
img_lion_gray = Image.open(lion_path).convert("L")

Wc, Hc = img_car.size

for f in factors:
    # Spatial sampling (car)
    nw, nh = max(1, Wc//f), max(1, Hc//f)
    down_nearest = img_car.resize((nw, nh), Image.NEAREST)
    up_nearest = down_nearest.resize((Wc, Hc), Image.NEAREST)
    up_nearest.save(os.path.join(out_dir, f"car_spatial_nearest_1_over_{f}.png"))

    down_area = img_car.resize((nw, nh), Image.BOX)
    up_area = down_area.resize((Wc, Hc), Image.BICUBIC)
    up_area.save(os.path.join(out_dir, f"car_spatial_area_1_over_{f}.png"))

    # Frequency sampling (lion)
    arr = np.array(img_lion_gray).astype(np.float32)
    H, W = arr.shape
    F = np.fft.fft2(arr)
    Fshift = np.fft.fftshift(F)
    keep_w, keep_h = max(1, W//f), max(1, H//f)
    mask = lowpass_mask((H, W), keep_w, keep_h)
    Fmasked = Fshift * mask
    F_ishift = np.fft.ifftshift(Fmasked)
    img_back = np.fft.ifft2(F_ishift)
    img_back = np.real(img_back)
    img_back -= img_back.min()
    if img_back.max() > 0:
        img_back = img_back / img_back.max() * 255.0
    img_back_uint8 = img_back.astype(np.uint8)
    Image.fromarray(img_back_uint8).convert("RGB").save(
        os.path.join(out_dir, f"lion_freq_lowpass_1_over_{f}.png")
    )

print("Saved outputs in:", out_dir)
