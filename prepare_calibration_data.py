import os
import shutil
import sys
import numpy as np
from PIL import Image

_here = os.path.dirname(os.path.abspath(__file__))

dataset_path = os.path.join(_here, "datasets", "coco128", "images", "train2017")
imgsz = 640
for arg in sys.argv[1:]:
    if arg.startswith("path="):
        in_path = arg.split("=", 1)[1]
        dataset_path = (
            in_path if os.path.isabs(in_path) else os.path.join(_here, in_path)
        )
    elif arg.startswith("imgsz="):
        imgsz = int(arg.split("=", 1)[1])

calib_root = os.path.join(_here, "calibration_dataset")
os.makedirs(calib_root, exist_ok=True)

calib_dir = os.path.join(calib_root, str(imgsz))
if os.path.isdir(calib_dir):
    shutil.rmtree(calib_dir)
os.makedirs(calib_dir)

image_files = sorted(
    [f for f in os.listdir(dataset_path) if f.endswith((".jpg", ".jpeg", ".png"))]
)

for idx, img_file in enumerate(image_files):
    img_path = os.path.join(dataset_path, img_file)

    img = Image.open(img_path).convert("RGB")
    img = img.resize((imgsz, imgsz), Image.BILINEAR)

    im = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
    im = np.expand_dims(im, axis=0)

    np.save(os.path.join(calib_dir, f"batch-{idx:05d}.npy"), im)
    print(f"Saved batch {idx:05d}: {img_file}")

print(f"Done. {len(image_files)} batches saved to {calib_dir}/")
