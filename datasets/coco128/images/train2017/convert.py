import cv2
import numpy as np
import os


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    new_h, new_w = new_shape

    r = min(new_h / h, new_w / w)

    resized_w = int(round(w * r))
    resized_h = int(round(h * r))

    img = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    dw = new_w - resized_w
    dh = new_h - resized_h

    left = int(round(dw / 2 - 0.1))
    right = int(round(dw / 2 + 0.1))
    top = int(round(dh / 2 - 0.1))
    bottom = int(round(dh / 2 + 0.1))

    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    return img


def image_to_bin_uint8(image_path, bin_path):
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(image_path)

    # 1. Resize + padding to 640x640.
    img = letterbox(img, (640, 640))

    # 2. BGR -> RGB.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 3. HWC -> CHW.
    img = img.transpose(2, 0, 1)

    # 4. Add batch dimension to get 1x3x640x640.
    img = np.expand_dims(img, axis=0)

    # 5. Keep uint8 values without dividing by 255.
    img = img.astype(np.uint8)

    print("shape:", img.shape)
    print("dtype:", img.dtype)
    print("num elements:", img.size)
    print("expected elements:", 1 * 3 * 640 * 640)

    img.tofile(bin_path)

    file_size = os.path.getsize(bin_path)
    print("bin file size:", file_size, "bytes")

    if file_size == 1228800:
        print("OK: file size matches 1228800")
    else:
        print("ERROR: file size mismatch")


if __name__ == "__main__":
    image_to_bin_uint8("000000000009.jpg", "input.bin")
