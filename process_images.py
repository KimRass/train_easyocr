import cv2
from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


def load_image_as_array(img_path="", gray=False):
    img_path = str(img_path)

    if not gray:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
    else:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return img


def save_image(img, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if img.ndim == 3:
        cv2.imwrite(filename=str(path), img=img[:, :, :: -1], params=[cv2.IMWRITE_JPEG_QUALITY, 100])
    elif img.ndim == 2:
        cv2.imwrite(filename=str(path), img=img, params=[cv2.IMWRITE_JPEG_QUALITY, 100])


def show_image(img1, img2=None, alpha=0.5):
    plt.figure(figsize=(11, 9))
    plt.imshow(img1)
    if img2 is not None:
        plt.imshow(img2, alpha=alpha)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def convert_to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img


def convert_to_array(img):
    img = np.array(img)
    return img


def draw_easyocr_result(img, bboxes):
    img_copied = convert_to_pil(img.copy())

    draw = ImageDraw.Draw(img_copied)
    for xmin, ymin, xmax, ymax, text in bboxes.values:
        draw.rectangle(xy=(xmin, ymin, xmax, ymax), outline=(255, 0, 0), width=2)
        draw.text(
            xy=(xmin, ymin - 4),
            text=text,
            fill=(255, 0, 0),
            font=ImageFont.truetype(font="fonts/NanumSquareNeo-bRg.ttf", size=26),
            anchor="ls"
        )
    img_copied = convert_to_array(img_copied)
    return img_copied


def get_image_cropped_by_rectangle(img, xmin, ymin, xmax, ymax):
    if img.ndim == 3:
        return img[ymin: ymax, xmin: xmax, :]
    else:
        return img[ymin: ymax, xmin: xmax]
