import cv2
import math
import pandas as pd
import numpy as np
import easyocr

from process_image import(
    get_masked_image
)


def get_word_level_bounding_boxes(img, text_score_map, link_score_map, thr=300):
    _, text_mask = cv2.threshold(src=text_score_map, thresh=120, maxval=255, type=cv2.THRESH_BINARY)
    _, link_mask = cv2.threshold(src=link_score_map, thresh=160, maxval=255, type=cv2.THRESH_BINARY)
    word_mask = text_mask + link_mask
    
    _, segmap_word, stats, _ = cv2.connectedComponentsWithStats(image=word_mask, connectivity=4)
    segmap_word = get_masked_image(img=segmap_word, mask=link_mask, invert=True)
    
    bboxes = pd.DataFrame(stats[1:, :], columns=["xmin", "ymin", "width", "height", "pixel_count"])

    bboxes = bboxes[bboxes["pixel_count"].ge(thr)]
    
    bboxes["xmax"] = bboxes["xmin"] + bboxes["width"]
    bboxes["ymax"] = bboxes["ymin"] + bboxes["height"]

    bboxes["margin"] = bboxes.apply(
        lambda x: int(
            math.sqrt(
                x["pixel_count"] * min(x["width"], x["height"]) / (x["width"] * x["height"])
            ) * 2.2
        ), axis=1
    )
    bboxes["xmin"] = bboxes.apply(
        lambda x: max(0, x["xmin"] - x["margin"]), axis=1
    )
    bboxes["ymin"] = bboxes.apply(
        lambda x: max(0, x["ymin"] - x["margin"]), axis=1
    )
    bboxes["xmax"] = bboxes.apply(
        lambda x: min(img.shape[1], x["xmax"] + x["margin"]), axis=1
    )
    bboxes["ymax"] = bboxes.apply(
        lambda x: min(img.shape[0], x["ymax"] + x["margin"]), axis=1
    )

    bboxes = bboxes[["xmin", "ymin", "xmax", "ymax"]]
    return bboxes


def detect_texts_baseline(img, reader):
    result = reader.detect(img)
    bboxes_pred = np.array(result[0][0])
    return bboxes_pred
    # cols = ["xmin", "xmax", "ymin", "ymax"]
    # if result:
    #     rectangles = pd.DataFrame(result[0][0], columns=cols)
    # else:
    #     rectangles = pd.DataFrame(columns=cols)
    # return rectangles


def detect_texts(img, reader):
    # reader = easyocr.Reader(lang_list=["ko", "en"], gpu=cuda)
    result = reader.detect(
        img=img,
        min_size=20,
        text_threshold=0.7,
        low_text=0.4,
        link_threshold=0.4,
        canvas_size=2560,
        # slope_ths=0, # No merge
        # ycenter_ths=0.5,
        # height_ths=0.5,
        width_ths=0 # No merge
    )
    cols = ["xmin", "xmax", "ymin", "ymax"]
    if result:
        rectangles = pd.DataFrame(result[0][0], columns=cols)
    else:
        rectangles = pd.DataFrame(columns=cols)
    return rectangles
