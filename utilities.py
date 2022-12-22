# import argparse
import math
import numpy as np
import pandas as pd
from time import time
from datetime import timedelta
from scipy.sparse import coo_matrix
import cv2
from skimage.feature import peak_local_max
import re

from process_image import(
    load_image_as_array,
    convert_2d_to_3d,
    get_adaptive_thresholded_image,
    get_masked_image,
    thicken_image,
    get_pixel_count
)


def get_local_maxima_coordinates(text_score_map):
    _, text_mask = cv2.threshold(text_score_map, thresh=120, maxval=255, type=cv2.THRESH_BINARY)
    _, text_segmentation_map = cv2.connectedComponents(image=text_mask, connectivity=4)
    local_max = peak_local_max(
        image=text_score_map, min_distance=5, labels=text_segmentation_map, num_peaks_per_label=8
    )
    local_max = local_max[:, ::-1]
    return local_max


def get_local_maxima_array(text_score_map):
    local_max_coor = get_local_maxima_coordinates(text_score_map)

    vals = np.array([1] * local_max_coor.shape[0])
    rows = local_max_coor[:, 1]
    cols = local_max_coor[:, 0]
    local_max = coo_matrix(
        (vals, (rows, cols)), shape=text_score_map.shape
    ).toarray().astype("bool")
    return local_max


# def get_text_segmentation_map(text_score_map, text_thresh=30):
#     local_max_arr = get_local_maxima_array(text_score_map)
#     markers = ndimage.label(local_max_arr, structure=np.ones((3, 3)))[0]

#     _, text_mask = cv2.threshold(
#         text_score_map, thresh=text_thresh, maxval=255, type=cv2.THRESH_BINARY
#     )
#     segmap = watershed(image=-text_score_map, markers=markers, mask=text_mask)
#     return segmap


def split_text_segmentation_map_into_two(text_segmentation_map, centers):
    ls_idx = centers[centers["inside"]].apply(
        lambda x: text_segmentation_map[x["y"], x["x"]], axis=1
    ).values.tolist()

    segmap_in = get_masked_image(img=text_segmentation_map, mask=np.isin(text_segmentation_map, ls_idx))
    segmap_out = get_masked_image(img=text_segmentation_map, mask=~np.isin(text_segmentation_map, ls_idx))
    return segmap_in, segmap_out


def get_line_segmentation_map(line_score_map):
    _, line_mask = cv2.threshold(src=line_score_map, thresh=130, maxval=255, type=cv2.THRESH_BINARY)
    _, line_segmentation_map = cv2.connectedComponents(image=line_mask, connectivity=4)
    return line_segmentation_map


# def get_3d_block_segmentation_map(img, rectangles):
#     segmap_block = np.zeros(shape=(img.shape[0], img.shape[1], len(rectangles) + 1))
#     for idx, (xmin, ymin, xmax, ymax) in enumerate(
#         rectangles[["xmin", "ymin", "xmax", "ymax"]].values,
#         start=1
#     ):
#         segmap_block[ymin: ymax, xmin: xmax, idx] = 255
#     return segmap_block


def get_pseudo_character_centers(text_score_map, rectangles):
    local_max_coor = get_local_maxima_coordinates(text_score_map)
    df_center = pd.DataFrame(local_max_coor, columns=["x", "y"])

    ls_idx = list()
    for idx, (x, y) in enumerate(df_center.values):
        for xmin, ymin, xmax, ymax in rectangles[["xmin", "ymin", "xmax", "ymax"]].values:
            if xmin <= x <= xmax and ymin <= y <= ymax:
                ls_idx.append(idx)
                break
    df_center["inside"] = df_center.index.isin(ls_idx)
    return df_center


def get_mask_that_overlaps_chracter_segmentation_map(
    text_segmentation_map, image_segmentation_map, overlap=0.6
):
    cnts_img = get_pixel_count(image_segmentation_map, sort=True, include_zero=False)

    segmap_overlap = get_masked_image(img=image_segmentation_map, mask=(text_segmentation_map != 0))
    cnts_overlap = get_pixel_count(segmap_overlap, sort=False, include_zero=False)

    df_cnts = pd.DataFrame.from_dict(cnts_img, orient="index", columns=["total_pixel_count"])
    df_cnts["overlap_pixel_count"] = df_cnts.apply(
        lambda x: cnts_overlap.get(x.name, 0), axis=1
    )
    df_cnts["ratio"] = df_cnts["overlap_pixel_count"] / df_cnts["total_pixel_count"]

    ls_idx = df_cnts[df_cnts["ratio"] > overlap].index.tolist()

    mask = np.isin(image_segmentation_map, ls_idx).astype("uint8")
    mask = convert_2d_to_3d(mask * 255)
    return mask


def make_segmentation_map_rectangle(segmentation_map):
    segmentation_map_copied = segmentation_map.copy()
    for idx in range(1, np.max(segmentation_map_copied) + 1):
        segmentation_map_sub = (segmentation_map_copied == idx)
        nonzero_x = np.where((segmentation_map_sub != 0).any(axis=0))[0]
        nonzero_y = np.where((segmentation_map_sub != 0).any(axis=1))[0]
        if nonzero_x.size != 0 and nonzero_y.size != 0:
            segmentation_map_copied[nonzero_y[0]: nonzero_y[-1], nonzero_x[0]: nonzero_x[-1]] = idx
    return segmentation_map_copied


def get_image_segmentation_map(img, text_score_map):
    _, text_mask = cv2.threshold(text_score_map, thresh=20, maxval=255, type=cv2.THRESH_BINARY)
    text_mask = thicken_image(img=text_mask, kernel_shape=(44, 44), iterations=1)
    img_masked = get_masked_image(img=img, mask=text_mask)

    img_thr1 = get_adaptive_thresholded_image(img_masked, invert=False, block_size=3)
    img_thr2 = get_adaptive_thresholded_image(img_masked, invert=True, block_size=3)
    
    n, segmap1 = cv2.connectedComponents(image=img_thr1, connectivity=4)
    _, segmap2 = cv2.connectedComponents(image=img_thr2, connectivity=4)

    segmap = segmap1 + get_masked_image(
        img=segmap2 + n - 1, mask=(segmap2 != 0)
    )

    cnts = get_pixel_count(segmap, sort=True, include_zero=True)
    segmap = get_masked_image(img=segmap, mask=(segmap != list(cnts)[0]))
    return segmap


def get_mask_for_texts_inside_rectangles(img, text_score_map, rectangles, overlap=0.5):
    segmap_text = get_text_segmentation_map(text_score_map, text_thresh=30)
    df_center = get_pseudo_character_centers(text_score_map=text_score_map, rectangles=rectangles)
    segmap_text_in, segmap_text_out = split_text_segmentation_map_into_two(
        text_segmentation_map=segmap_text, centers=df_center
    )

    segmap_img = get_image_segmentation_map(img=img, text_score_map=text_score_map)

    mask_in = get_mask_that_overlaps_chracter_segmentation_map(
        text_segmentation_map=segmap_text_in, image_segmentation_map=segmap_img, overlap=overlap
    )
    mask_out = get_mask_that_overlaps_chracter_segmentation_map(
        text_segmentation_map=segmap_text_out, image_segmentation_map=segmap_img, overlap=overlap
    )
    return mask_in, mask_out


def get_elapsed_time(time_start):
    return timedelta(seconds=round(time() - time_start))


def convert_csv_to_rectangles2(csv_path, resize=False):
    df = pd.read_csv(csv_path)

    ls_rect = list()
    for coor, content in df[["coordinates", "content"]].values:
        coor = re.sub(pattern=r"\(|\)", repl="", string=coor)
        coor = coor.split(",")

        rect = list(map(int, coor))
        ls_rect.append((rect[2], rect[3], rect[0], rect[1], content))
    rectangles = pd.DataFrame(ls_rect, columns=["xmin", "ymin", "xmax", "ymax", "transcript"])

    rectangles["area"] = rectangles.apply(
        lambda x: (x["xmax"] - x["xmin"]) * (x["ymax"] - x["ymin"]), axis=1
    )
    rectangles.sort_values(["area"], inplace=True)
    rectangles.drop(["area"], axis=1, inplace=True)

    img = load_image_as_array(df["image_url"].values[0])

    if resize:
        rectangles, img = resize_coordinates_and_image_to_fit_to_maximum_pixel_counts(
            rectangles=rectangles, img=img
        )
    return rectangles, img
