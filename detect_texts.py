import cv2
import math
import pandas as pd


def get_horizontal_list(img, text_score_map, link_score_map, thr=300):
    _, text_mask = cv2.threshold(src=text_score_map, thresh=120, maxval=255, type=cv2.THRESH_BINARY)
    _, link_mask = cv2.threshold(src=link_score_map, thresh=160, maxval=255, type=cv2.THRESH_BINARY)
    word_mask = text_mask + link_mask
    
    _, _, stats, _ = cv2.connectedComponentsWithStats(image=word_mask, connectivity=4)
    
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

    bboxes = bboxes[["xmin", "xmax", "ymin", "ymax"]]
    return bboxes.values.tolist()


# img = load_image_as_array("/Users/jongbeom.kim/Downloads/materials/original_image.jpg")
# craft = load_craft_checkpoint(cuda=False)
# text_score_map, link_score_map = get_text_score_map_and_link_score_map(img=img, craft=craft, cuda=False)

# thr = 300
# _, text_mask = cv2.threshold(src=text_score_map, thresh=120, maxval=255, type=cv2.THRESH_BINARY)
# _, link_mask = cv2.threshold(src=link_score_map, thresh=160, maxval=255, type=cv2.THRESH_BINARY)
# word_mask = text_mask + link_mask

# _, _, stats, _ = cv2.connectedComponentsWithStats(image=word_mask, connectivity=4)

# bboxes = pd.DataFrame(stats[1:, :], columns=["xmin", "ymin", "width", "height", "pixel_count"])

# bboxes = bboxes[bboxes["pixel_count"].ge(thr)]

# bboxes["xmax"] = bboxes["xmin"] + bboxes["width"]
# bboxes["ymax"] = bboxes["ymin"] + bboxes["height"]

# bboxes["margin"] = bboxes.apply(
#     lambda x: int(
#         math.sqrt(
#             x["pixel_count"] * min(x["width"], x["height"]) / (x["width"] * x["height"])
#         ) * 2.2
#     ), axis=1
# )
# bboxes["xmin"] = bboxes.apply(
#     lambda x: max(0, x["xmin"] - x["margin"]), axis=1
# )
# bboxes["ymin"] = bboxes.apply(
#     lambda x: max(0, x["ymin"] - x["margin"]), axis=1
# )
# bboxes["xmax"] = bboxes.apply(
#     lambda x: min(img.shape[1], x["xmax"] + x["margin"]), axis=1
# )
# bboxes["ymax"] = bboxes.apply(
#     lambda x: min(img.shape[0], x["ymax"] + x["margin"]), axis=1
# )
# bboxes = bboxes[["xmin", "ymin", "xmax", "ymax"]]

# import easyocr
# reader = easyocr.Reader(lang_list=["ko"], gpu=False)
# rect = reader.detect(img)
# bboxes2 = pd.DataFrame(rect[0][0], columns=["xmin", "xmax", "ymin", "ymax"])

# img = load_image_as_array("/Users/jongbeom.kim/Downloads/materials/original_image.jpg")
# drawn = draw_rectangles_on_image(img, bboxes, bboxes2)
# show_image(drawn)
