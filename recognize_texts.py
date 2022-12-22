# from paddleocr import PaddleOCR
import easyocr
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from process_image import (
    get_image_cropped_by_rectangle,
    convert_to_pil,
)


# def get_paddleocr_result(
#     img,
#     lang="en",
#     text_detection=True,
#     text_recognition=True,
#     rectangle=False
# ):
#     if lang == "ko":
#         lang = "korean"
#     elif lang == "ja":
#         lang = "japan"

#     ocr = PaddleOCR(lang=lang)
#     result = ocr.ocr(img=img, det=text_detection, rec=text_recognition, cls=False)

#     if text_detection and not text_recognition:
#         df_pred = pd.DataFrame(
#             np.array(result).reshape(-1, 8),
#             columns=["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"],
#             dtype="int"
#         )
#     elif not text_detection and text_recognition:
#         return pd.DataFrame(
#             result,
#             columns=["text", "confidence"]
#         )
#     elif text_detection and text_recognition:
#         cols = ["text", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
#         if result:
#             df_pred = pd.DataFrame(
#                 [(row[1][0], *list(map(int, sum(row[0], [])))) for row in result],
#                 columns=cols
#             )
#         else:
#             df_pred = pd.DataFrame(columns=cols)

#     if rectangle:
#         df_pred = convert_quadrilaterals_to_rectangles(df_pred)
#     return df_pred


def recognize_texts(img, reader):
    # reader = easyocr.Reader(lang_list=["ko", "en"], gpu=cuda)
    bw=3
    result = reader.recognize(
        img_cv_grey=img,
        # img_cv_grey=patch,
        decoder="beamsearch",
        beamWidth=bw,
        workers=2,
        detail=0,
        paragraph=False,
    )
    return result


def add_transcript(img, rectangles, reader):
    ls_transcript = list()
    for xmin, ymin, xmax, ymax in tqdm(rectangles[["xmin", "ymin", "xmax", "ymax"]].values):
        patch = get_image_cropped_by_rectangle(
            img=img, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax
        )
        transcript = recognize_texts(img=patch, reader=reader)

        ls_transcript.append(transcript)
    rectangles["transcript"] = ls_transcript
    return rectangles
