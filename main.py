from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import easyocr
import argparse

# from process_image import (
    # convert_quadrilaterals_to_rectangles
# )
from prepare_dataset import (
    parse_json_file,
    # get_image_and_label
)
# from recognize_texts import (
    # get_paddleocr_result,
#     get_easyocr_result
# )
from craft_utilities import (
    load_craft_checkpoint,
    load_craft_refiner_checkpoint,
    get_text_score_map_and_link_score_map
)
from evaluate import (
    # get_text_detection_f1_score,
    get_end_to_end_f1_score
)
from detect_texts import (
    get_word_level_bounding_boxes    
)
from recognize_texts import (
    add_transcript
)


def get_arguments():
    parser = argparse.ArgumentParser(description="ocr")

    parser.add_argument("--dir")
    parser.add_argument("--cuda", default=False, action="store_true")

    args = parser.parse_args()
    return args


def convert_quadrilaterals_to_rectangles(df):
    df.insert(1, "ymax", df[["y1", "y2", "y3", "y4"]].max(axis=1))
    df.insert(1, "xmax", df[["x1", "x2", "x3", "x4"]].max(axis=1))
    df.insert(1, "ymin", df[["y1", "y2", "y3", "y4"]].min(axis=1))
    df.insert(1, "xmin", df[["x1", "x2", "x3", "x4"]].min(axis=1))
    df.drop(["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"], axis=1, inplace=True)

    df[["xmin", "ymin", "xmax", "ymax"]] = df[["xmin", "ymin", "xmax", "ymax"]].astype("int")
    return df


# def spot_texts_baseline(img, reader, rectangle=True):
#     result = reader.readtext(img)

#     cols = ["text", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
#     if result:
#         df_pred = pd.DataFrame(
#             [(row[1], *list(map(int, sum(row[0], [])))) for row in result],
#             columns=cols
#         )
#     else:
#         df_pred = pd.DataFrame(columns=cols)

#     if rectangle:
#         df_pred = convert_quadrilaterals_to_rectangles(df_pred)
#     return df_pred


def spot_texts(img, reader):
    result2 = reader.readtext(img)

    ls_bbox = list()
    ls_text = list()
    for ((x1, y1), (x2, y2), (x3, y3), (x4, y4)), text, _ in result2:
        if x1.dtype != "int64":
            continue

        xmin, _, _, xmax = sorted([x1, x2, x3, x4])
        ymin, _, _, ymax = sorted([y1, y2, y3, y4])
        ls_bbox.append((xmin, ymin, xmax, ymax))
        ls_text.append(text)
    bboxes_pred = np.array(ls_bbox)
    texts_pred = np.array(ls_text)
    return bboxes_pred, texts_pred


def main():
    args = get_arguments()

    # craft = load_craft_checkpoint(cuda=False)
    # craft = load_craft_checkpoint(cuda=args.cuda)

    # reader = easyocr.Reader(lang_list=["ko"], gpu=False)
    reader = easyocr.Reader(lang_list=["ko"], gpu=args.cuda)

    # dir = Path("/Users/jongbeom.kim/Documents/New_sample/라벨링데이터")
    dir = Path("/home/ubuntu/project/New_sample/라벨링데이터")

    sum_f1 = 0
    for idx, path_json in enumerate(
        tqdm(sorted(list(dir.glob("**/*.json"))))
    ):
        # path_json = "/Users/jongbeom.kim/Documents/New_sample/라벨링데이터/인.허가/5350109/1994/5350109-1994-0001-0010.json"
        
        try:
            img, gt_bboxes, gt_texts = parse_json_file(path_json)
        except Exception:
            continue

        """ Baseline """
        pred_bboxes, pred_texts = spot_texts(img=img, reader=reader)
        f1 = get_end_to_end_f1_score(gt_bboxes, gt_texts, pred_texts, pred_bboxes, iou_thr=0.5, rec=True)
        sum_f1 += f1
        
        if idx % 100 == 0 and idx != 0:
            print(sum_f1 / (idx + 1))
    print(sum_f1 / (idx + 1))

        # result = spot_texts_baseline(img=img, reader=reader, rectangle=True)
        # result.to_excel(f"{dir.parent}/result/baseline/{path_json.stem}.xlsx", index=False)

        # f1_det = get_text_detection_f1_score(gt_bboxes=gt, pred_bboxes=result)
        # f1_e2e_true = get_end_to_end_f1_score(gt_bboxes=gt, pred_bboxes=result, iou_thr=0.5, rec=True)
        # f1_e2e_false = get_end_to_end_f1_score(gt_bboxes=gt, pred_bboxes=result, iou_thr=0.5, rec=False)

        # ls.append((path_json.stem, f1_det, f1_e2e_true, f1_e2e_false))

        # df = pd.DataFrame(ls, columns=["file", "f1_det", "f1_e2e_true", "f1_e22_false"])
        # df.to_excel(f"{dir.parent}/result/baseline.xlsx", index=False)

        # """ Ours """
        # text_score_map, link_score_map = get_text_score_map_and_link_score_map(
        #     img=img, craft=craft, cuda=args.cuda
        # )

        # pred_bboxes = get_word_level_bounding_boxes(
        #     img=img, text_score_map=text_score_map, link_score_map=link_score_map, thr=300
        # )
        # rectangles = add_transcript(img=img, rectangles=rectangles, reader=reader)


if __name__ == "__main__":
    main()
