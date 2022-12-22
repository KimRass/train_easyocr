from pathlib import Path
import extcolors
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import easyocr
import argparse

# from process_image import (
    # convert_quadrilaterals_to_rectangles
# )
from prepare_data import (
    get_image_and_label
)
# from recognize_texts import (
    # get_paddleocr_result,
#     get_easyocr_result
# )
from craft_utilities import (
    load_craft_checkpoint,
    load_craft_refiner_checkpoint
)
from evaluate import (
    get_f1_score
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


def spot_texts_baseline(img, reader, rectangle=True):
    result = reader.readtext(img)

    cols = ["text", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]
    if result:
        df_pred = pd.DataFrame(
            [(row[1], *list(map(int, sum(row[0], [])))) for row in result],
            columns=cols
        )
    else:
        df_pred = pd.DataFrame(columns=cols)

    if rectangle:
        df_pred = convert_quadrilaterals_to_rectangles(df_pred)
    return df_pred


def main():
    args = get_arguments()

    # craft = load_craft_checkpoint(cuda=False)
    # craft_refiner = load_craft_refiner_checkpoint(cuda=False)
    craft = load_craft_checkpoint(cuda=args.cuda)
    craft_refiner = load_craft_refiner_checkpoint(cuda=args.cuda)

    # reader = easyocr.Reader(lang_list=["ko", "en"], gpu=False)
    reader = easyocr.Reader(lang_list=["ko", "en"], gpu=args.cuda)

    # dir = Path("/Users/jongbeom.kim/Documents/New_sample/라벨링데이터")
    dir = Path(args.dir)
    ls = list()
    for path_json in tqdm(sorted(list(dir.glob("**/*.json")))):
        # path_json = "/Users/jongbeom.kim/Documents/New_sample/라벨링데이터/인.허가/5350109/1994/5350109-1994-0001-0010.json"
        img, gt = get_image_and_label(path_json=path_json)

        """ Baseline """
        result = spot_texts_baseline(img=img, reader=reader, rectangle=True)
        result.to_excel(f"{dir}/baseline/{path_json.stem}.xlsx", index=False)

        f1_score = get_f1_score(gt, result, iou_thr=0.5, rec=True)

        ls.append((path_json.stem, f1_score))

        df = pd.DataFrame(ls, columns=["file", "f1_score_baseline"])
        df.to_excel(f"{dir}/baseline.xlsx", index=False)

        # """ Ours """

        # show_image(img)
        # text_score_map, link_score_map = get_text_score_map_and_link_score_map(
        #     img=img, craft=craft, cuda=False
        # )
        # show_image(text_score_map, img)
        # rectangles = get_word_level_bounding_boxes(img, text_score_map, link_score_map, thr=300)
        # rectangles.head()


if __name__ == "__main__":
    main()