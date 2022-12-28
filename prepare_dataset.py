import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

from process_image import (
    load_image_as_array,
    save_image,
    get_image_cropped_by_rectangle
)


def get_arguments():
    parser = argparse.ArgumentParser(description="prepare_dataset")

    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir")

    args = parser.parse_args()
    return args


def parse_json_file(json_path):
    # json_path = Path(json_path)

    with open(json_path, mode="r") as f:
        label = json.load(f)
    
    # img_path = f"{json_path.parents[4]}/원천데이터/인.허가/{json_path.parent.parent.stem}/{json_path.parent.stem}/{label['images'][0]['image.file.name']}"
    img_path = str(json_path).replace("/labels/", "/images/").replace(".json", ".jpg")
    img = load_image_as_array(img_path)

    gt_bboxes = np.array(
        [i["annotation.bbox"] for i in label["annotations"]]
    )
    gt_bboxes[:, 2] += gt_bboxes[:, 0]
    gt_bboxes[:, 3] += gt_bboxes[:, 1]

    gt_texts = np.array(
        [i["annotation.text"] for i in label["annotations"]]
    )
    return img, gt_bboxes, gt_texts


# def get_image(json_path):
#     json_path = Path(json_path)

#     with open(json_path, mode="r") as f:
#         label = json.load(f)

#     img_path = f"{json_path.parents[4]}/원천데이터/인.허가/{json_path.parent.parent.stem}/{json_path.parent.stem}/{label['images'][0]['image.file.name']}"

#     img = load_image_as_array(img_path)
#     return img


def get_image_and_label(json_path):
    json_path = Path(json_path)

    with open(json_path, mode="r") as f:
        label = json.load(f)

    gt_bboxes = np.array(
        [i["annotation.bbox"] for i in label["annotations"]]
    )
    gt_texts = np.array(
        [i["annotation.text"] for i in label["annotations"]]
    )

    ls_row = list()
    for sample in label["annotations"]:
        text = sample["annotation.text"]
        xmin, ymin, width, height = sample["annotation.bbox"]
        
        ls_row.append([text, xmin, ymin, xmin + width, ymin + height])
    df_label = pd.DataFrame(ls_row, columns=["text", "xmin", "ymin", "xmax", "ymax"])

    img_path = f"{json_path.parents[4]}/원천데이터/인.허가/{json_path.parent.parent.stem}/{json_path.parent.stem}/{label['images'][0]['image.file.name']}"

    img = load_image_as_array(img_path)
    return img, df_label


def create_dataset(input_dir, output_dir) -> None:
    input_dir = "/Users/jongbeom.kim/Documents/ocr/"
    output_dir = "/Users/jongbeom.kim/Documents/dataset"
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    ls_row = list()
    for json_path in tqdm(list(input_dir.glob("**/*.json"))):
        try:
            img, gt_bboxes, gt_texts = parse_json_file(json_path)
        except Exception:
            continue

        for text, (xmin, ymin, xmax, ymax) in zip(gt_texts, gt_bboxes):
            split1 = "training" if "training" in str(json_path) else "validation"
            split2 = "select_data"

            patch = get_image_cropped_by_rectangle(
                img=img, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax
            )
            fname = Path(f"{json_path.stem}_{xmin}-{ymin}-{xmax}-{ymax}.png")
            save_image(img=patch, path=output_dir/split1/split2/"images"/fname)

            # with open(output_dir/split1/split2/"gt.txt", mode="a") as f:
            #     f.write(f"{fname}\t{text}\n")
            #     f.close()

    for path_txt in output_dir.glob("**/*.txt"):
        df = pd.DataFrame(
            [line.strip().split("\t") for line in open(path_txt, mode="r").readlines()],
            columns=["filename", "words"]
        )
        df.to_csv(path_txt.parent/"labels.csv", index=False)


if __name__ == "__main__":
    args = get_arguments()

    create_dataset(input_dir=args.input_dir, output_dir=args.output_dir)
