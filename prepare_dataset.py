import json
import argparse
import numpy as np
import pandas as pd
from zipfile import ZipFile
from pathlib import Path
from tqdm.auto import tqdm
import re

from process_image import (
    load_image_as_array,
    save_image,
    get_image_cropped_by_rectangle
)


def get_arguments():
    parser = argparse.ArgumentParser(description="prepare_dataset")

    # parser.add_argument("--input_dir")
    # parser.add_argument("--output_dir")
    parser.add_argument("--dataset")

    args = parser.parse_args()
    return args


def parse_json_file(json_path):
    with open(json_path, mode="r") as f:
        label = json.load(f)

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


def _unzip(zip_file, unzip_to):
    with ZipFile(zip_file, mode="r") as zip_obj:
        ls_member = zip_obj.infolist()
        # ls_member = ls_member[::20]

        for member in tqdm(ls_member):
            try:
                member.filename = member.filename.encode("cp437").decode("euc-kr", "ignore")
            except Exception:
                print(member.filename)
                continue

            member.filename = member.filename.lower()
            member.filename = re.sub(
                pattern=r"[0-9.가-힣a-z() ]+원천[0-9.가-힣a-z() ]+",
                repl="images",
                string=member.filename
            )
            member.filename = re.sub(
                pattern=r"[0-9.가-힣a-z() ]+라벨[0-9.가-힣a-z() ]+",
                repl="labels",
                string=member.filename
            )
            zip_obj.extract(member=member, path=unzip_to)


def unzip_dataset(dataset_dir) -> None:
    dataset_dir = Path(dataset_dir)

    for zip_file in tqdm(list(dir.glob("**/*.zip"))):
        unzip_to = Path(str(zip_file).replace("공공행정문서 OCR", "unzipped").lower()).parent
        unzip_to.mkdir(parents=True, exist_ok=True)

        _unzip(zip_file=zip_file, unzip_to=unzip_to)


def create_dataset_for_training(input_dir, output_dir) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    split2 = "select_data"
    ls_row = list()
    for subdir in tqdm(list(input_dir.glob("*"))):
        split1 = subdir.name
        save_dir = output_dir/split1/split2
        for json_path in tqdm(list((subdir/"labels").glob("**/*.json"))):
            # json_path
            # try:
            img, gt_bboxes, gt_texts = parse_json_file(json_path)
            # except Exception:
            #     continue

            for text, (xmin, ymin, xmax, ymax) in zip(gt_texts, gt_bboxes):
                patch = get_image_cropped_by_rectangle(
                    img=img, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax
                )
                fname = Path(f"{json_path.stem}_{xmin}-{ymin}-{xmax}-{ymax}.png")
                save_image(img=patch, path=save_dir/"images"/fname)

                ls_row.append(
                    (fname, text)
                )
            df_labels = pd.DataFrame(ls_row, columns=["filename", "words"])
            df_labels.to_csv(save_dir/"labels.csv", index=False)


if __name__ == "__main__":
    args = get_arguments()

    unzip_dataset(args.dataset)

    create_dataset_for_training(
        input_dir=args.dataset.parent/"unzipped",
        output_dir=args.dataset.parent/"dataset_for_training",
    )

