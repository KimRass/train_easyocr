import json
import argparse
import numpy as np
import pandas as pd
from zipfile import ZipFile
from pathlib import Path
from tqdm.auto import tqdm
import re
import random
import yaml
import shutil
from csv import writer

from utils import (
    AttrDict
)
from process_image import (
    load_image_as_array,
    save_image,
    get_image_cropped_by_rectangle
)


def get_arguments():
    parser = argparse.ArgumentParser(description="prepare_dataset")

    parser.add_argument("--unzip", action="store_true", default=False)
    parser.add_argument("--training", action="store_true", default=False)
    parser.add_argument("--validation", action="store_true", default=False)
    parser.add_argument("--evaluation", action="store_true", default=False)
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

        # if not evaluation:
        #     # ls_member = ls_member[:: 10]
        #     ls_member = ls_member[:: 10000]
        # else:
        #     # ls_member = ls_member[1:: 20]
        #     ls_member = ls_member[1:: 20000]

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
    print("Unzipping the original dataset...")

    dataset_dir = Path(dataset_dir)

    for zip_file in tqdm(list(dataset_dir.glob("**/*.zip"))):
        unzip_to = Path(
            str(zip_file).replace(dataset_dir.name, "unzipped").lower()
        ).parent
        unzip_to.mkdir(parents=True, exist_ok=True)

        _unzip(zip_file=zip_file, unzip_to=unzip_to)

    print("Completed unzipping the original dataset.")


def save_image_patches(output_dir, split, select_data, json_file_list):
    print(f"Creating image patches for {split}...")

    save_dir = Path(output_dir)/split/select_data

    # ls_row = list()
    labels_csv_path = Path(save_dir/"labels.csv")
    labels_csv_path.mkdir(parents=True, exist_ok=True)
    print(labels_csv_path)
    
    df_labels = pd.DataFrame(columns=["filename", "words"])
    df_labels.to_csv(labels_csv_path, index=False)

    for json_path in tqdm(json_file_list):
        try:
            img, gt_bboxes, gt_texts = parse_json_file(json_path)
        except Exception:
            # print(f"    No image file paring with '{json_path}'")
            continue

        for text, (xmin, ymin, xmax, ymax) in zip(gt_texts, gt_bboxes):
            xmin = max(0, xmin)
            ymin = max(0, ymin)

            try:
                patch = get_image_cropped_by_rectangle(
                    img=img, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax
                )
                fname = Path(f"{json_path.stem}_{xmin}-{ymin}-{xmax}-{ymax}.png")
                save_path = save_dir/"images"/fname
                if not save_path.exists():
                    save_image(img=patch, path=save_path)

                    # ls_row.append(
                    #     (fname, text)
                    # )
                    
                    with open(labels_csv_path, mode="a") as f:
                        writer(f).writerow((fname, text))
                        f.close()
            except Exception:
                print(f"    Failed to save '{fname}'.")

    print(f"Completed creating image patches for {split}.")


def prepare_evaluation_set(eval_set) -> None:
    print(f"Preparing evaluation set...")

    for json_path in tqdm(eval_set):
        new_json_path = Path(str(json_path).replace("unzipped/validation/", "evaluation_set/"))
        new_json_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(src=json_path, dst=new_json_path)

        img_path = str(json_path).replace("labels/", "images/").replace(".json", ".jpg")
        new_img_path = Path(str(new_json_path).replace("labels/", "images/").replace(".json", ".jpg"))
        new_img_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(src=img_path, dst=new_img_path)
    
    print(f"Completed preparing evaluation set.")


def count_images(dataset):
    # dataset = "/Users/jongbeom.kim/Documents/공공행정문서 OCR"

    tr = Path(dataset).parent/"training_and_validation_set/training"
    val = Path(dataset).parent/"training_and_validation_set/validation"

    n_img_tr = len(list(tr.glob("select_data/images/*.png")))
    n_img_val = len(list(val.glob("select_data/images/*.png")))

    df_labels_tr = pd.read_csv(tr/"select_data/labels.csv")
    df_labels_val = pd.read_csv(val/"select_data/labels.csv")
    
    # if n_img_tr == len(df_labels_tr):
    #     print(f"Number of training images: {n_img_tr:,}")
    # if n_img_val == len(df_labels_val):
    #     print(f"Number of validation images: {n_img_val:,}")
    print(n_img_tr)
    print(len(df_labels_tr))
    print(n_img_val)
    print(len(df_labels_val))


if __name__ == "__main__":
    args = get_arguments()

    with open("./config_files/configuration.yaml", mode="r", encoding="utf8") as f:
        config = AttrDict(
            yaml.safe_load(f)
        )

    random.seed(config.seed)

    if args.unzip:
        unzip_dataset(args.dataset)

    unzipped_dir = Path(args.dataset).parent/"unzipped"

    train_set = random.choices(
        list((unzipped_dir/"training"/"labels").glob("**/*.json")), k=10000
    )
    val_set = random.choices(
        list((unzipped_dir/"validation"/"labels").glob("**/*.json")), k=2000
    )
    eval_set = random.choices(
        list(
            set((unzipped_dir/"validation"/"labels").glob("**/*.json")) - set(val_set)
        ), k=500
    )

    if args.training:
        save_image_patches(
            output_dir=Path(args.dataset).parent/"training_and_validation_set",
            split="training",
            select_data=config.select_data,
            json_file_list=train_set,
        )
    if args.validation:
        save_image_patches(
            output_dir=Path(args.dataset).parent/"training_and_validation_set",
            split="validation",
            select_data=config.select_data,
            json_file_list=val_set,
        )
    if args.evaluation:
        prepare_evaluation_set(eval_set)

    count_images(args.dataset)
