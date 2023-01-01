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

from train_easyocr.utils import (
    AttrDict
)
from process_image import (
    load_image_as_array,
    save_image,
    get_image_cropped_by_rectangle
)


def get_arguments():
    parser = argparse.ArgumentParser(description="prepare_dataset")

    parser.add_argument("--dataset")
    parser.add_argument("--unzip", action="store_true", default=False)
    parser.add_argument("--training", action="store_true", default=False)
    parser.add_argument("--validation", action="store_true", default=False)
    parser.add_argument("--evaluation", action="store_true", default=False)

    args = parser.parse_args()
    return args


def parse_json_file(json_path):
    img_path = str(json_path).replace("/labels/", "/images/").replace(".json", ".jpg")
    img = load_image_as_array(img_path)

    with open(json_path, mode="r") as f:
        label = json.load(f)

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
    print(f"Generating image patches for {split}...")

    save_dir = Path(output_dir)/split/select_data
    save_dir.mkdir(parents=True, exist_ok=True)

    labels_csv_path = save_dir/"labels.csv"
    
    df_labels = pd.DataFrame(columns=["filename", "words"])
    df_labels.to_csv(labels_csv_path, index=False)

    for json_path in tqdm(json_file_list):
        try:
            img, gt_bboxes, gt_texts = parse_json_file(json_path)
        except Exception:
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

                    with open(labels_csv_path, mode="a") as f:
                        writer(f).writerow((fname, text))
                        f.close()
            except Exception:
                print(f"    Failed to save '{fname}'.")

    print(f"Completed generating image patches for {split}.", end=" ")

    df_labels = pd.read_csv(labels_csv_path)
    print(f"({len(df_labels)})")


def prepare_evaluation_set(eval_set) -> None:
    print(f"Preparing evaluation set...")

    for json_path in tqdm(eval_set):
        new_json_path = Path(str(json_path).replace("unzipped/validation/", "evaluation_set/"))
        new_json_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src=json_path, dst=new_json_path)

        img_path = str(json_path).replace("labels/", "images/").replace(".json", ".jpg")
        new_img_path = Path(str(new_json_path).replace("labels/", "images/").replace(".json", ".jpg"))
        new_img_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src=img_path, dst=new_img_path)
    
    print(f"Completed preparing evaluation set.")


if __name__ == "__main__":
    args = get_arguments()

    with open(Path(__file__).parent/"config_files/configuration.yaml", mode="r", encoding="utf8") as f:
        config = AttrDict(
            yaml.safe_load(f)
        )

    random.seed(config.seed)

    if args.unzip:
        unzip_dataset(args.dataset)

    unzipped_dir = Path(args.dataset).parent/"unzipped"

    train_set = random.choices(
        list((unzipped_dir/"training"/"labels").glob("**/*.json")), k=config.n_train_images
    )
    val_set = random.choices(
        list((unzipped_dir/"validation"/"labels").glob("**/*.json")), k=config.n_val_images
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
