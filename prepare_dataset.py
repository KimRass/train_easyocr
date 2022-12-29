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


def create_image_patches(unzipped_dir, output_dir, split2="select_data") -> None:
    print("Creating image patches...")

    # unzipped_dir = "/Users/jongbeom.kim/Documents/unzipped"
    # output_dir = "/Users/jongbeom.kim/Documents/dataset_for_training"
    
    unzipped_dir = Path(unzipped_dir)
    output_dir = Path(output_dir)
    
    for split1, n in zip(["training", "validation"], [10000, 2000]):
        save_dir = output_dir/split1/split2/"images"
        save_dir.mkdir(parents=True, exist_ok=True)

        ls_json = list((unzipped_dir/split1/"labels").glob("**/*.json"))[: n]
        
        ls_row = list()
        for json_path in tqdm(ls_json):
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
                    # save_path = save_dir/"images"/fname
                    save_path = save_dir/fname
                    if not save_path.exists():
                        save_image(img=patch, path=save_path)

                    ls_row.append(
                        (fname, text)
                    )
                except Exception:
                    print(f"    Failed to save '{fname}'.")

        df_labels = pd.DataFrame(ls_row, columns=["filename", "words"])
        df_labels.to_csv(save_dir.parent/"labels.csv", index=False)

    print("Completed creating image patches.")


def prepare_dataset_for_evaluation(dataset_dir) -> None:
    dataset_dir = Path(dataset_dir)

    for zip_file in (dataset_dir/"validation").glob("**/*.zip"):
        unzip_to = Path(
            str(zip_file).replace(dataset_dir.name, "dataset_for_evaluation").replace("/validation/", "/").lower()
        ).parent
        unzip_to.mkdir(parents=True, exist_ok=True)

        _unzip(zip_file=zip_file, unzip_to=unzip_to)


def check_number_of_images(dataset):
    # dataset = "/Users/jongbeom.kim/Documents/공공행정문서 OCR"

    tr = Path(dataset).parent/"dataset_for_training/training"
    val = Path(dataset).parent/"dataset_for_training/validation"

    n_img_tr = len(list(tr.glob("select_data/images/*.png")))
    n_img_val = len(list(val.glob("select_data/images/*.png")))

    df_labels_tr = pd.read_csv(tr/"select_data/labels.csv")
    df_labels_val = pd.read_csv(val/"select_data/labels.csv")
    
    if n_img_tr == len(df_labels_tr):
        print(f"Number of training images: {n_img_tr:,}")
    if n_img_val == len(df_labels_val):
        print(f"Number of validation images: {n_img_val:,}")


if __name__ == "__main__":
    args = get_arguments()

    unzip_dataset(args.dataset)

    create_image_patches(
        unzipped_dir=Path(args.dataset).parent/"unzipped",
        output_dir=Path(args.dataset).parent/"dataset_for_training",
    )

    prepare_dataset_for_evaluation(args.dataset)

    check_number_of_images(args.dataset)
