import json
import argparse
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


def get_image_and_label(path_json):
    path_json = Path(path_json)

    with open(path_json, mode="r") as f:
        label = json.load(f)

    ls_row = list()
    for sample in label["annotations"]:
        text = sample["annotation.text"]
        xmin, ymin, width, height = sample["annotation.bbox"]
        
        ls_row.append([text, xmin, ymin, xmin + width, ymin + height])
    df_label = pd.DataFrame(ls_row, columns=["text", "xmin", "ymin", "xmax", "ymax"])

    path_img = f"{path_json.parents[4]}/원천데이터/인.허가/{path_json.parent.parent.stem}/{path_json.parent.stem}/{label['images'][0]['image.file.name']}"

    img = load_image_as_array(path_img)
    return img, df_label


def create_dataset(input_dir, output_dir) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    for path_json in tqdm(list(input_dir.glob("**/*.json"))):
        try:
            img, gt = get_image_and_label(path_json)
        except Exception:
            continue

        for idx, (text, xmin, ymin, xmax, ymax) in enumerate(gt.values):
            remainder = idx % 5
            if remainder in [0, 1, 2, 3]:
                split1 = "training"
            else:
                split1 = "validation"
            
            remainder = idx % 2
            if remainder == 0:
                split2 = "MJ"
            else:
                split2 = "ST"
            # split2 = "dataset"

            patch = get_image_cropped_by_rectangle(
                img=img, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax
            )
            fname = Path(f"{path_json.stem}_{xmin}-{ymin}-{xmax}-{ymax}.png")
            save_image(img=patch, path=output_dir/split1/split2/"images"/fname)

            with open(output_dir/split1/split2/"gt.txt", mode="a") as f:
                # f.write(f"images/{fname}\t{text}\n")
                f.write(f"{fname}\t{text}\n")
                f.close()

    for path_txt in output_dir.glob("**/*.txt"):
        df = pd.DataFrame(
            [line.strip().split("\t") for line in open(path_txt, mode="r").readlines()],
            columns=["filename", "words"]
        )
        df.to_csv(path_txt.parent/"labels.csv", index=False)


if __name__ == "__main__":
    args = get_arguments()

    create_dataset(input_dir=args.input_dir, output_dir=args.output_dir)
