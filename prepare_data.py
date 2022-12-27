import json
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

from process_image import (
    load_image_as_array,
    save_image,
    get_image_cropped_by_rectangle
    # show_image,
)


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


def create_dataset(dir, dir_save) -> None:
    dir = Path(dir)
    dir_save = Path(dir_save)
    
    for path_json in tqdm(list(dir.glob("**/*.json"))):
        path_json
        try:
            img, gt = get_image_and_label(path_json)
        except Exception:
            continue

        # ls_line = list()
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

            patch = get_image_cropped_by_rectangle(
                img=img, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax
            )
            fname = Path(f"{path_json.stem}_{xmin}-{ymin}-{xmax}-{ymax}.png")
            save_image(img=patch, path=dir_save/split1/split2/"images"/fname)

            with open(dir_save/split1/split2/"gt.txt", mode="a") as f:
                f.write(f"images/{fname}\t{text}\n")
                f.close()

        #     ls_line.append(f"{dir_save.stem/fname}\t{text}\n")
        # with open(dir_save/split1/split2/"gt.txt", mode="a") as f:
        #     f.writelines(ls_line)
        #     f.close()


create_dataset(
    dir="/Users/jongbeom.kim/Documents/New_sample",
    dir_save=f"/Users/jongbeom.kim/Documents/output/"
)
