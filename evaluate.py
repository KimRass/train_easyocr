# Reference: https://gist.github.com/tarlen5/008809c3decf19313de216b9208f3734

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import jiwer
import easyocr
import argparse
from collections import defaultdict
import os

from process_image import (
    draw_easyocr_result,
    save_image
)
from craft_utilities import (
    load_craft_checkpoint,
    get_text_score_map_and_link_score_map
)
from detect_texts import (
    get_horizontal_list
)
from prepare_dataset import (
    parse_json_file
)

save_dir = Path(os.environ["PYTHONPATH"])/"evaluation_result"
result_csv_path = save_dir/"evaluation_result.csv"


def get_arguments():
    parser = argparse.ArgumentParser(description="evaluate")

    parser.add_argument("--eval_set")
    parser.add_argument("--baseline", action="store_true", default=False)
    parser.add_argument("--finetuned", action="store_true", default=False)
    parser.add_argument("--cuda", default=False, action="store_true")

    args = parser.parse_args()
    return args


def get_iou(bbox1, bbox2):
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    xmin_inter = max(xmin1, xmin2)
    xmax_inter = min(xmax1, xmax2)
    ymin_inter = max(ymin1, ymin2)
    ymax_inter = min(ymax1, ymax2)
    
    area_inter = max(0, xmax_inter - xmin_inter) * max(0, ymax_inter - ymin_inter)
    area_union = area1 + area2 - area_inter

    iou = area_inter / area_union
    return iou


def get_end_to_end_f1_score(gt_bboxes, pred_bboxes, iou_thr=0.5):
    ls_idx_gt = list()
    ls_idx_pred = list()
    ls_iou = list()
    for idx_pred, pred_bbox in enumerate(pred_bboxes.values):
        pred_bbox = pred_bbox[: 4]
        for idx_gt, gt_bbox in enumerate(gt_bboxes.values):
            gt_bbox = gt_bbox[: 4]

            iou = get_iou(pred_bbox, gt_bbox)

            if iou >= iou_thr:
                ls_idx_gt.append(idx_gt)
                ls_idx_pred.append(idx_pred)
                ls_iou.append(iou)

    argsort_iou_desc = np.argsort(ls_iou)[:: -1]
    # No matches
    if len(argsort_iou_desc) == 0:
        return 0
    else:
        ls_idx_match_gt = list()
        ls_idx_match_pred = list()
        ls_score = list()
        for idx in argsort_iou_desc:
            idx_gt = ls_idx_gt[idx]
            idx_pred = ls_idx_pred[idx]

            gt_label = gt_bboxes.iloc[idx_gt, 4]
            pred_label = pred_bboxes.iloc[idx_pred, 4]
            
            cer = jiwer.cer(truth=gt_label, hypothesis=pred_label)
            score = 1 - cer

            if (
                idx_gt not in ls_idx_match_gt and
                idx_pred not in ls_idx_match_pred
            ):
                ls_idx_match_gt.append(idx_gt)
                ls_idx_match_pred.append(idx_pred)

                ls_score.append(score)
        
        sum_score = sum(ls_score)
        # tp = sum_score
        # fp = len(pred_bboxes) - sum_score
        # fn = len(gt_bboxes) - sum_score
        precision = sum_score / len(pred_bboxes)
        recall = sum_score / len(gt_bboxes)

        f1_score = 2 * (precision * recall) / (precision + recall)
        f1_score = round(f1_score, 4)
        return f1_score


def spot_texts_using_baseline_model(img, reader):
    result = reader.readtext(img)

    ls_bbox = list()
    for ((x1, y1), (x2, y2), (x3, y3), (x4, y4)), text, _ in result:
        if x1.dtype != "int64":
            continue

        xmin, _, _, xmax = sorted([x1, x2, x3, x4])
        ymin, _, _, ymax = sorted([y1, y2, y3, y4])
        ls_bbox.append((xmin, ymin, xmax, ymax, text))
    pred_bboxes = pd.DataFrame(ls_bbox, columns=["xmin", "ymin", "xmax", "ymax", "text"])
    return pred_bboxes


def spot_texts_using_finetuned_model(img, craft, reader, cuda=False):
    text_score_map, link_score_map = get_text_score_map_and_link_score_map(img=img, craft=craft, cuda=cuda)

    horizontal_list = get_horizontal_list(img, text_score_map, link_score_map, thr=300)
    result = reader.recognize(img_cv_grey=img, horizontal_list=horizontal_list, free_list=list())

    ls_bbox = list()
    for ((x1, y1), (x2, y2), (x3, y3), (x4, y4)), text, _ in result:
        xmin, _, _, xmax = sorted([x1, x2, x3, x4])
        ymin, _, _, ymax = sorted([y1, y2, y3, y4])
        ls_bbox.append((xmin, ymin, xmax, ymax, text))
    pred_bboxes = pd.DataFrame(ls_bbox, columns=["xmin", "ymin", "xmax", "ymax", "text"])
    return pred_bboxes


def evaluate_using_baseline_model(dataset_dir, reader, eval_result):
    print(f"Evaluating '{dataset_dir}' using baseline model...")

    dataset_dir = Path(dataset_dir)

    for json_path in tqdm(list(dataset_dir.glob("**/*.json"))):
        json_path = "/Users/jongbeom.kim/Documents/evaluation_set/labels/주민복지/5350129/1999/5350129-1999-0001-0282.json"
        fname = "/".join(str(json_path).rsplit("/", 4)[1:])

        img, gt_bboxes = parse_json_file(json_path, load_image=True)

        pred_bboxes = spot_texts_using_baseline_model(img=img, reader=reader)
        f1 = get_end_to_end_f1_score(gt_bboxes=gt_bboxes, pred_bboxes=pred_bboxes, iou_thr=0.5)
        
        eval_result[fname]["baseline"] = f1

        drawn = draw_easyocr_result(img=img, bboxes=pred_bboxes)
        save_path = save_dir/"baseline"/str(Path(fname).name).replace(".json", ".jpg")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(img=drawn, path=save_path)
    return eval_result


def evaluate_using_finetuned_model(dataset_dir, reader, eval_result, craft, cuda):
    print(f"Evaluating '{dataset_dir}' using fine-tuned model...")

    dataset_dir = Path(dataset_dir)

    for json_path in tqdm(list(dataset_dir.glob("**/*.json"))):
        fname = "/".join(str(json_path).rsplit("/", 4)[1:])

        img, gt_bboxes = parse_json_file(json_path, load_image=True)

        pred_bboxes = spot_texts_using_finetuned_model(
            img=img, craft=craft, reader=reader, cuda=cuda
        )
        f1 = get_end_to_end_f1_score(gt_bboxes=gt_bboxes, pred_bboxes=pred_bboxes, iou_thr=0.5)
        
        eval_result[fname]["finetuned"] = f1

        drawn = draw_easyocr_result(img=img, bboxes=pred_bboxes)
        save_path = save_dir/"baseline"/str(Path(fname).name).replace(".json", ".jpg")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(img=drawn, path=save_path)
    return eval_result


def save_evaluation_result_as_csv(eval_result) -> None:
    df_result = pd.DataFrame.from_dict(eval_result, orient="index")
    df_result.reset_index(inplace=True)
    df_result.rename({"index": "file"}, axis=1, inplace=True)

    df_result.to_csv(result_csv_path, index=False)


def print_summary():
    if result_csv_path.exists():
        df_result = pd.read_csv(result_csv_path)

    cols = df_result.columns.tolist()
    if "baseline" in cols:
        f1_bl = df_result["baseline"].mean()
        print(f"Mean f1 score for baseline model:   {f1_bl:.3f}")
    if "finetuned" in cols:
        f1_ft = df_result["finetuned"].mean()
        print(f"Mean f1 score for fine-tuned model: {f1_ft:.3f}")
    if "baseline" in cols and "finetuned" in cols:
        print(f"F1 score increased {(f1_ft - f1_bl) / f1_bl:.1%}")


def main():
    args = get_arguments()

    craft = load_craft_checkpoint(cuda=args.cuda)

    # Baseline
    reader_bl = easyocr.Reader(lang_list=["ko"], gpu=args.cuda)
    # Fine-tuned
    reader_ft = easyocr.Reader(
        lang_list=["ko"],
        gpu=args.cuda,
        model_storage_directory="/home/ubuntu/.EasyOCR/model",
        user_network_directory="/home/ubuntu/.EasyOCR/user_network",
        recog_network="finetuned"
    )

    if args.baseline:
        eval_result = evaluate_using_baseline_model(
            dataset_dir=args.eval_set, reader=reader_bl, eval_result=defaultdict(dict)
        )
        save_evaluation_result_as_csv(eval_result)
        if args.finetuned:
            eval_result = evaluate_using_finetuned_model(
                dataset_dir=args.eval_set, reader=reader_ft, eval_result=eval_result, craft=craft, cuda=args.cuda
            )
            save_evaluation_result_as_csv(eval_result)
    elif args.finetuned:
        eval_result = evaluate_using_finetuned_model(
            dataset_dir=args.eval_set, reader=reader_ft, eval_result=defaultdict(dict), craft=craft, cuda=args.cuda
        )
        save_evaluation_result_as_csv(eval_result)

    print_summary()


if __name__ == "__main__":
    main()
