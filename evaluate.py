import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import jiwer
import easyocr
import argparse
from collections import defaultdict

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


def get_arguments():
    parser = argparse.ArgumentParser(description="evaluate")

    parser.add_argument("--eval_set")
    # parser.add_argument("--baseline", action="store_true", default=False)
    # parser.add_argument("--finetuned", action="store_true", default=False)
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


def get_end_to_end_f1_score(gt_bboxes, gt_texts, pred_texts, pred_bboxes, iou_thr=0.5):
    # gt_bboxes = np.array(gt_bboxes[["xmin", "ymin", "xmax", "ymax"]])
    # pred_bboxes = np.array(pred_bboxes[["xmin", "ymin", "xmax", "ymax"]])
    
    # gt_texts = np.array(gt_bboxes["text"])
    # pred_texts = np.array(pred_bboxes["text"])
    # iou_thr=0.5
    ls_idx_thr_gt = list()
    ls_idx_thr_pred = list()
    ls_iou = list()
    for ipb, pred_bbox in enumerate(pred_bboxes):
        for igb, gt_bbox in enumerate(gt_bboxes):
            iou = get_iou(pred_bbox, gt_bbox)

            if iou > iou_thr:
                ls_idx_thr_gt.append(igb)
                ls_idx_thr_pred.append(ipb)
                ls_iou.append(iou)

    args_desc = np.argsort(ls_iou)[::-1]
    # No matches
    if len(args_desc) == 0:
        return 0
    else:
        ls_idx_match_gt = list()
        ls_idx_match_pred = list()
        for idx in args_desc:
            idx_gt = ls_idx_thr_gt[idx]
            idx_pred = ls_idx_thr_pred[idx]
            
            gt_label = gt_texts[idx_gt]
            pred_label = pred_texts[idx_pred]
            
            cer = jiwer.cer(gt_label, pred_label)
            score = 1 - cer

            # if rec:
            if (
                idx_gt not in ls_idx_match_gt and
                idx_pred not in ls_idx_match_pred
            ):
                ls_idx_match_gt.append(score)
                ls_idx_match_pred.append(score)
            # else:
            #     if (
            #         idx_gt not in ls_idx_match_gt and
            #         idx_pred not in ls_idx_match_pred and
            #         score == 1
            #     ):
            #         ls_idx_match_gt.append(idx_gt)
            #         ls_idx_match_pred.append(idx_pred)

        # if rec:
        tp = sum(ls_idx_match_gt)
        fp = len(pred_bboxes) - sum(ls_idx_match_pred)
        fn = len(gt_bboxes) - sum(ls_idx_match_gt)
        # else:
        #     tp = len(ls_idx_match_gt)
        #     fp = len(pred_bboxes) - len(ls_idx_match_pred)
        #     fn = len(gt_bboxes) - len(ls_idx_match_gt)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        f1_score = 2 * (precision * recall) / (precision + recall)
        f1_score = round(f1_score, 3)
        return f1_score


def spot_texts_using_baseline_model(img, reader):
    result = reader.readtext(img)

    ls_bbox = list()
    ls_text = list()
    for ((x1, y1), (x2, y2), (x3, y3), (x4, y4)), text, _ in result:
        if x1.dtype != "int64":
            continue

        xmin, _, _, xmax = sorted([x1, x2, x3, x4])
        ymin, _, _, ymax = sorted([y1, y2, y3, y4])
        ls_bbox.append((xmin, ymin, xmax, ymax))
        ls_text.append(text)
    pred_bboxes = np.array(ls_bbox)
    pred_texts = np.array(ls_text)
    return pred_bboxes, pred_texts


def spot_texts_using_finetuned_model(img, craft, reader, cuda=False):
    text_score_map, link_score_map = get_text_score_map_and_link_score_map(img=img, craft=craft, cuda=cuda)

    horizontal_list = get_horizontal_list(img, text_score_map, link_score_map, thr=300)
    result = reader.recognize(img_cv_grey=img, horizontal_list=horizontal_list, free_list=list())

    ls_bbox = list()
    ls_text = list()
    for ((x1, y1), (x2, y2), (x3, y3), (x4, y4)), text, _ in result:
        xmin, _, _, xmax = sorted([x1, x2, x3, x4])
        ymin, _, _, ymax = sorted([y1, y2, y3, y4])
        ls_bbox.append((xmin, ymin, xmax, ymax))
        ls_text.append(text)
    pred_bboxes = np.array(ls_bbox)
    pred_texts = np.array(ls_text)
    return pred_bboxes, pred_texts


def evaluate_using_baseline_model(dataset_dir, reader, eval_result):
    print(f"Evaluating '{dataset_dir}' using baseline model...")

    dataset_dir = Path(dataset_dir)

    for json_path in tqdm(list(dataset_dir.glob("**/*.json"))):
        fname = "/".join(str(json_path).rsplit("/", 4)[1:])

        try:
            img, gt_bboxes, gt_texts = parse_json_file(json_path)

            pred_bboxes, pred_texts = spot_texts_using_baseline_model(img=img, reader=reader)
            f1 = get_end_to_end_f1_score(gt_bboxes, gt_texts, pred_texts, pred_bboxes, iou_thr=0.5)
            
            eval_result[fname]["baseline"] = f1
        except Exception:
            print(f"    No image file paring with '{json_path}'")
    return eval_result


def evaluate_using_finetuned_model(dataset_dir, reader, eval_result, craft, cuda):
    print(f"Evaluating '{dataset_dir}' using fine-tuned model...")

    dataset_dir = Path(dataset_dir)

    for json_path in tqdm(list(dataset_dir.glob("**/*.json"))):
        fname = "/".join(str(json_path).rsplit("/", 4)[1:])

        try:
            img, gt_bboxes, gt_texts = parse_json_file(json_path)

            pred_bboxes, pred_texts = spot_texts_using_finetuned_model(
                img=img, craft=craft, reader=reader, cuda=cuda
            )
            f1 = get_end_to_end_f1_score(gt_bboxes, gt_texts, pred_texts, pred_bboxes, iou_thr=0.5)
            
            eval_result[fname]["finetuned"] = f1
        except Exception:
            print(f"    No image file paring with '{json_path}'")
    return eval_result


def save_evaluation_result_as_csv(eval_result) -> None:
    df_result = pd.DataFrame.from_dict(eval_result, orient="index")
    df_result.reset_index(inplace=True)
    df_result.rename({"index": "file"}, axis=1, inplace=True)

    df_result.to_csv("evaluation_result.csv", index=False)


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

    # if args.baseline and args.finetuned:
    eval_result = evaluate_using_baseline_model(
        dataset_dir=args.eval_set, reader=reader_bl, eval_result=defaultdict(dict)
    )
    save_evaluation_result_as_csv(eval_result)

    eval_result = evaluate_using_finetuned_model(
        dataset_dir=args.eval_set, reader=reader_ft, eval_result=eval_result, craft=craft, cuda=args.cuda
    )
    save_evaluation_result_as_csv(eval_result)

    # elif args.baseline and not args.finetuned:
    #     eval_result = evaluate_using_baseline_model(
    #         dataset_dir=args.eval_set, reader=reader_bl, eval_result=defaultdict(dict)
    #     )
    #     save_evaluation_result_as_csv(eval_result)

    # elif not args.baseline and args.finetuned:
    #     eval_result = evaluate_using_finetuned_model(
    #         dataset_dir=args.eval_set, reader=reader_ft, eval_result=defaultdict(dict), craft=craft, cuda=args.cuda
    #     )
    #     save_evaluation_result_as_csv(eval_result)


if __name__ == "__main__":
    main()

    # f1_bl = df_result["baseline"].mean()
    # f1_ft = df_result["finetuned"].mean()

    # print(f"Mean f1 score for baseline model:   {f1_bl:.3f}")
    # print(f"Mean f1 score for fine-tuned model: {f1_ft:.3f}")
    # print(f"Increased {(f1_ft - f1_bl) / f1_bl:.1%}")