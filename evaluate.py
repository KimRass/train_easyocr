# Reference: https://gist.github.com/tarlen5/008809c3decf19313de216b9208f3734
import numpy as np
import jiwer


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


def get_giou(bbox1, bbox2):
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    xmin_enclose, xmin_inter = sorted([xmin1, xmin2])
    ymin_enclose, ymin_inter = sorted([ymin1, ymin2])
    xmax_inter, xmax_enclose = sorted([xmax1, xmax2])
    ymax_inter, ymax_enclose = sorted([ymax1, ymax2])
    
    area_inter = max(0, xmax_inter - xmin_inter) * max(0, ymax_inter - ymin_inter)
    area_union = area1 + area2 - area_inter

    iou = area_inter / area_union
    
    area_enclose = (xmax_enclose - xmin_enclose) * (ymax_enclose - ymin_enclose)
    giou = iou - (area_enclose - area_union) / area_enclose
    return giou


# def get_text_detection_f1_score(gt_bboxes, pred_bboxes, iou_thr=0.5):
#     gt_bboxes = np.array(gt_bboxes[["xmin", "ymin", "xmax", "ymax"]])
#     pred_bboxes = np.array(pred_bboxes[["xmin", "ymin", "xmax", "ymax"]])

#     ls_idx_thr_gt = list()
#     ls_idx_thr_pred = list()
#     ls_iou = list()
#     for ipb, pred_bbox in enumerate(pred_bboxes):
#         for igb, gt_bbox in enumerate(gt_bboxes):
#             iou = get_iou(pred_bbox, gt_bbox)

#             if iou > iou_thr:
#                 ls_idx_thr_gt.append(igb)
#                 ls_idx_thr_pred.append(ipb)
#                 ls_iou.append(iou)

#     args_desc = np.argsort(ls_iou)[::-1]
#     # No matches
#     if len(args_desc) == 0:
#         tp = 0
#         fp = len(pred_bboxes)
#         fn = len(gt_bboxes)
#     else:
#         ls_idx_match_gt = list()
#         ls_idx_match_pred = list()
#         for idx in args_desc:
#             idx_gt = ls_idx_thr_gt[idx]
#             idx_pred = ls_idx_thr_pred[idx]

#             if (
#                 idx_gt not in ls_idx_match_gt and
#                 idx_pred not in ls_idx_match_pred
#             ):
#                 ls_idx_match_gt.append(idx_gt)
#                 ls_idx_match_pred.append(idx_pred)

#         tp = len(ls_idx_match_gt)
#         fp = len(pred_bboxes) - len(ls_idx_match_pred)
#         fn = len(gt_bboxes) - len(ls_idx_match_gt)

#     precision = tp / (tp + fp)
#     recall = tp / (tp + fn)
#     f1_score = 2 * (precision * recall) / (precision + recall)
#     f1_score = round(f1_score, 3)
#     return f1_score


def get_end_to_end_f1_score(gt_bboxes, gt_texts, pred_texts, pred_bboxes, iou_thr=0.5, rec=False):
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
        tp = 0
        fp = len(pred_bboxes)
        fn = len(gt_bboxes)
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

            if rec:
                if (
                    idx_gt not in ls_idx_match_gt and
                    idx_pred not in ls_idx_match_pred
                ):
                    ls_idx_match_gt.append(score)
                    ls_idx_match_pred.append(score)
            else:
                if (
                    idx_gt not in ls_idx_match_gt and
                    idx_pred not in ls_idx_match_pred and
                    score == 1
                    # cer == 0
                ):
                    ls_idx_match_gt.append(idx_gt)
                    ls_idx_match_pred.append(idx_pred)

        if rec:
            tp = sum(ls_idx_match_gt)
            fp = len(pred_bboxes) - sum(ls_idx_match_pred)
            fn = len(gt_bboxes) - sum(ls_idx_match_gt)
        else:
            tp = len(ls_idx_match_gt)
            fp = len(pred_bboxes) - len(ls_idx_match_pred)
            fn = len(gt_bboxes) - len(ls_idx_match_gt)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    f1_score = round(f1_score, 3)
    return f1_score
