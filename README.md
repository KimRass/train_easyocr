# Dataset
- Training set: 102,477
    [라벨]train.zip
    [원천]train1.zip
    [원천]train2.zip
    [원천]train3.zip
- Validation set: 11,397
  - [라벨]validation.zip
  - [원천]validation.zip
- 비율 약 9:1
## 구조
## 압축 풀기 전
공공행정문서 OCR
  Training
    [라벨]train.zip
    [원천]train1.zip
    [원천]train2.zip
    [원천]train3.zip
  Validation
    [라벨]validation.zip
    [원천]validation.zip
## 압축 푼 후
unzipped
  training
    images
      인.허가
      회계.예산
      도시개발
      일반행정
      주민자치
      지역환경.산림
      상.하수도관리
      농림.축산지원
      산업진흥
      주민복지
      지역문화
      주민생활지원
    labels
  validation
    images
    labels

# Metric
# Edit distance
- Reference: https://en.wikipedia.org/wiki/Edit_distance
- Edit distance is a way of quantifying how dissimilar two strings (e.g., words) are to one another by counting the minimum number of operations required to transform one string into the other.
Types of edit distance
  - Levenshtein distance allows deletion, insertion and substitution.

# Error Rate (CER (Character Error Rate), WER (Word Error Rate))
## CER
- CER calculation is based on the concept of Levenshtein distance, where we count the minimum number of character-level operations required to transform the ground truth text (aka reference text) into the OCR output.
- CER = (S + D + I) / N
  - N: Number of characters in grund truth (= S + D + C)
```python
import import jiwer

cer = jiwer.cer(gt, pred)
wer = jiwer.wer(gt, pred)
```

# PaddleOCR
- Text detection: DBNet (AAAI'2020)
- Text recognition: ABINet (CVPR'2021)

# MMOCR
- DB_r18
- ABINet

# EasyOCR
- Text detection: CRAFT (Default), DBNet
- Text recognition: (None-VGG-BiLSTM-CTC)

```python
if self.detect_network == 'craft':
    from .detection import get_detector, get_textbox
elif self.detect_network in ['dbnet18']:
    from .detection_db import get_detector, get_textbox
else:
    raise RuntimeError("Unsupport detector network. Support networks are craft and dbnet18.")
```
- Text recognition:
`detect_network="craft", recog_network='standard'`

# Dataset
- 모든 이미지는 흰색과 검은색으로만 이루어져 있습니다.

# Baseline
- F1 score: 0.53

# Limitation of Evaluation Metric
- Reference: https://arxiv.org/pdf/2006.06244.pdf
- IoU + CRW
- One-to-many 또는 Many-to-one 상황에서 대응 x
## CLEval
- Not IoU-based evaluation metric.

# Improvements
## Hyperparameter Tunning
### Beam Search
- `decoder`: `"greedy"` -> `"beam"`?
## Image Processing
## Fine Tunning

# References
- Baseline: https://github.com/JaidedAI/EasyOCR
- CRAFT: https://github.com/clovaai/CRAFT-pytorch
- Intersection over Union: https://gaussian37.github.io/vision-detection-giou/
- Metric: https://gist.github.com/tarlen5/008809c3decf19313de216b9208f3734