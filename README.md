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