# Library Comparison
## PaddleOCR
- Source: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- 다양한 Text detection, text recongnition 모델을 지원합니다. ([Algorithms](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/algorithm_overview_en.md))
- 자체 개발한 Text detection + Text recongnition 모델 'PP-OCRv3'를 제공합니다.
- 다수의 한자를 포함하여 3,687개의 문자를 지원하나 마침표와 쉼표 등이 없습니다. ([korean_dict.txt](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/utils/dict/korean_dict.txt))
## MMOCR
- Source: [MMOCR](https://github.com/open-mmlab/mmocr)
- 다양한 Text detection, text recongnition 모델을 지원합니다. ([Model Zoo](https://mmocr.readthedocs.io/en/latest/modelzoo.html))
## EasyOCR
- Source: [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- Text detection: [CRAFT](https://github.com/clovaai/CRAFT-pytorch) (Default), DBNet
- Text recognition
  - Transformation: None or TPS ([Thin Plate Spline](https://en.wikipedia.org/wiki/Thin_plate_spline))
  - Feature extraction: VGG, RCNN or ResNet
  - Sequence modeling: None or BiLSTM
  - Prediction: CTC or Attention
- Pre-trained model로서 사용할 [korean_g2](https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/korean_g2.zip)의 구조는 'None-VGG-BiLSTM-CTC'입니다.
- 한국어에 대해서는 1,008개의 문자를 지원하고 한자는 없습니다.

# Improvements
## Text Detection
- 'CRAFT'의 예측값을 바탕으로 Bounding boxes를 정하는 알고리즘을 개선했습니다. ('detect_texts.py': `get_horizontal_list()`)
  - Baseline text detection result
    - <img src="https://i.imgur.com/WJSRjjA.png" alt="baseline" width="500"/>
  - Improved text detection result
    - <img src="https://i.imgur.com/L2Ea0bL.png" alt="improved" width="500"/>
## Text Recognition
- AIHub의 [공공행정문서 OCR](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=88) 데이터셋을 사용하여 Fine-tunning을 진행했습니다.

# How to Fine-tune
## Step 1: Environment Setting
- run `source step1_set_environment.sh`
- set 'train_easyocr/config_files/config.yaml'
## Step 2: Dataset Preparation
- run `bash step2_run_prepare_dataset_py.sh`
- The example of 'step2_run_prepare_dataset_py.sh':
  ```sh
  python3 prepare_dataset.py\
    --dataset="/data/공공행정문서 OCR"\ # Path to the original dataset directory "공공행정문서 OCR"
    --unzip\ # Whether to unzip
    --training\ # Whether to generate training set
    --validation\ # Whether to generate validation set
    --evaluation # Whether to generate evaluation set
  ```
## Step 3: Training (Fine-tunning)
- run `bash step3_run_train_py.sh`
## Step 4: Fine-tuned Model Setting
- Reference: [Custom recognition models](https://github.com/JaidedAI/EasyOCR/blob/master/custom_model.md)
- run `bash step4_set_finetuned_model.sh`
- The example of 'step4_set_finetuned_model.sh':
  ```sh
  cp train_easyocr/saved_models/phase4/best_norm_ed.pth ~/.EasyOCR/model/finetuned.pth
  cp finetuned/finetuned.py ~/.EasyOCR/user_network/finetuned.py
  cp finetuned/finetuned.yaml ~/.EasyOCR/user_network/finetuned.yaml
  ```
- Then the structure of directory '~/.EasyOCR' would be like,
  ```
  ~/.EasyOCR
  ├── model
  │   └── finetuned.pth
  └── user_network
      ├── finetuned.py
      └── finetuned.yaml
  ```
## Step 5: Evaluation
- run `bash step5_run_evaluate_py.sh`
- The example of 'step5_run_evaluate_py.sh':
  ```sh
  python3 evaluate.py\
    --eval_set="/data/evaluation_set"\ # Path to the evaluation set
    --baseline\ # Whether to evaluate EasyOCR baseline model
    --finetuned\ # Whether to evaluate fine-tuned model
    --cuda # Whether to use GPU
  ```

# Step 1: Environment Setting
## Configurations ('train_easyocr/config_files/config.yaml')
  ```yaml
  ### Environment ###
  seed: # Seed
  experiment_name: # 'train_easyocr/saved_models'에 생성될 폴더 이름입니다.

  ### Dataset ###
  train_data: # Training set의 디렉토리
  val_data: # Validation set의 디렉토리
  select_data: # Subdirectory
  batch_ratio: 
    # Modulate the data ratio in the batch.
    # For example, when `select_data` is `MJ-ST` and `batch_ratio` is `0.5-0.5`,
    # the 50% of the batch is filled with 'MJ' and the other 50% of the batch is filled with 'ST'.
  total_data_usage_ratio: # How much ratio of the data to use
  train_images: # Number of training images
  val_images: # Number of validation images
  eval_images: # Number of evaluation images
  ### Data processing ###
    img_height: # Height of input image
  img_width: # Width of input image
  PAD: # If `True` pad to input images to keep aspect ratio
  contrast_adjust: # Adjust contrast
  character:
    # 예측에 사용할 문자들
    # Pre-trained model로서 'korean_g2'를 사용할 것이므로 사용할 문자들을 다음을 참고하여 동일하게 설정합니다. (https://github.com/JaidedAI/EasyOCR/blob/master/easyocr/config.py)
  sensitive: # Case sensitivity
  batch_max_length: # Maximum length of label
  data_filtering_off: 
    # If `False` filter images containing characters not in `character`
    # and whose label is longer than `batch_max_length`

  ### Training ###
  workers: # Same as `num_workers` from `torch.utils.data.DataLoader`
  batch_size: # Batch size
  n_iter: # Number of iterations
  val_period: # Period to run validation
  show_number: # How many validation result to show
  continue_from: 
    # Checkpoint from which to continue training
    # 첫 학습시에는 'korean_g2' (https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/korean_g2.zip)를 사용합니다.
  strict: # If `False` ignore non-matching keys when loading a model from checkpoint
  ### Optimizer ###
  adam: # If `True` use `torch.optim.Adam`, if `False` use `torch.optim.Adadelta`
  lr: 
  rho: 
  eps: 
  grad_clip: 

  ### Model ###
  Transformation: # `None` or `TPS`
  FeatureExtraction: # `VGG`, `RCNN` or `ResNet`
  SequenceModeling: # `None` or `BiLSTM`
  Prediction: # `CTC` or `Attn`
  ### VGG ###
  freeze_FeatureFxtraction: # If `True` do not update feature extraction parameters
  rgb: False # `True` for RGB input image
  input_channel: # `1` for grayscale input image, `3` for RGB
  output_channel: # Output dimension of featrue extraction result
  ### BiLSTM ###
  freeze_SequenceModeling: # If `True` do not update sequence modeling parameters
  hidden_size: # `hidden_size` of `torch.nn.LSTM`
  ### Prediction ###
  new_prediction: False # If `True` dimension of model prediction changes according to checkpoint
  ### CTC ###
  decode: # `greedy` or `beamsearch`
  ```

# Step 2: Dataset Preparation
## Original ('공공행정문서 OCR')
- 전체 데이터셋의 크기가 너무 커서 학습시키기에 어려움이 있으므로 아래 디렉토리 구조에 나타난 데이터만을 대상으로 했습니다.
  ```
  공공행정문서 OCR
  ├── Training
  │   ├── [라벨]train.zip
  │   ├── [원천]train1.zip
  │   ├── [원천]train3.zip
  │   └── [원천]train2.zip
  └── Validation
      ├── [라벨]validation.zip
      └── [원천]validation.zip
  ```
- Number of training images: 102,477
- Number of validation images: 11,397
## 'unzipped'
- 'step2_run_prepare_dataset_py.sh' 실행시 `--unzip`을 옵션으로 주면 아래와 같은 디렉토리 구조로 압축을 풉니다.
  ```
  unzipped
  ├── training
  │   ├── images
  │   │   └── ...
  │   └── labels
  │       └── ...
  └── validation
      ├── images
      │   └── ...
      └── labels
          └── ...
  ```
## Training Set & Validation Set
- 아래와 같은 디렉토리 구조로 이미지 패치가 생성됩니다.
- 'step2_run_prepare_dataset_py.sh' 실행시 `--training`를 옵션으로 주면 'training' 폴더가, `--validation`을 옵션으로 주면 'validation' 폴더가 생성됩니다.
  ```
  training_and_validation_set
  ├── training
  │   └── select_data
  │       ├── images
  │       │   └── ...
  │       └── labels.csv
  └── validation
      └── select_data
          ├── images
          │   └── ...
          └── labels.csv
  ```
- 'train_easyocr/config_files/config.yaml'에서 `train_images`와 `val_images`에 어떤 값을 주느냐에 따라 이미지 패치의 수가 달라지며 제가 사용한 이미지 패치의 수는 다음과 같습니다.
  - Number of training images: 40,000, image patches: 4,494,278
  - Number of validation images: 2,000, image patches: 223,115
- Structure of 'labels.csv':
  |filename|words|
  |-|-|
  |5350178-1999-0001-0344_829-262-1003-318.png|김해를|
  |5350178-1999-0001-0344_1022-262-1215-321.png|아름답게|
  |5350178-1999-0001-0344_1231-259-1384-324.png|시민을|
  |5350178-1999-0001-0344_1405-259-1620-324.png|행복하게|
  |...|...|
  - 일부 좌표가 음수인 경우 0으로 수정했습니다.
## Evaluation Set
- Validation set과 중복되지 않도록 'training_and_validation_set/validation'에서 무작위로 500개의 이미지를 뽑아 Evaluation set으로 선정했습니다.
```
evaluation_set
├── images
│   └── ...
└── labels
    └── ...
```

# Step 3: Training (Fine-tunning)
- Server specification: [AWS EC2 g5.xlarge](https://instances.vantage.sh/aws/ec2/g5.xlarge)
- Total number of trainable parameters: 4,015,729
- Total iterations: 약 600,000 (약 1 epoch)
- Total training time: 약 50시간

# Step 5: Evaluation
## Metric
- Text detection에 대해서는 'IoU >= 0.5'인 경우를 True positive로 하는 F1 score를, Text recognition에 대해서는 CER (Character Error Rate)를 사용했습니다.
- 그러나 위와 같이 Text detection과 Text recognition 각각에 대해서 평가하는 방법으로는 End-to-end evaluation을 실현할 수 없습니다. 따라서 'IoU >= 0.5'인 경우에 한해 CER을 측정하여 '1 - CER'로서 계산한 Score를 사용해 True positive, False positive, False negative를 측정했습니다. 이를 바탕으로 F1 score를 계산하여 최종 Metric으로 사용했습니다.
- 즉 완전히 Ground truth를 맞히기 위해서는 'IoU >= 0.5'이 되도록 Text detection을 수행하고 'CER = 0'이 되도록 Text recognition을 수행해야만 합니다. 'CER = 0'이라 하더라도 'IoU < 0.5'라면 예측이 전혀 맞지 않은 것이며 `IoU >= 0.5'라면 CER에 따라서 일종의 부분점수를 받게 됩니다.
## Result
- Baseline: 0.369
- Fine-tuned: 0.710
- 92.6% 성능이 향상됐습니다.

# Limitations
## Metric
- IoU를 기반으로 하는 Metric을 사용하므로 성능을 제대로 측정할 수 없는 상황이 발생할 수 있습니다. (Source: https://arxiv.org/pdf/2006.06244.pdf)
  - Split case
    - <img src="https://i.imgur.com/TjlMSm9.png" alt="split_case" width="300"/>
    - 바람직한 점수는 1입니다.
    - 예측 결과 'RIVER'의 부분점수는 0.56 ('1 - CER')이고 예측 결과 'SIDE'는 'IoU < 0.5'이므로 부분점수가 0입니다. 따라서 합은 0.56입니다.
  - Merged case:
    - <img src="https://i.imgur.com/eTM8mxc.png" alt="merged_case" width="300"/>
    - 바람직한 점수는 1입니다.
    - 예측 결과 'RIVERSIDE'는 IoU가 가장 정답 'RIVER'과 대응합니다. 따라서 정답 'SIDE'에 대한 부분점수는 0이고 정답 'RIVER'에 대한 부분점수는 0.2입니다. 따라서 합은 0.2입니다.
  - Missing characters 
    - <img src="https://i.imgur.com/8P3WNa8.png" alt="missing_characters" width="300"/>
    - 바람직한 점수는 0.56입니다.
    - 예측 결과 'SIDE'는 'IoU < 0.5'이므로 부분점수가 0입니다.
  - Overlapping characters
    - <img src="https://i.imgur.com/tmDV7b7.png" alt="overlapping_characters" width="300"/>
    - 바람직한 점수는 1입니다.
    - 예측 결과 'RIVER'과 'RSIDE' 모두 'IoU >= 0.5'라고 가정하면 둘 다 부분점수가 0.56이므로 합은 1.12입니다.
## Dataset
- '공종행정문서 OCR'의 전체 384.9GB 중 79.4GB (20%)밖에 사용하지 못했습니다. 그 이유는 첫째, 네트워크 속도가 제한되어 있는 상황 하에서 데이터셋을 다운로드 받는 데 매우 많은 시간이 소요되었으며 둘째, 사용 가능한 컴퓨팅 자원의 한계로 학습 중 자꾸 서버가 다운되는 현상이 발생하였기 때문입니다.
- 위와 비슷한 이유로 40,000개의 이미지에 대해서 약 1 epoch밖에 학습시키지 못했습니다.

# Future Improvements
## Metric
- CLEval ([CLEval: Character-Level Evaluation for Text Detection and Recognition Tasks](https://arxiv.org/pdf/2006.06244.pdf))
  - Not IoU-based evaluation metric.
  - 문자 단위로 Text detection and recognition을 평가하므로 좀 더 정교하게 성능 측정이 가능합니다.

# References
- Intersection over Union: [GIoU(Generalized Intersection over Union)](https://gaussian37.github.io/vision-detection-giou/)
- Metric: [calculate_mean_ap.py](https://gist.github.com/tarlen5/008809c3decf19313de216b9208f3734)
- EasyOCR training:
  - [What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis](https://github.com/clovaai/deep-text-recognition-benchmark/blob/master/train.py)
  - [EasyOCR 사용자 모델 학습하기](https://davelogs.tistory.com/76)
- Font: [나눔스퀘어 네오 Regular](https://campaign.naver.com/nanumsquare_neo/#download)
