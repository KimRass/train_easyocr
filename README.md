# How to Run
## Environment Setting
```sh
source set_environment.sh
```
## Dataset Preparation
```sh
# Example
python3 prepare_dataset.py\
  --dataset="/data/original"\ # Path to the original dataset "공공행정문서 OCR"
  --unzip\ # Whether to unzip
  --training\ # Whether to generate training set
  --validation\ # Whether to generate validation set
  --evaluation # Whether to generate evaluation set
```
## Training (Fine-tunning)
```sh
python3 train_easyocr/train.py
```
## Fine-tuned Model Setting
```sh
# Example
cp train_easyocr/saved_models/none-vgg-bilstm-ctc_1111/best_accuracy.pth ~/.EasyOCR/model/finetuned.pth
```
## Evaluation
```sh
# Example
python3 evaluate.py\
  --eval_set="/data/evaluation_set"\ # Path to the evaluation set
  --baseline\ # Whether to evaluate EasyOCR baseline model
  --finetuned\ # Whether to evaluate fine-tuned model
  --cuda # Whether to use GPU
```

# Dataset
## Original
```
공공행정문서 OCR
├── Training (102,477)
│   ├── [라벨]train.zip
│   ├── [원천]train1.zip
│   ├── [원천]train3.zip
│   └── [원천]train2.zip
│   └── ...
└── Validation (11,397)
    ├── [라벨]validation.zip
    └── [원천]validation.zip
```
## "unzipped"
```
unzipped
├── training
│   ├── images
│   └── labels
└── validation
    ├── images
    └── labels
```
## "training_and_validation_set"
```
training_and_validation_set
├── training (1,120,590)   *(2,144,196) *(2,038,547?)
│   └── select_data
│       ├── images
│       │   └── ...
│       └── labels.csv
└── validation *(211,323) (201,254)   (398,731) (379,235?)
    └── select_data
        ├── images
        │   └── ...
        └── labels.csv
```
- "labels.csv"
  |filename|words|
  |-|-|
  |5350178-1999-0001-0344_829-262-1003-318.png|김해를|
  |5350178-1999-0001-0344_1022-262-1215-321.png|아름답게|
  |5350178-1999-0001-0344_1231-259-1384-324.png|시민을|
  |5350178-1999-0001-0344_1405-259-1620-324.png|행복하게|
  |...|...|
## "evaluation_set"
- 484 - 1
```
evaluation_set
├── images
│   ├── ...
└── labels
    └── ...
```
## Preprocessing
- 일부 좌표가 음수인 경우 0으로 수정했습니다.

# Configuration
- Batch size: 64
- Optimizer: Adadelta (Learning rate: 1, rho: 0.95, eps: 1e-8)
  Adadelta (
Parameter Group 0
    eps: 1e-08
    foreach: None
    lr: 1.0
    maximize: False
    rho: 0.95
    weight_decay: 0
)

# Training
- Number of image patches
  - Training set: 1,120,590
  - Validation set: 211,323
- Batch size: 16
- Number of iterations per epoch: 70,036

- Maximum length of label (`batch_max_length`): 34
- Input image resolution: 600 x 64 (Width x Height)
- Characters:
  - Numbers: 0123456789
  - Special characters: !\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
  - Alphabets: ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
  - Hangul: 가각간갇갈감갑값강갖같갚갛개객걀거걱건걷걸검겁것겉게겨격겪견결겹경곁계고곡곤곧골곰곱곳공과관광괜괴굉교구국군굳굴굵굶굽궁권귀규균그극근글긁금급긋긍기긴길김깅깊까깎깐깔깜깝깥깨꺼꺾껍껏껑께껴꼬꼭꼴꼼꼽꽂꽃꽉꽤꾸꿀꿈뀌끄끈끊끌끓끔끗끝끼낌나낙낚난날낡남납낫낭낮낯낱낳내냄냉냐냥너넉널넓넘넣네넥넷녀녁년념녕노녹논놀놈농높놓놔뇌뇨누눈눕뉘뉴늄느늑는늘늙능늦늬니닐님다닥닦단닫달닭닮담답닷당닿대댁댐더덕던덜덤덥덧덩덮데델도독돈돌돕동돼되된두둑둘둠둡둥뒤뒷드득든듣들듬듭듯등디딩딪따딱딴딸땀땅때땜떠떡떤떨떻떼또똑뚜뚫뚱뛰뜨뜩뜯뜰뜻띄라락란람랍랑랗래랜램랫략량러럭런럴럼럽럿렁렇레렉렌려력련렬렵령례로록론롬롭롯료루룩룹룻뤄류륙률륭르른름릇릎리릭린림립릿마막만많말맑맘맙맛망맞맡맣매맥맨맵맺머먹먼멀멈멋멍멎메멘멩며면멸명몇모목몰몸몹못몽묘무묵묶문묻물뭄뭇뭐뭣므미민믿밀밉밌및밑바박밖반받발밝밟밤밥방밭배백뱀뱃뱉버번벌범법벗베벤벼벽변별볍병볕보복볶본볼봄봇봉뵈뵙부북분불붉붐붓붕붙뷰브블비빌빗빚빛빠빨빵빼뺨뻐뻔뻗뼈뽑뿌뿐쁘쁨사삭산살삶삼상새색샌생서석섞선설섬섭섯성세센셈셋션소속손솔솜솟송솥쇄쇠쇼수숙순술숨숫숲쉬쉽슈스슨슬슴습슷승시식신싣실싫심십싱싶싸싹쌀쌍쌓써썩썰썹쎄쏘쏟쑤쓰쓸씀씌씨씩씬씹씻아악안앉않알앓암압앗앙앞애액야약얇양얗얘어억언얹얻얼엄업없엇엉엌엎에엔엘여역연열엷염엽엿영옆예옛오옥온올옮옳옷와완왕왜왠외왼요욕용우욱운울움웃웅워원월웨웬위윗유육율으윽은을음응의이익인일읽잃임입잇있잊잎자작잔잖잘잠잡장잦재쟁저적전절젊점접젓정젖제젠젯져조족존졸좀좁종좋좌죄주죽준줄줌줍중쥐즈즉즌즐즘증지직진질짐집짓징짙짚짜짝짧째쨌쩌쩍쩐쪽쫓쭈쭉찌찍찢차착찬찮찰참창찾채책챔챙처척천철첫청체쳐초촉촌총촬최추축춘출춤춥춧충취츠측츰층치칙친칠침칭카칸칼캐캠커컨컬컴컵컷켓켜코콜콤콩쾌쿠퀴크큰클큼키킬타탁탄탈탑탓탕태택탤터턱털텅테텍텔템토톤톱통퇴투툼퉁튀튜트특튼튿틀틈티틱팀팅파팎판팔패팩팬퍼퍽페펴편펼평폐포폭표푸푹풀품풍퓨프플픔피픽필핏핑하학한할함합항해핵핸햄햇행향허헌험헤헬혀현혈협형혜호혹혼홀홍화확환활황회획횟효후훈훌훔훨휘휴흉흐흑흔흘흙흡흥흩희흰히힘

# Fine-tunning
- Pre-trained Model: [korean_g2](https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/korean_g2.zip)
## Appling Fined-tuned Model
```
~/.EasyOCR
├── model
│   ├── finetuned.pth
└── user_network
    ├── finetuned.py
    └── finetuned.yaml
```

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
import jiwer

cer = jiwer.cer(gt, pred)
wer = jiwer.wer(gt, pred)
```
## Limitations of Evaluation Metric
- Reference: https://arxiv.org/pdf/2006.06244.pdf
- IoU + CRW (Correctly Recognized Words)
- One-to-many 또는 Many-to-one 상황에서 대응 x
## CLEval
- Not IoU-based evaluation metric.

# Evaluation
- Number of images: 484
## Baseline
- F1 score: 0.53

# Library Comparison
## PaddleOCR
- Text detection: DBNet (AAAI'2020)
- Text recognition: ABINet (CVPR'2021)
## MMOCR
- Text detection: DB_r18
- Text recognition: ABINet
## EasyOCR
- Text detection: CRAFT (Default), DBNet (18)
- Text recognition
  - Transformation: None or TPS ([Thin Plate Spline](https://en.wikipedia.org/wiki/Thin_plate_spline))
  - Feature extraction: VGG, RCNN or ResNet
  - Sequence modeling: None or BiLSTM
  - Prediction: CTC or Attention
  - (None-VGG-BiLSTM-CTC)


# Improvements
## Hyperparameter Tunning
### Beam Search
- `decoder`: `"greedy"` -> `"beam"`?
## Image Processing
## Fine Tunning

# To Do
- 문맥을 고려한 교정

# References
- Baseline: https://github.com/JaidedAI/EasyOCR
- CRAFT: https://github.com/clovaai/CRAFT-pytorch
- Intersection over Union: https://gaussian37.github.io/vision-detection-giou/
- Metric: https://gist.github.com/tarlen5/008809c3decf19313de216b9208f3734
- [What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis](https://github.com/clovaai/deep-text-recognition-benchmark/blob/master/train.py)
- https://davelogs.tistory.com/82

Total number of trainable parameters: 4,015,729