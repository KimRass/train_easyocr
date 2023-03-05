import sys
from pathlib import Path
from time import time
import random
import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.nn import DataParallel
import torch.nn.init as init
import torch.backends.cudnn as cudnn
from torch.optim import Adam, Adadelta
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm

from utils import (
    get_elapsed_time,
    AttrDict,
    CTCLabelConverter,
    AttnLabelConverter,
    Averager
)
from dataset import (
    hierarchical_dataset,
    AlignCollate,
    BatchBalancedDataset
)
from model import Model
from train_easyocr.test import validation

cudnn.benchmark = True
cudnn.deterministic = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(config, amp=False):
    experiment_dir = Path(__file__).parent/f"saved_models/{config.experiment_name}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    dashed_line = "-" * 80
    """ Dataset preparation """
    if not config.data_filtering_off:
        print(dashed_line)
        print("Filtering the images containing characters not in `config.character`")
        print("Filtering the images whose label is longer than `config.batch_max_length`")

    config.select_data = config.select_data.split("-")
    config.batch_ratio = config.batch_ratio.split("-")

    train_dataset = BatchBalancedDataset(config)

    log = open(experiment_dir/"log_dataset.txt", mode='a', encoding="utf8")

    AlignCollate_val = AlignCollate(
        img_height=config.img_height,
        img_width=config.img_width,
        keep_ratio_with_pad=config.PAD,
        contrast_adjust=config.contrast_adjust
    )
    val_dataset, val_dataset_log = hierarchical_dataset(root=config.val_data, config=config)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(config.workers),
        prefetch_factor=512,
        collate_fn=AlignCollate_val,
        pin_memory=True,
        drop_last=True
        
    )
    log.write(val_dataset_log)

    print("-" * 80)
    log.write("-" * 80 + '\n')
    log.close()
    
    """ Model configuration """
    if "CTC" in config.Prediction:
        converter = CTCLabelConverter(config.character)
    else:
        converter = AttnLabelConverter(config.character)
    config.num_class = len(converter.character)

    if config.rgb:
        config.input_channel = 3

    model = Model(config)

    if config.continue_from != "":
        state = torch.load(config.continue_from, map_location=device)
        if config.new_prediction:
            model.Prediction = nn.Linear(
                model.SequenceModeling_output, len(state['module.Prediction.weight'])
            )
        
        model = DataParallel(model).to(device)
        print(
            f"Loaded trained parameters from checkpoint\n\
                '{config.continue_from}'"
        )
        if config.strict:
            model.load_state_dict(state, strict=False)
        else:
            model.load_state_dict(state)
        if config.new_prediction:
            model.module.Prediction = nn.Linear(model.module.SequenceModeling_output, config.num_class)  
            for name, param in model.module.Prediction.named_parameters():
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)
            model = model.to(device) 
    else:
        # Weight initialization
        for name, param in model.named_parameters():
            if 'localization_fc2' in name:
                print(f'Skip {name} as it is already initialized')
                continue
            try:
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)
            except Exception as e:  # For batch normalization.
                if 'weight' in name:
                    param.data.fill_(1)
                continue
        model = DataParallel(model).to(device)
    
    model.train()
    
    """ Loss function """
    if "CTC" in config.Prediction:
        criterion = nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
    # Loss averager
    loss_avg = Averager()

    # Freeze some layers
    try:
        if config.freeze_FeatureFxtraction:
            for param in model.module.FeatureExtraction.parameters():
                param.requires_grad = False
        if config.freeze_SequenceModeling:
            for param in model.module.SequenceModeling.parameters():
                param.requires_grad = False
    except:
        pass
    
    # Filter that only require gradient decent
    filtered_parameters = list()
    params_num = list()
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print(f"Total number of trainable parameters: {sum(params_num):,}")

    """ Optimizer """
    if config.adam:
        optimizer = Adam(params=filtered_parameters, lr=config.lr, betas=(config.beta1, 0.999))
    else:
        optimizer = Adadelta(params=filtered_parameters, lr=config.lr, rho=config.rho, eps=config.eps)
    # print("optimizer:")
    # print(f"    {optimizer}")

    """ Final configs """
    with open(experiment_dir/"config.txt", mode='a', encoding="utf8") as f:
        config_log = '------------ Configuration -------------\n'
        args = vars(config)
        for k, v in args.items():
            config_log += f'{str(k)}: {str(v)}\n'
        config_log += '---------------------------------------\n'
        print(config_log)
        f.write(config_log)

    """ Start training """
    # Continue training
    start_iter = 0
    if config.continue_from != "":
        try:
            start_iter = int(config.continue_from.split('_')[-1].split('.')[0])
            print(f"Continue to train from iteration {start_iter}")
        except:
            pass

    start_time = time()
    best_accuracy = -1
    best_norm_ed = -1
    i = start_iter

    scaler = GradScaler()
    t1 = time()

    while True:
        # Training part
        optimizer.zero_grad(set_to_none=True)

        if amp:
            with autocast():
                image_tensors, labels = train_dataset.get_batch()
                image = image_tensors.to(device)
                text, length = converter.encode(labels, batch_max_length=config.batch_max_length)
                batch_size = image.size(0)

                if "CTC" in config.Prediction:
                    preds = model(image, text).log_softmax(2)
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    preds = preds.permute(1, 0, 2)
                    cudnn.enabled = False 
                    loss = criterion(preds, text.to(device), preds_size.to(device), length.to(device))
                    cudnn.enabled = True
                else:
                    preds = model(image, text[:, :-1])  # Align with Attention.forward
                    target = text[:, 1:]  # Without [GO] Symbol
                    loss = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            image_tensors, labels = train_dataset.get_batch()
            image = image_tensors.to(device)
            text, length = converter.encode(labels, batch_max_length=config.batch_max_length)
            batch_size = image.size(0)

            if "CTC" in config.Prediction:
                preds = model(image, text).log_softmax(2)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                preds = preds.permute(1, 0, 2)
                cudnn.enabled = False
                loss = criterion(preds, text.to(device), preds_size.to(device), length.to(device))
                cudnn.enabled = True
            else:
                preds = model(image, text[:, :-1])  # Align with Attention.forward
                target = text[:, 1:]  # Without [GO] Symbol
                loss = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.grad_clip) 
            optimizer.step()
        loss_avg.add(loss)

        # Validation part
        if (i % config.val_period == 0) and (i != 0):
            print(f"Training time: {get_elapsed_time(t1)}")

            t1 = time()
            with open(experiment_dir/"log_train.txt", mode='a', encoding="utf8") as log:
                model.eval()
                with torch.no_grad():
                    (
                        val_loss,
                        current_accuracy,
                        current_norm_ed,
                        preds,
                        confidence_score,
                        labels,
                        infer_time,
                        length_of_data
                    ) = validation(
                        model=model,
                        criterion=criterion,
                        val_loader=val_loader,
                        converter=converter,
                        config=config,
                        device=device
                    )

                model.train()

                # Training loss and valation loss
                loss_log = f"[{i}/{config.n_iter}]\nTraining loss: {loss_avg.val():0.5f} | Validation loss: {val_loss:0.5f} | Total {get_elapsed_time(start_time)} elapsed"
                loss_avg.reset()

                current_model_log = f'{"Current accuracy":17s}: {current_accuracy:0.3f} | {"Current normalized edit distance":30s}: {current_norm_ed:0.4f}'

                # Keep best accuracy model (on validation set)
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(
                        model.state_dict(), experiment_dir/"best_accuracy.pth"
                    )
                if current_norm_ed > best_norm_ed:
                    best_norm_ed = current_norm_ed
                    torch.save(
                        model.state_dict(), experiment_dir/"best_norm_ed.pth"
                    )
                best_model_log = f'{"Best accuracy":17s}: {best_accuracy:0.3f} | {"Best normalized edit distance":30s}: {best_norm_ed:0.4f}'

                loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                print(loss_model_log)
                log.write(loss_model_log + '\n')

                # Show some predicted results.
                head = f'{"Ground Truth":25s}  |  {"Prediction":25s}  |  Confidence Score & T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                
                start = random.randint(0, len(labels) - config.show_number)
                for gt, pred, confidence in zip(
                    labels[start: start + config.show_number],
                    preds[start: start + config.show_number],
                    confidence_score[start: start + config.show_number]
                ):
                    if 'Attn' in config.Prediction:
                        gt = gt[: gt.find('[s]')]
                        pred = pred[: pred.find('[s]')]

                    predicted_result_log += f"{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n"
                predicted_result_log += f'{dashed_line}'
                print(predicted_result_log)
                log.write(predicted_result_log + '\n')
                print(f"Validation time: {get_elapsed_time(t1)}\n")
                t1 = time()
        # Save model every 1e+4 iteration.
        if (i + 1) % 1e+4 == 0:
            torch.save(
                model.state_dict(), experiment_dir/f"iter_{i + 1}.pth"
            )

        if i == config.n_iter:
            print("The end the training")
            sys.exit()

        i += 1


def main():
    with open(
        Path(__file__).parent/"config_files/config.yaml", mode="r", encoding="utf8"
    ) as f:
        config = AttrDict(
            yaml.safe_load(f)
        )

    train(config)


if __name__ == "__main__":
    main()
