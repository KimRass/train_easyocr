import os
import sys
import time
import random
import torch
import torch.nn as nn
from torch.nn import DataParallel
import torch.nn.init as init
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np

import yaml
from utils import (
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
from test import validation

cudnn.benchmark = True
cudnn.deterministic = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_parameters(model):
    print("Modules, Parameters")
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        param = parameter.numel()
        #table.add_row([name, param])
        total_params += param
        print(f"{name:40s}: {param:,}")
    print(f"Total Trainable Parameterss: {total_params:,}")
    return total_params


def train(opt, show_number=5, amp=False):
    """ Dataset preparation """
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')

    opt.select_data = opt.select_data.split("-")
    opt.batch_ratio = opt.batch_ratio.split("-")

    train_dataset = BatchBalancedDataset(opt)

    log = open(f'./saved_models/{opt.experiment_name}/log_dataset.txt', mode='a', encoding="utf8")

    AlignCollate_valid = AlignCollate(
        imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, contrast_adjust=opt.contrast_adjust
    )
    valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=min(32, opt.batch_size),
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers),
        prefetch_factor=512,
        collate_fn=AlignCollate_valid,
        pin_memory=True
    )
    log.write(valid_dataset_log)

    print("-" * 80)
    log.write("-" * 80 + '\n')
    log.close()
    
    """ Model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3

    model = Model(opt)
    # print(
    #     f'model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
    #     opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
    #     opt.SequenceModeling, opt.Prediction
    # )

    if opt.saved_model != "":
        state = torch.load(opt.saved_model, map_location=device)
        if opt.new_prediction:
            model.Prediction = nn.Linear(
                model.SequenceModeling_output, len(state['module.Prediction.weight'])
            )
        
        model = DataParallel(model).to(device) 
        print(f"Loaded pretrained model from '{opt.saved_model}'")
        if opt.FT:
            model.load_state_dict(state, strict=False)
        else:
            model.load_state_dict(state)
        if opt.new_prediction:
            model.module.Prediction = nn.Linear(model.module.SequenceModeling_output, opt.num_class)  
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
            except Exception as e:  # for batchnorm.
                if 'weight' in name:
                    param.data.fill_(1)
                continue
        model = DataParallel(model).to(device)
    
    model.train()

    print("Model:")
    print(model)
    count_parameters(model)
    
    """ Loss function """
    if 'CTC' in opt.Prediction:
        criterion = nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
    # loss averager
    loss_avg = Averager()

    # Freeze some layers
    try:
        if opt.freeze_FeatureFxtraction:
            for param in model.module.FeatureExtraction.parameters():
                param.requires_grad = False
        if opt.freeze_SequenceModeling:
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
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    """ Optimizer """
    if opt.optim == 'adam':
        #optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
        optimizer = optim.Adam(filtered_parameters)
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    print("Optimizer:")
    print(f"    {optimizer}")

    """ Final options """
    # print(opt)
    with open(f'./saved_models/{opt.experiment_name}/opt.txt', mode='a', encoding="utf8") as f:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        f.write(opt_log)

    """ Start training """
    # Continue training
    start_iter = 0
    if opt.saved_model != "":
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            print(f"Continue to train, start_iter: {start_iter}")
        except:
            pass

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    i = start_iter

    scaler = GradScaler()
    t1= time.time()
        
    while(True):
        # Training part
        optimizer.zero_grad(set_to_none=True)

        if amp:
            with autocast():
                image_tensors, labels = train_dataset.get_batch()
                image = image_tensors.to(device)
                text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
                batch_size = image.size(0)

                if 'CTC' in opt.Prediction:
                    preds = model(image, text).log_softmax(2)
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    preds = preds.permute(1, 0, 2)
                    cudnn.enabled = False 
                    loss = criterion(preds, text.to(device), preds_size.to(device), length.to(device))
                    cudnn.enabled = True
                else:
                    preds = model(image, text[:, :-1])  # align with Attention.forward
                    target = text[:, 1:]  # without [GO] Symbol
                    loss = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=opt.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            image_tensors, labels = train_dataset.get_batch()
            image = image_tensors.to(device)
            text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
            batch_size = image.size(0)
            if 'CTC' in opt.Prediction:
                preds = model(image, text).log_softmax(2)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                preds = preds.permute(1, 0, 2)
                cudnn.enabled = False
                loss = criterion(preds, text.to(device), preds_size.to(device), length.to(device))
                cudnn.enabled = True
            else:
                preds = model(image, text[:, :-1])  # align with Attention.forward
                target = text[:, 1:]  # without [GO] Symbol
                loss = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=opt.grad_clip) 
            optimizer.step()
        loss_avg.add(loss)

        # Validation part
        if (i % opt.valInterval == 0) and (i!=0):
            print(f"Training time: {time.time() - t1}")

            t1 = time.time()
            elapsed = time.time() - start_time
            # for log
            with open(f'./saved_models/{opt.experiment_name}/log_train.txt', mode='a', encoding="utf8") as log:
                model.eval()
                with torch.no_grad():
                    (
                        valid_loss,
                        current_accuracy,
                        current_norm_ED,
                        preds,
                        confidence_score,
                        labels,
                        infer_time,
                        length_of_data
                    ) = validation(model, criterion, valid_loader, converter, opt, device)
                model.train()

                # Training loss and validation loss
                loss_log = f'[{i}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, elapsed: {elapsed:0.5f}'
                loss_avg.reset()

                current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.4f}'

                # Keep best accuracy model (on validation set)
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(
                        model.state_dict(), f'./saved_models/{opt.experiment_name}/best_accuracy.pth'
                    )
                if current_norm_ED > best_norm_ED:
                    best_norm_ED = current_norm_ED
                    torch.save(
                        model.state_dict(), f'./saved_models/{opt.experiment_name}/best_norm_ED.pth'
                    )
                best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.4f}'

                loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                print(loss_model_log)
                log.write(loss_model_log + '\n')

                # Show some predicted results.
                dashed_line = "-" * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                
                # show_number = min(show_number, len(labels))
                
                start = random.randint(0, len(labels) - show_number)
                for gt, pred, confidence in zip(
                    labels[start: start + show_number],
                    preds[start : start + show_number],
                    confidence_score[start : start + show_number]
                ):
                    if 'Attn' in opt.Prediction:
                        gt = gt[: gt.find('[s]')]
                        pred = pred[: pred.find('[s]')]

                    predicted_result_log += f"{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n"
                predicted_result_log += f'{dashed_line}'
                print(predicted_result_log)
                log.write(predicted_result_log + '\n')
                print(f"Validation time: {time.time() - t1}", end="\n")
                t1=time.time()
        # Save model per 1e+4 iteration.
        if (i + 1) % 1e+4 == 0:
            torch.save(
                model.state_dict(), f'./saved_models/{opt.experiment_name}/iter_{i + 1}.pth')

        if i == opt.num_iter:
            print('end the training')
            sys.exit()
        i += 1


def main():
    file_path = "./config_files/configuration.yaml"
    with open(file_path, mode="r", encoding="utf8") as f:
        opt = AttrDict(
            yaml.safe_load(f)
        )
    #     opt = yaml.safe_load(f)
    # opt = AttrDict(opt)
    # opt.character = opt.number + opt.symbol + opt.lang_char
    os.makedirs(f"./saved_models/{opt.experiment_name}", exist_ok=True)

    train(opt)


if __name__ == "__main__":
    main()
