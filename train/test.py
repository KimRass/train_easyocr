from time import time
from tqdm.auto import tqdm
import torch
import torch.utils.data
import torch.nn.functional as F
from nltk.metrics.distance import edit_distance

from utils import (
    get_elapsed_time,
    Averager
)


def validation(model, criterion, val_loader, converter, config, device):
    """ validation or evaluation """
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()

    t1 = time()
    for i, (image_tensors, labels) in enumerate(val_loader):
        if i % 10000 == 0 and i != 0:
            print(i, get_elapsed_time(t1))
            t1 = time()
    # for image_tensors, labels in tqdm(val_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        # For max length prediction
        length_for_pred = torch.IntTensor([config.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, config.batch_max_length + 1).fill_(0).to(device)

        text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=config.batch_max_length)
        
        start_time = time()
        if 'CTC' in config.Prediction:
            preds = model(image, text_for_pred)
            forward_time = time() - start_time

            # Calculate evaluation loss for CTC decoder.
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            loss = criterion(
                # Permute 'preds' to use `nn.CTCloss` format
                log_probs=preds.log_softmax(2).permute(1, 0, 2),
                targets=text_for_loss,
                input_lengths=preds_size,
                target_lengths=length_for_loss
            )

            if config.decode == 'greedy':
                # Select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_index = preds_index.view(-1)
                preds_str = converter.decode_greedy(preds_index.data, preds_size.data)
            elif config.decode == 'beamsearch':
                preds_str = converter.decode_beamsearch(preds, beamWidth=2)

        else:
            preds = model(image, text_for_pred, is_train=False)
            forward_time = time() - start_time

            preds = preds[:, :text_for_loss.shape[1] - 1, :]
            target = text_for_loss[:, 1:]  # without [GO] Symbol
            loss = criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
            labels = converter.decode(text_for_loss[:, 1:], length_for_loss)

        infer_time += forward_time
        valid_loss_avg.add(loss)

        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []
        
        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            if 'Attn' in config.Prediction:
                gt = gt[:gt.find('[s]')]
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

            if pred == gt:
                n_correct += 1

            '''
            (old version) ICDAR2017 DOST Normalized Edit Distance https://rrc.cvc.uab.es/?ch=7&com=tasks
            "For each word we calculate the normalized edit distance to the length of the ground truth transcription." 
            if len(gt) == 0:
                norm_ED += 1
            else:
                norm_ED += edit_distance(pred, gt) / len(gt)
            '''
            
            # ICDAR2019 Normalized Edit Distance 
            if len(gt) == 0 or len(pred) ==0:
                norm_ED += 0
            elif len(gt) > len(pred):
                norm_ED += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ED += 1 - edit_distance(pred, gt) / len(pred)

            # calculate confidence score (= multiply of pred_max_prob)
            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
            confidence_score_list.append(confidence_score)
            # print(pred, gt, pred==gt, confidence_score)

    accuracy = n_correct / float(length_of_data) * 100
    norm_ED = norm_ED / float(length_of_data) # ICDAR2019 Normalized Edit Distance

    return (
        valid_loss_avg.val(),
        accuracy,
        norm_ED,
        preds_str,
        confidence_score_list,
        labels,
        infer_time,
        length_of_data
    )
