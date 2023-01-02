import os
import re
import math
import torch
import pandas  as pd
from pathlib import Path
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, Subset, DataLoader
from torch._utils import _accumulate
import torchvision.transforms as T


def contrast_grey(img):
    high = np.percentile(img, 90)
    low  = np.percentile(img, 10)
    return (high - low) / (high + low), high, low


def adjust_contrast_grey(img, target=0.4):
    contrast, high, low = contrast_grey(img)
    if contrast < target:
        img = img.astype(int)
        ratio = 200. / (high - low)
        img = (img - low + 25) * ratio
        img = np.maximum(
            np.full(img.shape, 0),
            np.minimum(
                np.full(img.shape, 255), img
            )
        ).astype(np.uint8)
    return img


class BatchBalancedDataset(object):
    def __init__(self, config):
        log = open(
            Path(__file__).parent/f"saved_models/{config.experiment_name}/log_dataset.txt",
            mode="a"
        )
        dashed_line = "-" * 80
        log.write(f"{dashed_line}\n")

        assert len(config.select_data) == len(config.batch_ratio)

        _AlignCollate = AlignCollate(
            img_height=config.img_height,
            img_width=config.img_width,
            keep_ratio_with_pad=config.PAD,
            contrast_adjust=config.contrast_adjust
        )

        self.data_loader_list = list()
        self.dataloader_iter_list = list()
        batch_size_list = list()
        Total_batch_size = 0
        for selected_d, batch_ratio_d in zip(config.select_data, config.batch_ratio):
            _batch_size = max(round(config.batch_size * float(batch_ratio_d)), 1)

            print(dashed_line)
            log.write(f"{dashed_line}\n")

            _dataset, _dataset_log = hierarchical_dataset(
                root=config.train_data, config=config, select_data=[selected_d]
            )
            total_number_dataset = len(_dataset)
            log.write(_dataset_log)

            number_dataset = int(
                total_number_dataset * float(config.total_data_usage_ratio)
            )
            dataset_split = [number_dataset, total_number_dataset - number_dataset]
            indices = range(total_number_dataset)
            _dataset, _ = [
                Subset(_dataset, indices[offset - length:offset])
                for offset, length
                in zip(
                    _accumulate(dataset_split),
                    dataset_split
                )
            ]
            selected_d_log = f"Number of samples of '{selected_d}': {len(_dataset):,}\n"
            selected_d_log += f"Number of samples of '{selected_d}' per batch: {_batch_size}"

            print(selected_d_log)
            log.write(f"{selected_d_log}\n")

            batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size

            _data_loader = DataLoader(
                dataset=_dataset,
                batch_size=_batch_size,
                shuffle=True,
                num_workers=int(config.workers),
                collate_fn=_AlignCollate,
                pin_memory=True,
                drop_last=True,
                prefetch_factor=2,
                persistent_workers=True,
            )
            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(iter(_data_loader))

        Total_batch_size_log = f'{dashed_line}\n'
        batch_size_sum = '+'.join(batch_size_list)
        Total_batch_size_log += f'Total batch size: {batch_size_sum} = {Total_batch_size}\n'
        Total_batch_size_log += f'{dashed_line}'
        config.batch_size = Total_batch_size

        print(Total_batch_size_log)
        log.write(f"{Total_batch_size_log}\n")
        log.close()

    def get_batch(self):
        balanced_batch_images = list()
        balanced_batch_texts = list()

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image, text = next(data_loader_iter) # Fix here!
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, text = next(self.dataloader_iter_list[i]) # Fix here!
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except ValueError:
                pass

        balanced_batch_images = torch.cat(balanced_batch_images, 0)
        return balanced_batch_images, balanced_batch_texts


# `select_data="/"` contains all sub-directory of root directory
def hierarchical_dataset(root, config, select_data="/"):
    dataset_list = list()
    dataset_log = f"Dataset root: '{root}'"
    print(dataset_log)

    dataset_log += "\n"
    for dirpath, dirnames, filenames in os.walk(root + "/"):
        if not dirnames:
            select_flag = False
            for selected_d in select_data:
                if selected_d in dirpath:
                    select_flag = True
                    break

            if select_flag:
                dataset = OCRDataset(dirpath, config)
                sub_dataset_log = f"Subdirectory: '{os.path.relpath(dirpath, root)}'\nNumber of samples: {len(dataset):,}"
                print(sub_dataset_log)
                dataset_log += f'{sub_dataset_log}\n'
                dataset_list.append(dataset)

    concatenated_dataset = ConcatDataset(dataset_list)
    return concatenated_dataset, dataset_log


class OCRDataset(Dataset):
    def __init__(self, root, config):
        self.root = root
        self.config = config
        self.df = pd.read_csv(
            os.path.join(
                os.path.dirname(root), 'labels.csv'
            ),
            sep='^([^,]+),',
            engine='python',
            usecols=['filename', 'words'],
            keep_default_na=False
        )
        self.n_samples = len(self.df)

        if self.config.data_filtering_off:
            self.filtered_index_list = [index + 1 for index in range(self.n_samples)]
        else:
            self.filtered_index_list = list()
            for index in range(self.n_samples):
                label = self.df.at[index, 'words']
                try:
                    if len(label) > self.config.batch_max_length:
                        continue
                except:
                    print(label)
                out_of_char = f'[^{self.config.character}]'
                if re.search(out_of_char, label.lower()):
                    continue
                self.filtered_index_list.append(index)
            self.n_samples = len(self.filtered_index_list)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        index = self.filtered_index_list[index]
        img_fname = self.df.at[index,'filename']
        img_fpath = os.path.join(self.root, img_fname)
        label = self.df.at[index,'words']

        if self.config.rgb:
            img = Image.open(img_fpath).convert('RGB')
        else:
            img = Image.open(img_fpath).convert('L')

        if not self.config.sensitive:
            label = label.lower()

        # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
        out_of_char = f'[^{self.config.character}]'
        label = re.sub(out_of_char, '', label)

        return (img, label)

class ResizeNormalize(object):
    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = T.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class NormalizePAD(object):
    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = T.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # Right pad
        if self.max_size[2] != w:  # Add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)
        return Pad_img


class AlignCollate(object):
    def __init__(self, img_height=32, img_width=100, keep_ratio_with_pad=False, contrast_adjust=0.):
        self.img_height = img_height
        self.img_width = img_width
        self.keep_ratio_with_pad = keep_ratio_with_pad
        self.contrast_adjust = contrast_adjust

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        if self.keep_ratio_with_pad:  # Same concept with 'Rosetta' paper
            resized_max_w = self.img_width
            input_channel = 3 if images[0].mode == 'RGB' else 1
            transform = NormalizePAD((input_channel, self.img_height, resized_max_w))

            resized_images = list()
            for image in images:
                w, h = image.size

                # Augmentation here - change contrast
                if self.contrast_adjust > 0:
                    image = np.array(image.convert("L"))
                    image = adjust_contrast_grey(image, target = self.contrast_adjust)
                    image = Image.fromarray(image, 'L')

                ratio = w / float(h)
                if math.ceil(self.img_height * ratio) > self.img_width:
                    resized_w = self.img_width
                else:
                    resized_w = math.ceil(self.img_height * ratio)

                resized_image = image.resize((resized_w, self.img_height), Image.BICUBIC)
                resized_images.append(transform(resized_image))

            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        else:
            transform = ResizeNormalize((self.img_width, self.img_height))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)
        return image_tensors, labels


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
