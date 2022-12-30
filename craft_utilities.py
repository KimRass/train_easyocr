from pathlib import Path
import numpy as np
import cv2
import torch
import torchvision.transforms as T

from torch_utilities import (
    copy_state_dict
)
from models.craft import (
    CRAFT
)
# from models.craft_refiner import (
#     CRAFTRefiner
# )


def load_craft_checkpoint(cuda=False):
    craft = CRAFT()
    if cuda:
        craft = craft.to("cuda")

    ckpt_path = Path(__file__).parent/"pretrained/craft.pth"
    # ckpt_path = "/Users/jongbeom.kim/Desktop/workspace/text_spotting/pretrained/craft.pth"
    state = torch.load(ckpt_path, map_location="cuda" if cuda else "cpu")
    craft.load_state_dict(
        copy_state_dict(state), strict=True
    )

    print(f"Loaded pre-trained parameters for 'CRAFT' from checkpoint '{ckpt_path}'.")
    return craft


# def load_craft_refiner_checkpoint(cuda=False):
#     craft_refiner = CRAFTRefiner()
#     if cuda:
#         craft_refiner = craft_refiner.to("cuda")

#     ckpt_path = Path(__file__).parent/"pretrained/craft_refiner.pth"
#     # ckpt_path = "/Users/jongbeom.kim/Desktop/workspace/text_spotting/pretrained/craft_refiner.pth"
#     state = torch.load(ckpt_path, map_location="cuda" if cuda else "cpu")
#     craft_refiner.load_state_dict(
#         copy_state_dict(state), strict=True
#     )

#     print(f"Loaded pre-trained parameters for 'CRAFT' refiner from checkpoint '{ckpt_path}'.")
#     return craft_refiner


def resize_image_for_craft(img):
    # 가로와 세로가 각각 32의 배수가 되도록 Resize합니다.
    height, width, channel = img.shape

    # Make canvas and paste image
    height32, width32 = height, width
    if height % 32 != 0:
        height32 = height + (32 - height % 32)
    if width % 32 != 0:
        width32 = width + (32 - width % 32)

    canvas = np.zeros(shape=(height32, width32, channel), dtype=np.uint8)
    img_resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)
    canvas[: height, : width, :] = img_resized
    return canvas


def normalize_score_map(score_map):
    score_map -= np.min(score_map)
    score_map /= np.max(score_map)
    score_map *= 255
    score_map = score_map.astype(np.uint8)
    return score_map


def infer_using_craft(img, craft, cuda=False):
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )
    z = transform(img)
    z = z.unsqueeze(0)
    if cuda:
        z = z.to("cuda")

    craft.eval()
    with torch.no_grad():
        z, feature = craft(z)
    return z, feature


def get_text_score_map_and_link_score_map(img, craft, cuda=False):
    img_resized = resize_image_for_craft(img)

    z, _ = infer_using_craft(img=img_resized, craft=craft, cuda=cuda)
    
    z0 = z[0, :, :, 0].detach().cpu().numpy()
    z1 = z[0, :, :, 1].detach().cpu().numpy()

    height, width, _ = img.shape
    height_resized, width_resized, _ = img_resized.shape
    
    z0_resized = cv2.resize(src=z0, dsize=(width_resized, height_resized), interpolation=cv2.INTER_LANCZOS4)
    z0_resized = z0_resized[: height, : width]
    text_score_map = normalize_score_map(z0_resized)

    z1_resized = cv2.resize(src=z1, dsize=(width_resized, height_resized))
    z1_resized = z1_resized[: height, : width]
    link_score_map = normalize_score_map(z1_resized)
    return text_score_map, link_score_map


# def get_score_maps(
#     img, craft, craft_refiner=None, cuda=False
# ):
#     img_resized = resize_image_for_craft(img)

#     z, feature = infer_using_craft(img=img_resized, craft=craft, cuda=cuda)
    
#     z0 = z[0, :, :, 0].detach().cpu().numpy()
#     z1 = z[0, :, :, 1].detach().cpu().numpy()

#     height, width, _ = img.shape
#     height_resized, width_resized, _ = img_resized.shape
    
#     z0_resized = cv2.resize(src=z0, dsize=(width_resized, height_resized))
#     z0_resized = z0_resized[: height, : width]
#     text_score_map = normalize_score_map(z0_resized)
    
#     z1_resized = cv2.resize(src=z1, dsize=(width_resized, height_resized))
#     z1_resized = z1_resized[: height, : width]
#     link_score_map = normalize_score_map(z1_resized)

#     if craft_refiner is None:
#         return text_score_map, link_score_map
#     else:
#         with torch.no_grad():
#             z = craft_refiner(z, feature)
#         z0_ref = z[0, :, :, 0].detach().cpu().numpy()
        
#         z0_ref_resized = cv2.resize(src=z0_ref, dsize=(width_resized, height_resized))
#         z0_ref_resized = z0_ref_resized[: height, : width]
#         line_score_map = normalize_score_map(z0_ref_resized)
#         return text_score_map, link_score_map, line_score_map
