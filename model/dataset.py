import numpy as np
import torch
import os
from torch.utils.data import Dataset
from torchvision.transforms.functional import center_crop, resize
from torch.nn.functional import interpolate
import decord
decord.bridge.set_bridge('torch')

from PIL import Image
import torchvision.transforms as transforms

# Video dataset
def nb_pad(img, pad_hw):
    # bench: reflect-pad CHW frame to (H, W), centered, so the published strides divide it
    H, W = [int(x) for x in pad_hw.split('_')]
    h, w = img.shape[-2:]
    t, l = (H - h) // 2, (W - w) // 2
    return torch.nn.functional.pad(img[None], (l, W - w - l, t, H - h - t), mode='reflect')[0]


def nb_unpad(img, hw):
    # inverse of nb_pad for a batch [N, C, H, W]
    h, w = hw
    t, l = (img.shape[-2] - h) // 2, (img.shape[-1] - w) // 2
    return img[..., t:t + h, l:l + w]


class VideoDataSet_pn(Dataset):
    def __init__(self, args):
        self.args = args
        if os.path.isfile(args.data_path):
            self.video = decord.VideoReader(args.data_path)
        else:
            self.video = [os.path.join(args.data_path, x) for x in sorted(os.listdir(args.data_path))]
            # bench: decode every frame once into a uint8 RAM cache instead of re-reading PNGs each epoch
            self.cache = [torch.from_numpy(np.asarray(Image.open(x).convert("RGB"))).permute(2, 0, 1).contiguous() for x in self.video]

        self.crop_list, self.resize_list = args.crop_list, args.resize_list  
        self.pad_hw = args.pad_hw
        first_frame = self.img_transform(self.img_load(0))
        self.final_size = first_frame.size(-2) * first_frame.size(-1)

    def img_load(self, idx):
        if isinstance(self.video, list):
            img = self.cache[idx].float() / 255.
        else:
            img = self.video[idx].permute(-1,0,1)
        return img

    def img_transform(self, img):
        if self.crop_list != '-1': 
            crop_h, crop_w = [int(x) for x in self.crop_list.split('_')[:2]]
            if 'last' not in self.crop_list:
                img = center_crop(img, (crop_h, crop_w))
        if self.resize_list != '-1':
            if '_' in self.resize_list:
                resize_h, resize_w = [int(x) for x in self.resize_list.split('_')]
                img = interpolate(img, (resize_h, resize_w), 'bicubic')
            else:
                resize_hw = int(self.resize_list)
                img = resize(img, resize_hw,  'bicubic')
        if 'last' in self.crop_list:
            img = center_crop(img, (crop_h, crop_w))
        if getattr(self, 'pad_hw', ''):
            img = nb_pad(img, self.pad_hw)
        return img

    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx):
        tensor_image = self.img_transform(self.img_load(idx))
        # bench: neighbour frames feed only the temporal variant (SNeRV_T); skip them for SNeRV
        if self.args.model == 'snerv':
            tensor_image_p = tensor_image_n = tensor_image
        else:
            tensor_image_p = self.img_transform(self.img_load(idx-1 if idx != 0 else idx+1))
            tensor_image_n = self.img_transform(self.img_load(idx+1 if idx != (len(self.video)-1) else idx-1))
        norm_idx = float(idx) / len(self.video)
        sample = {'img': tensor_image, 'img_p': tensor_image_p, 'img_n' : tensor_image_n, 'norm_idx': norm_idx, 'idx': idx}
        
        return sample
        