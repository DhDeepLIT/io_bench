import os
import warnings
from argparse import Namespace
from typing import Optional
from pathlib import Path
from typing import List, Optional

import albumentations as A
# import cv2
import numpy as np
import pandas as pd
import rasterio
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader, Dataset
import cv2
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

from datetime import datetime
import sys

class TransformedDataset(Dataset):
    def __init__(self, ds, transform_fn):
        assert isinstance(ds, Dataset)
        assert callable(transform_fn)
        self.ds = ds
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.ds)

    def shuffle(self, random_state=None):
        self.ds.shuffle(random_state=random_state)

    def collate_fn(self, data):
        full_images = torch.stack([d["image"] for d in data], dim=0)
        full_labels = torch.stack([d["mask"] for d in data], dim=0)
        _B, C, _H, _W = full_images.shape
        if self.ds.siamese:
            images_1 = full_images[..., :C//2, :, :]
            images_2 = full_images[..., C//2:, :, :]
            return images_1, images_2, full_labels
        else:
            return full_images, full_labels

    def __getitem__(self, index):
        #dat1 = datetime.utcnow().isoformat(sep=' ', timespec='milliseconds')
        #print(f' {dat1} request index {index} ')
        #sys.stdout.flush()
        dp = self.ds[index]
        #dat2 = datetime.utcnow().isoformat(sep=' ', timespec='milliseconds')
        #print(f' {dat2}            fetched index {index} ')
        #sys.stdout.flush()
        return self.transform_fn(**dp)


class StackDataset(Dataset):
    """Reads in images, transforms pixel values, and serves a
    dictionary containing chip ids, image tensors, and
    label masks (where available).
    """

    def __init__(self,
                 dataset: pd.DataFrame,
                 img_folder: str,
                 selected_channels: Optional[list] = [],
                 label_folder: Optional[str] = None,
                 return_meta: Optional[bool] = False,
                 class_mapping: Optional[dict] = None, 
                 stratif_method: Optional[str] = None,
                 stratif_args: Optional[dict] = None,
                 siamese: Optional[bool] = True):
        """
        Instantiate the CloudDataset class.

        Args:
            dataset (pd.DataFrame): a dataframe with a row for each chip. There must be a column for chip_id,
                and a column with the path to the TIF for each of bands
            bands (list[str]): list of the bands included in the data
            labels (pd.DataFrame, optional): a dataframe with, for each chip, columns for chip_id
                and the path to the label TIF with ground truth cloud cover
        """
        super().__init__()
        self.base_dataset = dataset
        self.label_folder = label_folder
        self.return_meta = return_meta
        self.img_folder = img_folder
        self.selected_channels = selected_channels
        self.class_mapping = class_mapping
        self.stratif_method = stratif_method
        self.stratif_args = stratif_args
        self.siamese = siamese
        # print(stratif_args)
        #self.stratif()
        if self.stratif_method=="stratif_by_class":
            self.dataset = stratif_by_class(df=self.base_dataset, stratif_args=self.stratif_args)
        elif self.stratif_method=="stratif_by_batch":
            self.dataset = stratif_by_batch(df=self.base_dataset, stratif_args=self.stratif_args)
        else:
            self.dataset = self.base_dataset


    def __len__(self):
        return len(self.dataset)

    def shuffle(self, random_state=None):
        self.stratif()
        if not self.stratif_method == "stratif_by_batch":
            self.dataset = self.dataset.sample(frac=1, random_state=random_state)

    def stratif(self):
        if self.stratif_method=="stratif_by_class":
            self.dataset = stratif_by_class(df=self.base_dataset, stratif_args=self.stratif_args)
        elif self.stratif_method=="stratif_by_batch":
            self.dataset = stratif_by_batch(df=self.base_dataset, stratif_args=self.stratif_args)
        else:
            self.dataset = self.base_dataset

    def __getitem__(self, idx: int):

        # Loads an n-channel image from a chip-level dataframe
        row = self.dataset.iloc[idx]

        img_path = os.path.join(self.img_folder, row["in"].replace('in:', ''))
        with rasterio.open(img_path) as src:

            if len(self.selected_channels) > 0:
                # print("selecting some channels", flush=True)
                # print(self.selected_channels, flush=True)
                x_arr = src.read(self.selected_channels).astype("float32")
                # print(x_arr.shape, flush=True)
            else:
                # print("opening stack image", flush=True)
                x_arr = src.read().astype("float32")
                # print(x_arr.shape, flush=True)
        x_arr = x_arr.transpose(1, 2, 0)
        x_arr = x_arr / eval(row["range_max"].replace("range_max:", ''))

        # Generate mask from RLE
        if row['RLE'] == 'RLE:None':
            rle = '[]'
        else:
            rle = row['RLE'].replace('RLE:', '')

        # Decode the RLE
        c_mask = masksAsImage(rle,
                              shape=(x_arr.shape[0], x_arr.shape[1]),
                              class_mapping=self.class_mapping)

        if self.return_meta:
            return {
                "image": x_arr,
                "mask": c_mask,
                "meta": {
                    "chip_id": row["in"],
                    "image_path": img_path,
                    # "mask_path": label_path,
                },
            }

        return {"image": x_arr, "mask": c_mask}


def setup_data(config: Namespace):
    train_csv = pd.read_csv(config.train_csv, sep=';', index_col=False)
    valid_csv = pd.read_csv(config.valid_csv, sep=';', index_col=False)

    train_csv.rename(columns={train_csv.columns[0]: train_csv.columns[0].replace('#', '')}, inplace=True)
    valid_csv.rename(columns={valid_csv.columns[0]: valid_csv.columns[0].replace('#', '')}, inplace=True)

    img_folder = os.path.dirname(os.path.realpath(config.train_csv))
    valid_img_folder = os.path.dirname(os.path.realpath(config.valid_csv))

    if config.transform == 'coherence-sar':
        print("Transform for images with coherence in band 1 and SAR amplitude in band 2 and 3 will be used")
        transform_train = A.Compose(
            [
                #A.PadIfNeeded(config.img_size, config.img_size), # , border_mode=cv2.BORDER_REFLECT_101, additional argument, cv2.BORDER_REFLECT_101 = default
                A.CenterCrop(config.img_size, config.img_size),
                #A.RandomCrop(config.img_size, config.img_size),
                #A.HorizontalFlip(p=0.5),
                #A.VerticalFlip(p=0.5),
                A.Normalize(mean=config.img_mean, std=config.img_std, max_pixel_value=1.),
                ToTensorV2(),
            ]
        )
    else:
        print("Default transform will be used")
        transform_train = A.Compose(
            [
                #A.RandomScale(scale_limit=0.1, p=1.0),
                #A.PadIfNeeded(config.img_size, config.img_size), # , border_mode=cv2.BORDER_REFLECT_101, additional argument, cv2.BORDER_REFLECT_101 = default
                #A.Rotate(limit=(-90, 90), p=1.0),
                A.CenterCrop(config.img_size, config.img_size),
                #A.RandomCrop(config.img_size, config.img_size),
                #A.HorizontalFlip(p=0.5),
                #A.VerticalFlip(p=0.5),
                #A.RandomBrightness(limit=0.1, always_apply=True),
                A.Normalize(mean=config.img_mean, std=config.img_std, max_pixel_value=1.),
                ToTensorV2(),
            ]
        )

    transform_eval = A.Compose(
        [
            #A.PadIfNeeded(config.img_size, config.img_size), # , border_mode=cv2.BORDER_REFLECT_101, additional argument, cv2.BORDER_REFLECT_101 = default
            A.CenterCrop(config.img_size, config.img_size),
            A.Normalize(mean=config.img_mean, std=config.img_std, max_pixel_value=1.),
            ToTensorV2(),
        ]
    )

    print(config.class_mapping)

    dataset_train = StackDataset(dataset=train_csv, img_folder=img_folder, class_mapping=config.class_mapping, selected_channels=config.selected_channels, stratif_method=config.stratif_method, stratif_args=config.stratif_args, siamese=config.siamese)
    dataset_eval = StackDataset(dataset=valid_csv, img_folder=valid_img_folder, class_mapping=config.class_mapping, selected_channels=config.selected_channels, siamese=config.siamese)

    dataset_train = TransformedDataset(ds=dataset_train, transform_fn=transform_train)
    dataset_eval = TransformedDataset(ds=dataset_eval, transform_fn=transform_eval)

    dataloader_train = DataLoader(
        dataset_train,
        shuffle=False,
        batch_size=config.train_batch_size,
        num_workers=config.num_workers,
        collate_fn=dataset_train.collate_fn,
        drop_last=False,
        #timeout=5,
        #persistent_workers=True,
        #worker_init_fn = worker_init_fn,
        pin_memory=False
    )
    dataloader_eval = DataLoader(
        dataset_eval,
        shuffle=False,
        batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        collate_fn=dataset_eval.collate_fn,
        #worker_init_fn = worker_init_fn,
        drop_last=False,
        #timeout=5,
        #persistent_workers=True,
        pin_memory=False
    )

    return dataloader_train, dataloader_eval


def denormalize(t, mean, std, max_pixel_value=255, cut_extreme=True):
    assert isinstance(t, torch.Tensor), f"{type(t)}"
    assert t.ndim == 3
    d = t.device
    mean = torch.tensor(mean, device=d).unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor(std, device=d).unsqueeze(-1).unsqueeze(-1)
    tensor = std * t + mean
    tensor *= max_pixel_value
    if cut_extreme:
        tensor = torch.maximum(tensor, torch.tensor(0.0))
        tensor = torch.minimum(tensor, torch.tensor(max_pixel_value))
    return tensor


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (size[2] * size[3]))

    return bbx1, bby1, bbx2, bby2, lam


def rleDecode_old(mask_rle: str, shape: tuple) -> np.ndarray:

    """Decode a RLE mask into a numpy array
    # ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
    Args:
        mask_rle (string): The mask encoded in a string (RLE).
        shape (tuple): Height and width.

    Returns:
        (Numpy Array): 1 - mask, 0 - background
    """

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape((shape[1], shape[0])).T  # Needed to align to RLE direction


def rleDecode(mask_rle: str, shape: tuple) -> np.ndarray:
    """Decode a RLE mask into a numpy array
    Args:
        mask_rle (string): The mask encoded in a string (RLE).
        shape (tuple): Height and width.

    Returns:
        (Numpy Array): 1 - mask, 0 - background
    """

    s = np.fromstring(mask_rle, sep=' ', dtype='int')
    s[::2] -= 1
    ends = s[::2] + s[1::2]
    img = np.zeros((shape[0]*shape[1]), dtype=np.uint8)
    for i in range(len(ends)):
        img[s[2*i]:ends[i]] = 1

    return img.reshape((shape[1], shape[0])).T


def masksAsImage(in_mask_list: list, shape: tuple, class_mapping: Optional[dict] = None) -> np.ndarray:

    """Convert the list of RLE masks for each class into a numpy array
    Args:
        in_mask_list (list): The RLE data from the `.csv` as string converted with the 'eval()' method.
        shape (tuple): Height and width.
        class_mapping (facultative : dict): Dict whose keys correspond to the labels (background counts as one)

    Returns:
        (Numpy Array): single channel mask
    """

    mask = np.zeros((shape[0], shape[1]), dtype=np.float32)

    in_mask_list = eval(in_mask_list)

    if len(in_mask_list) == 0:
        return mask
    else:
        for tup in in_mask_list:
            m = rleDecode(tup[1], shape)
            if class_mapping is None:
                mask[m == 1] = tup[0]
            else:
                mask[m == 1] = class_mapping[tup[0]]

        return mask


def stratif_by_class(df: pd.DataFrame, stratif_args) -> pd.DataFrame:
    attribut = stratif_args["attribut"]
    classes = stratif_args["classes"]
    if "classes_frac" in stratif_args.keys():
        classes_frac = stratif_args["classes_frac"]
    else:
        classes_frac=None
    if "random_state" in stratif_args.keys():
        random_state = stratif_args["random_state"]
    else:
        random_state = None
    out = pd.DataFrame()
    for cl in classes:
        df_cl = df[df[attribut] == cl]
        if classes_frac:
            df_cl = df_cl.sample(frac=classes_frac[cl], random_state=random_state)
        out = out.append(df_cl)
    return out


def stratif_by_batch(df: pd.DataFrame, stratif_args: dict) -> pd.DataFrame:
    # print(stratif_args, flush=True)
    attribut = stratif_args["attribut"]
    classes = stratif_args["classes"]
    if "classes_frac" in stratif_args.keys():
        classes_frac = stratif_args["classes_frac"]
    else:
        classes_frac=None
    out = pd.DataFrame()
    ds_length = len(df)
    batch_length = 0
    for cl in classes:
        df_cl = df[df[attribut] == cl].sample(frac=1)
        if classes_frac:
            df_cl["batch"] = np.arange(len(df_cl)) / classes_frac[cl]
            df_cl_length = len(df_cl) / classes_frac[cl]
            batch_length += classes_frac[cl]
        else:
            df_cl["batch"] = np.arange(len(df_cl))
            df_cl_length = len(df_cl)
            batch_length += 1
        out = out.append(df_cl)
        ds_length = min(ds_length, df_cl_length)
    out = out.sort_values(["batch", attribut])[:int(ds_length * batch_length)].reset_index(drop=True)
    return out

def worker_init_fn(worker_id):
    print(f'*************** setting worker {worker_id} *********')
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
