import random
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def load_data(cfg):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    """

    if not cfg.DATASETS.DATADIR:
        raise ValueError("unspecified data directory")

    dataset_dir = Path(cfg.DATASETS.DATADIR)
    subdir = "training" if cfg.TRAIN.IS_TRAIN else "validation"
    image_dir = dataset_dir.joinpath("images", subdir)
    image_pathes = sorted([p for p in image_dir.glob("*.png")])
    label_dir = dataset_dir.joinpath("sector_annotations", subdir)
    label_pathes = sorted([p for p in label_dir.glob("*.png")])

    dataset = ImageDataset(
        cfg.DATASETS.DATASET_MODE,
        cfg.TRAIN.IMG_SIZE,
        image_pathes,
        classes=label_pathes,
        random_crop=cfg.TRAIN.RANDOM_CROP,
        random_flip=cfg.TRAIN.RANDOM_FLIP,
        is_train=cfg.TRAIN.IS_TRAIN,
    )

    if cfg.TRAIN.IS_TRAIN:
        batch_size = cfg.TRAIN.BATCH_SIZE
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=cfg.TRAIN.NUM_WORKERS,
            drop_last=True,
        )
    else:
        batch_size = cfg.TEST.BATCH_SIZE
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=cfg.TRAIN.NUM_WORKERS,
            drop_last=True,
        )

    # ジェネレータを返却することで無限ループさせる
    while True:
        yield from loader


def random_crop_and_resize_image_label(
    image: Image.Image, label: Image.Image, resize_to=None
):
    """
    Randomly crops the image and label to 80% or 90% of their original size from sides or bottom,
    keeping the top part aligned, and then resizes them back to the original size or to a specified size.

    Args:
    - image (PIL.Image): The input image to augment.
    - label (PIL.Image): The corresponding label image to augment.
    - resize_to (tuple, optional): The size to which the image and label should be resized. If None, uses the original size.

    Returns:
    - PIL.Image: The augmented and resized image.
    - PIL.Image: The augmented and resized label with nearest neighbor interpolation.
    """
    original_width, original_height = image.size
    resize_to = resize_to if resize_to else (original_width, original_height)

    # Choose a random crop size: 80% or 90% of the original dimensions
    crop_size = random.uniform(0.8, 0.95)
    new_height = int(original_height * crop_size)

    # Randomly choose the bottom crop boundary if cropping is from the bottom
    bottom_crop = original_height - new_height

    # Randomly choose how much to crop from the left (the rest will be cropped from the right)
    crop_width = int(original_width * crop_size)
    left_crop = random.randint(0, original_width - crop_width)

    # Define the crop box
    crop_box = (left_crop, 0, left_crop + crop_width, new_height)

    # Crop and resize the image
    cropped_image = image.crop(crop_box)
    # resized_image = cropped_image.resize(resize_to, Image.ANTIALIAS)
    # ANTIALIAS is no longer available in PIL: module 'PIL.Image' has no attribute 'ANTIALIAS'
    resized_image = cropped_image.resize(resize_to, Image.Resampling.LANCZOS)
    # This change adapts the code to be compatible with Pillow 7.0.0 and later.
    # The LANCZOS filter is an excellent choice for resizing when quality is a priority,
    # as it typically provides better results for image downscaling.

    # Crop and resize the label with nearest neighbor interpolation
    cropped_label = label.crop(crop_box)
    resized_label = cropped_label.resize(resize_to, Image.Resampling.NEAREST)

    return resized_image, resized_label


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        instances=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
        is_train=True,
    ):
        super().__init__()
        self.is_train = is_train
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.local_instances = (
            None if instances is None else instances[shard:][::num_shards]
        )
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with open(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        out_dict = {}
        class_path = self.local_classes[idx]
        with open(class_path, "rb") as f:
            pil_class = Image.open(f)
            pil_class.load()
        pil_class = pil_class.convert("L")

        if self.local_instances is not None:
            instance_path = self.local_instances[
                idx
            ]  # DEBUG: from classes to instances, may affect CelebA
            with open(instance_path, "rb") as f:
                pil_instance = Image.open(f)
                pil_instance.load()
            pil_instance = pil_instance.convert("L")
        else:
            pil_instance = None

        arr_image, arr_class, arr_instance = resize_arr(
            [pil_image, pil_class, pil_instance], self.resolution, keep_aspect=False
        )

        if self.random_flip and random.random() < 0.5:
            arr_image = arr_image[:, ::-1].copy()
            arr_class = arr_class[:, ::-1].copy()
            arr_instance = (
                arr_instance[:, ::-1].copy() if arr_instance is not None else None
            )

        arr_image = arr_image.astype(np.float32) / 127.5 - 1

        out_dict["path"] = path
        out_dict["label_ori"] = arr_class.copy()

        out_dict["label"] = arr_class[None,]

        if arr_instance is not None:
            out_dict["instance"] = arr_instance[None,]

        return np.transpose(arr_image, [2, 0, 1]), out_dict


def resize_arr(pil_list, image_size, keep_aspect=True):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    pil_image, pil_class, pil_instance = pil_list

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    if keep_aspect:
        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )
    else:
        pil_image = pil_image.resize((image_size, image_size), resample=Image.BICUBIC)

    pil_class = pil_class.resize(pil_image.size, resample=Image.NEAREST)
    if pil_instance is not None:
        pil_instance = pil_instance.resize(pil_image.size, resample=Image.NEAREST)

    arr_image = np.array(pil_image)
    arr_class = np.array(pil_class)
    arr_instance = np.array(pil_instance) if pil_instance is not None else None
    return arr_image, arr_class, arr_instance
