import argparse
from pathlib import Path

from datasets import Dataset, DatasetInfo, DatasetDict
from PIL import Image
from tqdm import tqdm
import numpy as np
import json

parser = argparse.ArgumentParser()
parser.add_argument("target_dir")
parser.add_argument("--train_ratio", type=float, default=0.8)

CELEBAHQ_DICT = {
    0: "Background",
    1: "cloth",
    2: "skin",
    3: "hair",
    4: "neck",
    5: "neck_l",
    6: "mouth",
    7: "r_brow",
    8: "l_brow",
    9: "l_eye",
    10: "r_eye",
    11: "eye_g",
    12: "nose",
    13: "u_lip",
    14: "l_lip",
    15: "l_ear",
    16: "r_ear",
    17: "ear_r",
    18: "hat",
}


def combine_masks(image_path: str | Path, mask_root: str | Path):
    """
    各クラスの2値マスク画像を統合し、COCO形式のマスク画像を生成する。

    :param image_path: 元画像のパス (例: "A.png")
    :param mask_dir: マスク画像が保存されているディレクトリ
    :return: 統合されたマスク画像 (numpy array)
    """
    image_path = Path(image_path)
    name = image_path.stem
    mask_name = f"{int(name):05d}"
    mask_dir_name = f"{int(name) // 2000}"
    mask_dir = Path(mask_root).joinpath(mask_dir_name)
    coco_mask = np.zeros((512, 512), dtype=np.uint8)
    for class_id, class_name in CELEBAHQ_DICT.items():
        mask_path = mask_dir.joinpath(f"{mask_name}_{class_name}.png")
        if not mask_path.exists():
            continue
        mask = np.array(Image.open(mask_path).convert("L"))
        coco_mask[mask > 0] = class_id

    return coco_mask


def main(
    target_dir: Path | str,
    train_ratio: float,
    seed: int = 42,
):
    dataset_dict = {}
    target_dir = Path(target_dir)
    image_dir = target_dir.joinpath("CelebA-HQ-img")
    mask_dir = target_dir.joinpath("CelebAMask-HQ-mask-anno")
    assert image_dir.exists()
    assert mask_dir.exists()
    mask_save_dir = target_dir.joinpath("CelebAMask-HQ-mask-img")
    mask_save_dir.mkdir(exist_ok=True)
    hugging_face_dir = target_dir.joinpath("CelebAMask-HQ-huggingface")
    hugging_face_dir.mkdir(exist_ok=True)
    img_paths = sorted(image_dir.glob("*.jpg"))
    np.random.seed(seed)
    np.random.shuffle(img_paths)
    n_train = int(len(img_paths) * train_ratio)

    train_set = img_paths[:n_train]
    test_set = img_paths[n_train:]
    for set_name, set_names in zip(["train", "test"], [train_set, test_set]):
        data = {"image_path": [], "annotation_path": [], "image_id": []}
        for img_path in tqdm(set_names):
            mask = combine_masks(img_path, mask_dir)
            mask_path = mask_save_dir.joinpath(img_path.stem + ".png")
            Image.fromarray(mask).save(mask_path)
            data["image_path"].append(img_path.as_posix())
            data["annotation_path"].append(mask_path.as_posix())
            data["image_id"].append(img_path.stem)
        dataset_dict[set_name] = data

    citation = """
@inproceedings{
liu2015faceattributes,
title = {Deep Learning Face Attributes in the Wild},
author = {Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
month = {December},
year = {2015}
}
"""
    metadata = {
        "cls_count": 19,
        "cls_dict": CELEBAHQ_DICT,
        "content": "We splitted the original dataset with a 95%-5% ratio.",
    }
    info = DatasetInfo(
        dataset_name="Celeba-HQ Dataset Mask",
        description=json.dumps(metadata),
        citation=citation,
    )

    def transform(examples):
        examples["image"] = [Image.open(f) for f in examples["image_path"]]
        examples["annotation"] = [Image.open(f) for f in examples["annotation_path"]]
        return examples

    # We transform each dataset key in Dataset class...
    for split in list(dataset_dict.keys()):
        dataset = Dataset.from_dict(dataset_dict[split], info=info, split=split)
        dataset = dataset.map(
            transform, remove_columns=["image_path", "annotation_path"], batched=True
        )
        dataset_dict[split] = dataset

    final_dataset = DatasetDict(dataset_dict)
    final_dataset.save_to_disk(hugging_face_dir)


if __name__ == "__main__":
    main(**vars(parser.parse_args()))
