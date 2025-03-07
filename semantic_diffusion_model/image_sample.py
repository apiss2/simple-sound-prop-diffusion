"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision as tv
from config import cfg
from guided_diffusion.image_datasets import load_data
from guided_diffusion.script_util import (
    create_model_and_diffusion,
)
from PIL import Image
from skimage.color import label2rgb
from skimage.feature import canny


def main():
    print("creating model and diffusion...")

    torch.cuda.empty_cache()

    model, diffusion = create_model_and_diffusion(cfg)

    model_state = torch.load(cfg.TRAIN.RESUME_CHECKPOINT, map_location="cpu")

    model.load_state_dict(model_state)

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.cuda.current_device()
    else:
        device = torch.device("cpu")
    model.to(device)

    print("creating data loader...")
    data = load_data(cfg)
    model.convert_to_fp16()

    model.eval()

    results_dir = Path(cfg.TEST.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 各サブディレクトリを作成
    image_path = results_dir / "images"
    image_path.mkdir(exist_ok=True)
    label_path = results_dir / "labels"
    label_path.mkdir(exist_ok=True)
    visible_label_path = results_dir / "labels_visible"
    visible_label_path.mkdir(exist_ok=True)
    inference_path = results_dir / "samples"
    inference_path.mkdir(exist_ok=True)
    combined_path = results_dir / "combined"
    combined_path.mkdir(exist_ok=True)

    print("sampling...")
    all_samples = []
    for i, (batch, cond) in enumerate(data):
        src_img = ((batch + 1.0) / 2.0).to(device)
        label_img = cond["label_ori"].float()
        model_kwargs = preprocess_input(cond, num_classes=cfg.TRAIN.NUM_CLASSES)

        # set hyperparameter
        model_kwargs["s"] = cfg.TEST.S

        sample_fn = diffusion.p_sample_loop_with_snapshot
        inference_img, snapshots = sample_fn(
            model,
            (cfg.TEST.BATCH_SIZE, 3, src_img.shape[2], src_img.shape[3]),
            clip_denoised=cfg.TEST.CLIP_DENOISED,
            model_kwargs=model_kwargs,
            progress=True,
        )

        inference_img = (inference_img + 1) / 2.0

        gathered_samples = [torch.zeros_like(inference_img)]
        all_samples.extend([sample.cpu().numpy() for sample in gathered_samples])

        for j in range(inference_img.shape[0]):
            name = Path(image_path) / (Path(cond["path"][j]).stem + ".png")
            tv.utils.save_image(src_img[j], Path(image_path) / name)
            tv.utils.save_image(inference_img[j], Path(inference_path) / name)
            _label_img = label_img[j] / cfg.TRAIN.NUM_CLASSES
            tv.utils.save_image(_label_img, Path(visible_label_path) / name)
            _label_img = label_img[j].cpu().detach().numpy()
            label_save_img = Image.fromarray(_label_img).convert("RGB")
            label_save_img.save(_label_img, Path(label_path) / name)

            src_img_np = src_img[j].permute(1, 2, 0).detach().cpu().numpy()
            label_img_np = (
                label_img[j].repeat(3, 1, 1).permute(1, 2, 0).detach().cpu().numpy()
            )
            inference_img_np = inference_img[j].permute(1, 2, 0).detach().cpu().numpy()
            inference_img_np = (inference_img_np - np.min(inference_img_np)) / np.ptp(
                inference_img_np
            )
            inference_img_np = (
                255
                * (inference_img_np - np.min(inference_img_np))
                / np.ptp(inference_img_np)
            ).astype(int)

            combined_imgs = generate_combined_imgs(
                src_img_np, label_img_np.astype(np.int_), inference_img_np
            )

            im = Image.fromarray(combined_imgs)
            im.save(Path(combined_path) / (Path(cond["path"][j]).stem + ".png"))

        print(f"created {len(all_samples) * cfg.TEST.BATCH_SIZE} samples")

        log_images(inference_img, label_img, src_img, snapshots=snapshots)

        if len(all_samples) * cfg.TEST.BATCH_SIZE > cfg.TEST.NUM_SAMPLES:
            break

    print("sampling complete")


def log_images(inference_img, label_img, src_img, snapshots=None):
    if snapshots is None:
        snapshots = {}
    num_rows = 3 + len(snapshots)
    num_cols = inference_img.shape[0]
    # Base size for each subplot + some padding
    base_width = 4
    base_height = 4
    fig_width = num_cols * base_width + 2
    fig_height = num_rows * base_height

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
    fig.suptitle("Diffusion Model Results", fontsize=16)

    for k in range(num_cols):
        axs[0, k].imshow(src_img[k, 0, ...].cpu().detach().numpy(), cmap="gray")
        axs[0, k].axis("off")

        axs[1, k].imshow(inference_img[k, 0, ...].cpu().detach().numpy(), cmap="gray")
        axs[1, k].axis("off")

        axs[2, k].imshow(label_img[k, ...].cpu().detach().numpy(), cmap="gray")
        axs[2, k].axis("off")

        for i, snap in enumerate(snapshots):
            axs[i + 3, k].imshow(
                snapshots[snap][k, 0, ...].cpu().detach().numpy(), cmap="gray"
            )
            axs[i + 3, k].axis("off")

    # Set vertical labels for each row outside the loop
    axs[0, 0].set_title("Source Image")
    axs[1, 0].set_title("Inference Image")
    axs[2, 0].set_title("Label Image", pad=20)
    for i, snap in enumerate(snapshots):
        axs[i + 3, 0].set_title(f"Snapshot {snap}")

    plt.tight_layout(
        rect=[0, 0.0, 1, 0.95]
    )  # Adjust the layout to leave space for the suptitle
    result_dir = Path(cfg.TEST.RESULTS_DIR)
    n_sample = len([p for p in result_dir.glob("*")])
    plt.savefig(result_dir / f"sample_{n_sample}.png")
    plt.close()


def og_generate_combined_imgs(src_in_img, label_in_img, inference_in_img):
    overlayed_label = label2rgb(
        label=label_in_img[:, :, 0],
        image=inference_in_img,
        bg_label=0,
        channel_axis=-1,
        alpha=0.2,
        image_alpha=1,
    )

    src_out_img = (src_in_img * 255).astype("uint8")
    overlayed_label = (overlayed_label * 255).astype("uint8")

    edges = canny(label_in_img[:, :, 0] / label_in_img[:, :, 0].max())
    edges = np.expand_dims(edges, axis=-1)
    edges = np.concatenate((edges, edges, edges), axis=-1) * 255
    edges[:, :, 2] = 0

    overlayed_edge_label = np.copy(inference_in_img)
    overlayed_edge_label[edges == 255] = 255

    combined_imgs = np.concatenate(
        (src_out_img, inference_in_img, overlayed_label, overlayed_edge_label), axis=0
    ).astype(np.uint8)

    return combined_imgs


def generate_combined_imgs(src_in_img, label_in_img, inference_in_img):
    # label in img is already 3 channels
    label_rgb_img = label_in_img

    # Convert source to uint8
    src_out_img = (src_in_img * 255).astype("uint8")

    # Normalize label_rgb_img to [0, 255]
    label_rgb_img = (label_rgb_img / label_rgb_img.max() * 255).astype("uint8")

    # Blend the label image with the inference image
    # alpha = 0.2
    # blended_img = (alpha * label_rgb_img + (1 - alpha) * inference_in_img).astype(
    #     "uint8"
    # )

    # Perform edge detection on the label image
    edges = canny(label_in_img[:, :, 0] / label_in_img[:, :, 0].max())
    edges = np.expand_dims(edges, axis=-1)
    edges = np.concatenate((edges, edges, edges), axis=-1) * 255
    edges[:, :, 2] = 0

    # Overlay edges on the inference image
    overlayed_edge_label = np.copy(inference_in_img)
    overlayed_edge_label[edges == 255] = 255

    # Combine images vertically
    combined_imgs = np.concatenate(
        (src_out_img, inference_in_img, label_rgb_img, overlayed_edge_label), axis=0
    ).astype(np.uint8)

    return combined_imgs


def preprocess_input(data, num_classes):
    # move to GPU and change data types
    data["label"] = data["label"].long()

    # create one-hot label map
    label_map = data["label"]
    bs, _, h, w = label_map.size()
    input_label = torch.FloatTensor(bs, num_classes, h, w).zero_()
    input_semantics = input_label.scatter_(1, label_map, 1.0)

    # concatenate instance map if it exists
    if "instance" in data:
        inst_map = data["instance"]
        instance_edge_map = get_edges(inst_map)
        input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)

    return {"y": input_semantics}


def get_edges(t):
    edge = torch.ByteTensor(t.size()).zero_()
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    return edge.float()


if __name__ == "__main__":
    main()
