"""
Train a diffusion model on images.
"""

import json
from datetime import datetime
from pathlib import Path

from config import cfg
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import create_model_and_diffusion
from guided_diffusion.train_util import TrainLoop


def main():
    exp_name = "-".join(
        [
            f"probe_mode_{cfg.DATASETS.PROBE_MODE}"
            f"b_map_min_{cfg.TRAIN.DIFFUSION.B_MAP_MIN}"
            f"img_size_{cfg.TRAIN.IMG_SIZE}"
            f"lr_{cfg.TRAIN.LR}"
            f"diffusion_steps_{cfg.TRAIN.DIFFUSION_STEPS}"
            f"b_map_sch_{cfg.TRAIN.B_MAP_SCHEDULER_TYPE}"
        ]
    )
    save_dir = Path(cfg.TRAIN.SAVE_DIR)
    assert save_dir.exists()
    save_dir = save_dir.joinpath(exp_name, datetime.now().strftime("%Y-%m-%d"))
    save_dir.mkdir(exist_ok=True)
    cfg.TRAIN.SAVE_DIR = save_dir.as_posix()
    cfg.DATASETS.SAVE_DIR = cfg.TRAIN.SAVE_DIR
    cfg.TRAIN.CHECKPOINT_DIR = cfg.TRAIN.SAVE_DIR

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(cfg)

    print("Moving model to CUDA (GPU on a single machine)...")
    model.to("cuda")

    print("Converting model to fp16...")
    model.convert_to_fp16()

    print("creating schedule sampler...")
    schedule_sampler = create_named_schedule_sampler(
        cfg.TRAIN.SCHEDULE_SAMPLER, diffusion
    )

    print("creating data loader...")
    data = load_data(cfg)

    jsonpath = Path(cfg.DATASETS.SAVE_DIR).joinpath("train_test_config.json")
    with open(jsonpath, "w") as f:
        json.dump(cfg, f, indent=4)

    print("training...")

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        num_classes=cfg.TRAIN.NUM_CLASSES,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        lr=cfg.TRAIN.LR,
        ema_rate=cfg.TRAIN.EMA_RATE,
        log_interval=cfg.TRAIN.LOG_INTERVAL,
        save_interval=cfg.TRAIN.SAVE_INTERVAL,
        resume_checkpoint=cfg.TRAIN.RESUME_CHECKPOINT,
        fp16_scale_growth=cfg.TRAIN.FP16_SCALE_GROWTH,
        schedule_sampler=schedule_sampler,
        weight_decay=cfg.TRAIN.WEIGHT_DECAY,
        lr_anneal_steps=cfg.TRAIN.LR_ANNEAL_STEPS,
        output_dir=cfg.TRAIN.CHECKPOINT_DIR,
    ).run_loop()


if __name__ == "__main__":
    main()
