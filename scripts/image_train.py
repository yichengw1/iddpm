"""
Train a diffusion model on images.
"""

import argparse

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop


def main():
    # 初始化 argparser
    args = create_argparser().parse_args()

    # 用于设置分布式处理组，用于并行计算
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")

    # 创建预测模型和diffusion框架
    model, diffusion = create_model_and_diffusion(
        # args 是一个很大的map，这里只传入和model and diffusion相关的键值作为入参
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # 类似于to(device)
    model.to(dist_util.dev())

    # 创建时间步t的采样器
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")

    # 创建 DataLoader
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    """
    从字典中自动生成argument parser
    """
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    # 合并字典defaults & model_and_diffusion_defaults
    defaults.update(model_and_diffusion_defaults())

    # init parser
    parser = argparse.ArgumentParser()

    # 高效方法创建parser，从字典中解析arg即可
    add_dict_to_argparser(parser, defaults)

    return parser


if __name__ == "__main__":
    main()
