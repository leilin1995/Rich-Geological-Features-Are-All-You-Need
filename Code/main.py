"""
__author__: Lei Lin
__project__: main.py
__time__: 2024/5/9 
__email__: leilin1117@outlook.com
"""
import argparse
import os
from functools import partial
import json

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from trainer import run_training
from models import UNet3D
from dataset import StructureDataset, Sampler
# from losses import *
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.utils.enums import MetricReduction

parser = argparse.ArgumentParser(description="Unet3D for fault segmentation")
parser.add_argument("--logdir", default="Unet3D", type=str, help="directory to save the files in training")
parser.add_argument("--save_checkpoint", default=True, type=bool, help="save checkpoint during training")
parser.add_argument("--max_epochs", default=100, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=4, type=int, help="number of batch size")
parser.add_argument("--optim_lr", default=1e-5, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adam", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--val_every", default=1, type=int, help="validation frequency")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=1, type=int, help="number of output channels")
parser.add_argument("--final_sigmoid", default=True, type=bool,
                    help="apply element-wise nn.Sigmoid after the final 1x1 convolution, otherwise apply nn.Softmax (in this case softmax)")
parser.add_argument("--layer_order", default="cbrd", type=str,
                    help="determines the order of operators in a single layer (cbrd - Conv3D+Batchnorm+ReLU+Dropout)")
parser.add_argument("--training_dir", default="../Dataset/Train", type=str,
                    help="Training dataset directory")
parser.add_argument("--validation_dir", default="../Dataset/Val", type=str,
                    help="Validation dataset directory")
parser.add_argument("--raw_internal_path", default="seismic", type=str,
                    help="Internal path of .hdf5 of input seismic data")
parser.add_argument("--label_internal_path", default="label", type=str,
                    help="Internal path of .hdf5 of target label data")
parser.add_argument("--data_norm", default="Normalize", type=str,
                    help="Data standardization methods. null, Normalize, Standard")
parser.add_argument("--norm01", default=True, type=bool,
                    help="When data_norm == Normalize,normalize data to 0-1 or -1-1")
parser.add_argument("--random_crop", default=False, type=bool,
                    help="Using a data augmentation strategy with randomized cropping ensures that NN see different data each time when training")
parser.add_argument("--patch_size", default=[64, 64, 64], type=list,
                    help="Size of data after cropping")
parser.add_argument("--transform", default=True, type=bool,
                    help="Using flip and rotate or not")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str,
                    help="use scheduler")
parser.add_argument("--warmup_epochs", default=15, type=int, help="number of warmup epochs")
parser.add_argument("--resume_ckpt", default=False, type=bool, help="resume training from pretrained checkpoint")
parser.add_argument(
    "--pretrained_pth",
    default="../Results/Unet3D/model.pt",
    type=str,
    help="pretrained checkpoint directory",
)
parser.add_argument("--squared_dice", action="store_true", help="use squared Dice")


def main():
    args = parser.parse_args()
    args.amp = not args.noamp
    args.logdir = "../Results/" + args.logdir
    if os.path.exists(args.logdir) is False:
        os.makedirs(args.logdir)

    # 将参数保存到 JSON 文件中
    args_dict = vars(args)  # 将 Namespace 转换为字典
    with open(os.path.join(args.logdir, 'config.json'), 'w') as f:
        json.dump(args_dict, f, indent=4)

    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)


def main_worker(gpu, args):
    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True

    train_dataset = StructureDataset(file_dir=args.training_dir, phase="train",
                                     raw_internal_path=args.raw_internal_path,
                                     label_internal_path=args.label_internal_path, patch_size=args.patch_size,
                                     data_norm=args.data_norm, random_crop=args.random_crop,
                                     transform=args.transform, norm01=args.norm01)
    train_sampler = Sampler(train_dataset) if args.distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        sampler=train_sampler,
        pin_memory=True
    )
    val_dataset = StructureDataset(file_dir=args.validation_dir, phase="val",
                                   raw_internal_path=args.raw_internal_path,
                                   label_internal_path=args.label_internal_path, patch_size=args.patch_size,
                                   data_norm=args.data_norm, random_crop=args.random_crop,
                                   transform=args.transform, norm01=args.norm01)
    val_sampler = Sampler(val_dataset, shuffle=False) if args.distributed else None
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
    )

    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)
    model = UNet3D(in_channels=args.in_channels,
                   out_channels=args.out_channels,
                   final_sigmoid=args.final_sigmoid,
                   layer_order=args.layer_order)

    if args.resume_ckpt:
        model_dict = torch.load(args.pretrained_pth)["state_dict"]
        model.load_state_dict(model_dict)
        print("Using pretrained weights")

    if args.out_channels == 1:
        include_background = False
    else:
        include_background = True
    if args.squared_dice:
        dice_loss = DiceLoss(
            to_onehot_y=False, include_background=include_background, squared_pred=True, smooth_nr=args.smooth_nr,
            smooth_dr=args.smooth_dr
        )
    else:
        dice_loss = DiceLoss(to_onehot_y=False, include_background=include_background)

    if args.out_channels > 1:
        post_pred = AsDiscrete(argmax=True, to_onehot=2)
    else:
        post_pred = AsDiscrete(threshold=0.5, to_onehot=2)
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    start_epoch = 0
    model.cuda(args.gpu)
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if "b" in args.layer_order:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    elif args.lrschedule == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    else:
        scheduler = None

    accuracy = run_training(model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer,
                            loss_func=dice_loss, acc_func=dice_acc, args=args,
                            scheduler=scheduler,
                            start_epoch=start_epoch,
                            post_pred=post_pred)
    return accuracy


if __name__ == "__main__":
    main()
