import os
import torch
import wandb
import argparse
import torch.distributed as dist
from models import CreateModel
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from data import ISICDataset, Transforms, KvasirDataset
from torch.utils.data import DataLoader
from utils import trainEncoder, yaml_config_hook
from utils.sync_batchnorm import convert_model
from prepare_datasets import construct_ISIC2019LT


def main(gpu, args, wandb_logger):
    if gpu != 0:
        wandb_logger = None

    rank = args.nr * args.gpus + gpu
    args.rank = rank
    args.device = rank

    if args.world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(rank)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # training set
    transforms = Transforms(size=args.image_size)
    if 'ISIC' in args.dataset:
        train_dataset = ISICDataset(args.data_path, args.csv_file_train, transform=transforms)
    elif 'kvasir' in args.dataset:
        train_dataset = KvasirDataset(args.data_path, args.csv_file_train, transform=transforms)

    # set sampler for parallel training
    if args.world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True
        )
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler,
    )
    if rank == 0:
        if 'ISIC' in args.dataset:
            test_dataset = ISICDataset(args.data_path, args.csv_file_test, transform=transforms.test_transform)
            val_dataset = ISICDataset(args.data_path, args.csv_file_val, transform=transforms.test_transform)
        elif 'kvasir' in args.dataset:
            test_dataset = KvasirDataset(args.data_path, args.csv_file_test, transform=transforms.test_transform)
            val_dataset = KvasirDataset(args.data_path, args.csv_file_val, transform=transforms.test_transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    else:
        test_loader = None
        val_loader = None

    loaders = (train_loader, val_loader, test_loader)

    num_class = train_dataset.n_class

    # model init
    model = CreateModel(backbone=args.backbone, ema=False, out_features=num_class, pretrained=args.pretrained)
    ema_model = CreateModel(backbone=args.backbone, ema=True, out_features=num_class, pretrained=args.pretrained)
    if args.reload:
        model_fp = os.path.join(
            args.checkpoints, "epoch_{}_.pth".format(args.epochs)
        )
        model.load_state_dict(torch.load(model_fp, map_location=args.device.type))

    model = model.to(args.device)
    ema_model = ema_model.to(args.device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9)

    if args.dataparallel:
        model = convert_model(model)
        model = DataParallel(model)
        ema_model = convert_model(ema_model)
        ema_model = DataLoader(ema_model)
    else:
        if args.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[gpu])

    trainEncoder(model, ema_model, loaders, optimizer, wandb_logger, args)


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    # Add argument for config file path
    parser.add_argument('--config', type=str, default='./config/isic2019.yaml', help='Path to the configuration file')
    parser.add_argument('--debug', action="store_true", help='debug mode (disable wandb)')
    args, _ = parser.parse_known_args()

    yaml_config = yaml_config_hook(args.config)
    for k, v in yaml_config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes

    # Master address for distributed data parallel
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    project = "MVE_%s" % args.dataset

    # if the dataset is 2019LT, construct a new dataset split
    # with imbalance factor=args.imbalance_factor
    if args.dataset == "ISIC2019LT":
        project = "MVE_%s_%d" % (args.dataset, args.imbalance_factor)
        print("Constructing ISIC2019LT Dataset with imbalance factor=%d" % args.imbalance_factor)
        construct_ISIC2019LT(imbalance_factor=args.imbalance_factor, data_root=args.data_path,
                             csv_file_root=os.path.dirname(args.csv_file_train), random_seed=args.seed)

    # check checkpoints path
    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)

    # init wandb
    if not args.debug:
        wandb.login(key=[YOUR_WANDB_KEY])
        config = dict()

        for k, v in yaml_config.items():
            config[k] = v

        wandb_logger = wandb.init(
            project=project,
            config=config
        )
    else:
        wandb_logger = None

    if args.world_size > 1:
        print(
            f"Training with {args.world_size} GPUS, waiting until all processes join before starting training"
        )
        mp.spawn(main, args=(args, wandb_logger,), nprocs=args.world_size, join=True)
    else:
        main(0, args, wandb_logger)
