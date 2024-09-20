import os
import wandb
import argparse
import torch
from models import CreateModel, Linear
from data import Transforms, ISICDataset, virtual_representations, KvasirDataset
from torch.utils.data import DataLoader
from utils import epochVal, classwise_evaluation, get_features, create_data_loaders_from_arrays, yaml_config_hook
from utils.loss import GCELoss, FeatureDistributionConsistency
from torch.nn.parallel import DataParallel
from torch.optim.lr_scheduler import MultiStepLR
from torchsampler import ImbalancedDatasetSampler
from prepare_datasets import construct_ISIC2019LT


def distribution_ema_update(means, sigmas, ema_means, ema_sigmas, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    # update the means
    for i in range(means.shape[0]):
        ema_means[i] = ema_means[i] * alpha + means[i] * (1 - alpha)

    # update the sigma
    for i in range(sigmas.shape[0]):
        ema_sigmas[i] = ema_sigmas[i] * alpha + sigmas[i] * (1 - alpha)


def e_step(backbone, classifier, opt, loader, label_supervision, feature_supervision, logger, weight):
    """
    Freeze the classifier and train the backbone,
    i.e., estimate the expected performance of classification.
    :return:
    """
    backbone.train()
    classifier.eval()
    for step, (x, y) in enumerate(loader):
        x, y = x.to(args.device), y.to(args.device)
        activations, _ = backbone(x)
        with torch.no_grad():
            out = classifier(activations)

        # label supervision
        label_loss = label_supervision(out, y)
        if feature_supervision is not None:
            # feature supervision
            feature_loss = feature_supervision(activations, y)
            loss = label_loss + weight * feature_loss if weight > 0 else label_loss

        else:
            loss = label_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        if logger is not None:
            logger.log({"E Step": {"loss": loss.item(),
                                   "feature loss": feature_loss.item(),
                                   "label loss": label_loss.item()}})


def m_step(classifier, opt, loader, loss_func, logger):
    """
    Freeze the backbone and train the classifier with virtual samples,
    i.e., maximize the expectation of classification
    :return:
    """
    epoch_loss = 0
    epoch_acc = 0
    classifier.train()
    for step, (x, y) in enumerate(loader):
        x, y = x.to(args.device), y.to(args.device)
        out = classifier(x)
        loss = loss_func(out, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        predict = out.argmax(1)
        acc = (predict == y).sum().item() / y.size(0)

        epoch_acc += acc
        epoch_loss += loss.item()
        if logger is not None:
            logger.log({"M Step loss": loss.item()})
    return epoch_loss, epoch_acc


def main(args, wandb_logger):
    transforms = Transforms(size=args.image_size)
    if 'ISIC' in args.dataset:
        train_dataset = ISICDataset(args.data_path, args.csv_file_train, transform=transforms.test_transform)
        test_dataset = ISICDataset(args.data_path, args.csv_file_test, transform=transforms.test_transform)
        val_dataset = ISICDataset(args.data_path, args.csv_file_val, transform=transforms.test_transform)
    elif 'kvasir' in args.dataset:
        train_dataset = KvasirDataset(args.data_path, args.csv_file_train, transform=transforms.test_transform)
        test_dataset = KvasirDataset(args.data_path, args.csv_file_test, transform=transforms.test_transform)
        val_dataset = KvasirDataset(args.data_path, args.csv_file_val, transform=transforms.test_transform)

    sampler = ImbalancedDatasetSampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        sampler=None
    )

    balanced_train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
        sampler=sampler
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    # load pre-trained the backbone from checkpoint
    n_classes = train_dataset.n_class
    backbone_model = CreateModel(backbone=args.backbone, out_features=n_classes)
    model_fp = os.path.join(args.checkpoints, "epoch_{}_.pth".format(args.epochs))
    checkpoint = torch.load(model_fp, map_location=args.device)
    backbone_model.load_state_dict(checkpoint)
    n_features = backbone_model.n_features
    backbone_model = DataParallel(backbone_model, device_ids=[int(x) for x in args.visible_gpus.split(',')]).cuda()
    backbone_optimizer = torch.optim.SGD(backbone_model.parameters(),
                                         lr=args.backbone_lr, momentum=0.9, weight_decay=1e-4)
    backbone_label_supervision = GCELoss(num_classes=n_classes)

    backbone_scheduler = MultiStepLR(optimizer=backbone_optimizer, milestones=args.stage2_steps,
                                     gamma=args.stage2_gamma)
    # smoothing for the means and covariance
    ema_mean = None
    ema_cov = None
    cur_itrs = 0

    # Classifier
    classifier_model = Linear(n_features, n_classes)
    classifier_model = DataParallel(classifier_model, device_ids=[int(x) for x in args.visible_gpus.split(',')]).cuda()
    classifier_optimizer = torch.optim.SGD(classifier_model.parameters(),
                                           lr=args.classifier_lr, momentum=0.9, weight_decay=1e-4)
    classifier_scheduler = MultiStepLR(optimizer=classifier_optimizer, milestones=args.stage2_steps,
                                       gamma=args.stage2_gamma)
    classifier_criterion = torch.nn.CrossEntropyLoss()

    # if employ balanced sampling in the E step
    if args.balanced_e:
        e_train_loader = balanced_train_loader
    else:
        e_train_loader = train_loader
    # if employ balanced sampling in the M step
    if args.balanced_m:
        m_train_loader = balanced_train_loader
    else:
        m_train_loader = train_loader

    for epoch in range(args.stage2_epochs):
        # extract features with the backbone
        train_X, train_y, test_X, test_y, val_X, val_y = get_features(
            backbone_model, m_train_loader, test_loader, val_loader, args.device
        )

        # virtual feature compensation
        if args.virtual_size > 0:
            train_X, train_y, mean, covariance = virtual_representations(train_X, train_y, n_classes, args.virtual_size)
            # pass mean and cov to ema for the initialization
            if ema_mean is None or ema_cov is None:
                ema_mean = mean
                ema_cov = covariance

            distribution_ema_update(mean, covariance, ema_mean, ema_cov, args.distribution_decay, cur_itrs)
            backbone_feature_supervision = FeatureDistributionConsistency(ema_mean, ema_cov)

        arr_train_loader, arr_test_loader, arr_val_loader = create_data_loaders_from_arrays(
            train_X, train_y, test_X, test_y, val_X, val_y, args.stage2_batch_size
        )

        # m-step
        # the first e-step is done at the stage1
        # so, we start with m-step
        loss_epoch, acc_epoch = \
            m_step(classifier_model, classifier_optimizer, arr_train_loader, classifier_criterion, wandb_logger)

        # e-step
        e_step(backbone_model, classifier_model, backbone_optimizer, e_train_loader,
               backbone_label_supervision, backbone_feature_supervision, wandb_logger, args.distribution_loss)

        cur_itrs += 1
        test_acc, test_f1, test_auc, test_bac, test_sens, test_spec, test_prec, test_mcc, test_kappa = epochVal(
            classifier_model, arr_test_loader)
        val_acc, val_f1, val_auc, val_bac, val_sens, val_spec, val_prec, val_mcc, val_kappa = epochVal(classifier_model,
                                                                                                        arr_val_loader)
        backbone_lr = backbone_optimizer.param_groups[0]['lr']
        classifier_lr = classifier_optimizer.param_groups[0]['lr']
        if not args.debug:
            wandb_logger.log({'test': {'Accuracy': test_acc,
                                        'F1 score': test_f1,
                                        'AUC': test_auc,
                                        'Balanced Accuracy': test_bac,
                                        'Sensitivity': test_sens,
                                        'Specificity': test_spec,
                                        'Precision': test_prec,
                                        'MCC': test_mcc,
                                        'Kappa': test_kappa},
                                'validation': {'Accuracy': val_acc,
                                                'F1 score': val_f1,
                                                'AUC': val_auc,
                                                'Balanced Accuracy': val_bac,
                                                'Sensitivity': val_sens,
                                                'Specificity': val_spec,
                                                'Precision': val_prec,
                                                'MCC': val_mcc,
                                                'Kappa': val_kappa},
                                'learning rate': {'backbone': backbone_lr,
                                                'classifier': classifier_lr}})
        print(
            f"Epoch [{epoch}/{args.stage2_epochs}]\t Loss: {loss_epoch / len(arr_train_loader)}\t Accuracy: {acc_epoch / len(arr_train_loader)}"
        )

    # final test, retrieve the perclass performance
    saveModelPath = os.path.join(args.checkpoints, 'best_stage2_{:s}.pth'.format(args.dataset))
    state_dict = backbone_model.module.state_dict()
    torch.save(state_dict, saveModelPath)
    df = classwise_evaluation(classifier_model, arr_test_loader)
    if not args.debug:
        wandb_df = wandb.Table(dataframe=df)
        wandb_logger.log({"perclass performance": wandb_df})
    else:
        print(df)


if __name__ == "__main__":
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

    project = "MVE_Stage2_%s" % args.dataset

    if args.dataset == "ISIC2019LT":
        project = "MVE_Stage2_%s_%d" % (args.dataset, args.imbalance_factor)
        print("Constructing ISIC2019LT Dataset with imbalance factor=%d" % args.imbalance_factor)
        construct_ISIC2019LT(imbalance_factor=args.imbalance_factor, data_root=args.data_path,
                                csv_file_root=os.path.dirname(args.csv_file_train), random_seed=args.seed)

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

    main(args, wandb_logger)
