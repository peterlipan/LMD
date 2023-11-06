import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, \
    confusion_matrix, roc_auc_score, precision_score, matthews_corrcoef, cohen_kappa_score, classification_report
from imblearn.metrics import sensitivity_score, specificity_score


def compute_avg_metrics(groundTruth, activations):
    groundTruth = groundTruth.cpu().detach().numpy()
    activations = activations.cpu().detach().numpy()
    predictions = np.argmax(activations, -1)
    mean_acc = accuracy_score(y_true=groundTruth, y_pred=predictions)
    f1_macro = f1_score(y_true=groundTruth, y_pred=predictions, average='macro')
    try:
        auc = roc_auc_score(y_true=groundTruth, y_score=activations, multi_class='ovr')
    except ValueError as error:
        print('Error in computing AUC. Error msg:{}'.format(error))
        auc = 0
    bac = balanced_accuracy_score(y_true=groundTruth, y_pred=predictions)
    sens_macro = sensitivity_score(y_true=groundTruth, y_pred=predictions, average='macro')
    spec_macro = specificity_score(y_true=groundTruth, y_pred=predictions, average='macro')
    prec_macro = precision_score(y_true=groundTruth, y_pred=predictions, average="macro")
    mcc = matthews_corrcoef(y_true=groundTruth, y_pred=predictions)
    kappa = cohen_kappa_score(y1=groundTruth, y2=predictions, weights='quadratic')

    return mean_acc, f1_macro, auc, bac, sens_macro, spec_macro, prec_macro, mcc, kappa


def compute_perclass_performance(groundTruth, activations):

    groundTruth = groundTruth.cpu().detach().numpy()
    activations = activations.cpu().detach().numpy()
    predictions = np.argmax(activations, -1)

    cls_dict = classification_report(y_true=groundTruth, y_pred=predictions, output_dict=True)
    df = pd.DataFrame(cls_dict).transpose().iloc[:len(set(groundTruth)), :]

    return df


def epochVal(model, dataLoader):
    training = model.training
    model.eval()

    groundTruth = torch.Tensor().cuda()
    activations = torch.Tensor().cuda()

    with torch.no_grad():
        for i, (image, label) in enumerate(dataLoader):
            image, label = image.cuda(), label.cuda()
            output = model(image)
            if isinstance(output, tuple):
                _, output = output
            output = F.softmax(output, dim=1)
            groundTruth = torch.cat((groundTruth, label))
            activations = torch.cat((activations, output))

    acc, f1, auc, bac, sens, spec, prec, mcc, kappa = compute_avg_metrics(groundTruth, activations)

    model.train(training)

    return acc, f1, auc, bac, sens, spec, prec, mcc, kappa


def classwise_evaluation(model, dataLoader):
    training = model.training
    model.eval()

    groundTruth = torch.Tensor().cuda()
    activations = torch.Tensor().cuda()

    with torch.no_grad():
        for i, (image, label) in enumerate(dataLoader):
            image, label = image.cuda(), label.cuda()
            output = model(image)
            if isinstance(output, tuple):
                _, output = output
            output = F.softmax(output, dim=1)
            groundTruth = torch.cat((groundTruth, label))
            activations = torch.cat((activations, output))

    df = compute_perclass_performance(groundTruth, activations)
    model.train(training)

    return df
