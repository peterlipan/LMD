import torch
import numpy as np
from torch.utils.data import DataLoader


def inference(loader, backbone, device):
    feature_vector = []
    labels_vector = []
    backbone.eval()
    for step, (x, y) in enumerate(loader):
        x = x.to(device)

        # get encoding
        with torch.no_grad():
            activations, _ = backbone(x)

        activations = activations.detach()

        feature_vector.extend(activations.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    return feature_vector, labels_vector


def get_features(backbone, train_loader, test_loader, val_loader, device):
    train_X, train_y = inference(train_loader, backbone, device)
    test_X, test_y = inference(test_loader, backbone, device)
    val_X, val_y = inference(val_loader, backbone, device)
    return train_X, train_y, test_X, test_y, val_X, val_y


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, X_val, y_val, batch_size):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )

    val = torch.utils.data.TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(y_val)
    )
    val_loader = DataLoader(
        val, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader, val_loader