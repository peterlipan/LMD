import numpy as np


def virtual_representations(x, y, class_num, size=1000):
    """
    VFC: Virtual Feature Compensation
    :param x: feature matrix, size = N x C
    :param y: labels, size = N
    :param class_num: number of classes
    :param size: target number of generating virtual samples for each class
    :return:
    """
    assert len(set(y)) == class_num, \
        'Training set must include the samples from all the classes'
    virtual_samples = []
    virtual_labels = []
    mu_list = []
    sigma_list = []
    for i in range(class_num):
        class_samples = x[y == i]

        # calculate the mean and covariance of the
        # Gaussian distribution of the current class
        mean = np.mean(class_samples, axis=0)
        normed = class_samples - mean
        covariance = np.matmul(normed.T, normed) / (len(class_samples) - 1)

        # make the covariance positive-definite
        eps = 1e-2
        covariance += eps * np.eye(covariance.shape[0])

        assert not np.isnan(mean).any(), "mean contains NaN values"
        assert not np.isnan(covariance).any(), "covariance contains NaN values"

        mu_list.append(mean)
        sigma_list.append(covariance)

        gaussian_samples = np.random.multivariate_normal(mean, covariance, size)
        gaussian_labels = i * np.ones(size, dtype=np.int64)
        
        virtual_samples.extend(gaussian_samples)
        virtual_labels.extend(gaussian_labels)

    return np.array(virtual_samples, dtype=np.float32), np.array(virtual_labels), np.array(mu_list), np.array(sigma_list)
