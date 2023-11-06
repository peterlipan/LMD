import torch
import torch.nn as nn
import torch.nn.functional as F
from .gather import GatherLayer
from torch.distributions.multivariate_normal import MultivariateNormal, _batch_mahalanobis


def exp_mahalanobis_distance(sample, mean, covariance):
    """ Calculate the exp of the mahalanobis distance between the sample and class mean"""
    batch_shape = torch.broadcast_shapes(covariance.shape[:-2], mean.shape[:-1])
    sigma = covariance.expand(batch_shape + (-1, -1))
    _unbroadcasted_scale_tril = torch.linalg.cholesky(sigma)
    mu = mean.expand(batch_shape + (-1,))

    diff = sample - mu
    M = _batch_mahalanobis(_unbroadcasted_scale_tril, diff)

    return M


class MyMultivariateNormal(MultivariateNormal):
    """ Calculate the exp of the mahalanobis distance between the sample and class mean"""

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        diff = value - self.loc
        M = _batch_mahalanobis(self._unbroadcasted_scale_tril, diff)
        return torch.exp(-0.5 * M)


class ProbabilityLoss(nn.Module):
    def __init__(self):
        super(ProbabilityLoss, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.criterion = nn.KLDivLoss(reduction='sum')

    def forward(self, logits1, logits2):
        assert logits1.size() == logits2.size()
        softmax1 = self.softmax(logits1)
        softmax2 = self.softmax(logits2)

        probability_loss = self.criterion(softmax1.log(), softmax2)
        return probability_loss


class BatchLoss(nn.Module):
    def __init__(self, batch_size, world_size):
        super(BatchLoss, self).__init__()
        self.batch_size = batch_size
        self.world_size = world_size

    def forward(self, activations, ema_activations):
        assert activations.size() == ema_activations.size()
        N = self.batch_size * self.world_size
        # gather data from all the processes
        if self.world_size > 1:
            activations = torch.cat(GatherLayer.apply(activations), dim=0)
            ema_activations = torch.cat(GatherLayer.apply(ema_activations), dim=0)
        # reshape as N*C
        activations = activations.view(N, -1)
        ema_activations = ema_activations.view(N, -1)

        # form N*N similarity matrix
        similarity = activations.mm(activations.t())
        norm = torch.norm(similarity, 2, 1).view(-1, 1)
        similarity = similarity / norm

        ema_similarity = ema_activations.mm(ema_activations.t())
        ema_norm = torch.norm(ema_similarity, 2, 1).view(-1, 1)
        ema_similarity = ema_similarity / ema_norm

        batch_loss = (similarity - ema_similarity) ** 2 / N
        return batch_loss


class ChannelLoss(nn.Module):
    def __init__(self, batch_size, world_size):
        super(ChannelLoss, self).__init__()
        self.batch_size = batch_size
        self.world_size = world_size

    def forward(self, activations, ema_activations):
        assert activations.size() == ema_activations.size()
        N = self.batch_size * self.world_size
        # gather data from all the processes
        if self.world_size > 1:
            activations = torch.cat(GatherLayer.apply(activations), dim=0)
            ema_activations = torch.cat(GatherLayer.apply(ema_activations), dim=0)
        # reshape as N*C
        activations = activations.view(N, -1)
        ema_activations = ema_activations.view(N, -1)

        # form C*C channel-wise similarity matrix
        similarity = activations.t().mm(activations)
        norm = torch.norm(similarity, 2, 1).view(-1, 1)
        similarity = similarity / norm

        ema_similarity = ema_activations.t().mm(ema_activations)
        ema_norm = torch.norm(ema_similarity, 2, 1).view(-1, 1)
        ema_similarity = ema_similarity / ema_norm

        channel_loss = (similarity - ema_similarity) ** 2 / N
        return channel_loss


class GCELoss(nn.Module):
    def __init__(self, num_classes=10, q=0.7):
        super(GCELoss, self).__init__()
        self.q = q
        self.num_classes = num_classes

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return loss.mean()


class ClassConsistency(nn.Module):
    def __init__(self, mu, sigma, temperature=0.07):
        super(ClassConsistency, self).__init__()
        self.temperature = temperature
        self.mu = mu
        self.sigma = sigma

        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=-1)

    def forward(self, features, labels):
        """
        Args:
            features: hidden vector of shape [bsz, n_channels].
            labels: ground truth of shape [bsz].
        Returns:
            A loss scalar.
        """
        device = features.device

        # fetch the class mean for each sample
        # class_means: [N, C]
        class_means = torch.from_numpy(self.mu).to(device)[labels]

        # cosine similarity between features and class means
        sim = self.similarity_f(features, class_means) / self.temperature

        # mask for positive pairs
        mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float()
        # similarity over positive pairs
        sim_pos = torch.exp(sim) * mask
        # negative pairs
        sim_neg = torch.exp(sim) * (1 - mask)

        # contrastive loss
        loss = -torch.log(sim_pos.sum(dim=1) / sim_neg.sum(dim=1)).mean()

        return loss


class FeatureDistributionConsistency(nn.Module):
    def __init__(self, mu, sigma):
        super(FeatureDistributionConsistency, self).__init__()
        self.mu = mu
        self.sigma = sigma
        self.num_classes = self.mu.shape[0]

    def forward(self, features, labels):
        # the features should be of size B, C
        assert len(features.size()) == 2
        num_samples = features.size(0)
        device = features.device
        means, covs = torch.from_numpy(self.mu).to(device), torch.from_numpy(self.sigma).to(device)

        # attraction
        class_dists = exp_mahalanobis_distance(features, means[labels], covs[labels])
        attraction = torch.mean(class_dists)

        # repulsion
        # for each sample, select the classes that it does not belong to
        indices = [torch.where(labels[i] != torch.arange(self.num_classes).to(device))[0] for i in range(num_samples)]
        not_class_labels = torch.stack(indices).to(device)
        not_class_dists = exp_mahalanobis_distance(features[:, None, :], means[not_class_labels], covs[not_class_labels])
        repulsion = torch.mean(not_class_dists)

        return repulsion - attraction
