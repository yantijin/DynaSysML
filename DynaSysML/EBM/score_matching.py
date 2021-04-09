import torch
import torch.nn as nn
from tqdm import tqdm
import torch.autograd as autograd


'''
Note that exact score matching does not work for deep energy net,
not only because of computational complexity but also because of memory. 
'''
def score_matching(energy_net, samples):
    '''
    ||\nabla_x logp||_2^2 + \nabla^2_xlog p
    :param energy_net: a NN represents for energy of system
    :param samples: input for energy net
    :return:
    '''
    samples.requires_grad_(True)
    logp = -energy_net(samples).sum()
    grad1 = autograd.grad(logp, samples)[0]
    loss1 = (torch.norm(grad1, dim=-1) ** 2 / 2.).detach()

    loss2 = torch.zeros(samples.shape[0], device=samples.device)
    for i in tqdm(range(samples.shape[1])):
        logp = -energy_net(samples).sum()
        grad1 = autograd.grad(logp, samples, create_graph=True)[0]
        grad = autograd.grad(grad1[:, i].sum(), samples)[0][:, i]
        loss2 += grad.detach()

    loss = loss1 + loss2
    return loss.mean()


def exact_score_matching(energy_net, samples, train=False):
    samples.requires_grad_(True)
    logp = -energy_net(samples).sum()
    grad1 = autograd.grad(logp, samples, create_graph=True)[0]
    loss1 = torch.norm(grad1, dim=-1) ** 2 / 2.
    loss2 = torch.zeros(samples.shape[0], device=samples.device)

    iterator = range(samples.shape[1])

    for i in iterator:
        if train:
            grad = autograd.grad(grad1[:, i].sum(), samples, create_graph=True, retain_graph=True)[0][:, i]
        if not train:
            grad = autograd.grad(grad1[:, i].sum(), samples, create_graph=False, retain_graph=True)[0][:, i]
            grad = grad.detach()
        loss2 += grad

    loss = loss1 + loss2

    if not train:
        loss = loss.detach()

    return loss


# General implementations of SSM and SSM_VR ( variance reduction ) for arbitrary numbers of particles
def sliced_score_matching(energy_net, samples, n_particles=1):
    dup_samples = samples.unsqueeze(0).expand(n_particles, *samples.shape).contiguous().view(-1, *samples.shape[1:])
    dup_samples.requires_grad_(True)
    vectors = torch.randn_like(dup_samples)
    vectors = vectors / torch.norm(vectors, dim=-1, keepdim=True)

    logp = -energy_net(dup_samples).sum()
    grad1 = autograd.grad(logp, dup_samples, create_graph=True)[0]
    gradv = torch.sum(grad1 * vectors)
    loss1 = torch.sum(grad1 * vectors, dim=-1) ** 2 * 0.5
    grad2 = autograd.grad(gradv, dup_samples, create_graph=True)[0]
    loss2 = torch.sum(vectors * grad2, dim=-1)

    loss1 = loss1.view(n_particles, -1).mean(dim=0)
    loss2 = loss2.view(n_particles, -1).mean(dim=0)
    loss = loss1 + loss2
    return loss.mean(), loss1.mean(), loss2.mean()


def sliced_score_matching_vr(energy_net, samples, n_particles=1):
    dup_samples = samples.unsqueeze(0).expand(n_particles, *samples.shape).contiguous().view(-1, *samples.shape[1:])
    dup_samples.requires_grad_(True)
    vectors = torch.randn_like(dup_samples)

    logp = -energy_net(dup_samples).sum()
    grad1 = autograd.grad(logp, dup_samples, create_graph=True)[0]
    loss1 = torch.sum(grad1 * grad1, dim=-1) / 2.
    gradv = torch.sum(grad1 * vectors)
    grad2 = autograd.grad(gradv, dup_samples, create_graph=True)[0]
    loss2 = torch.sum(vectors * grad2, dim=-1)

    loss1 = loss1.view(n_particles, -1).mean(dim=0)
    loss2 = loss2.view(n_particles, -1).mean(dim=0)

    loss = loss1 + loss2
    return loss.mean(), loss1.mean(), loss2.mean()


def sliced_score_estimation(score_net, samples, n_particles=1):
    dup_samples = samples.unsqueeze(0).expand(n_particles, *samples.shape).contiguous().view(-1, *samples.shape[1:])
    dup_samples.requires_grad_(True)
    vectors = torch.randn_like(dup_samples)
    vectors = vectors / torch.norm(vectors, dim=-1, keepdim=True)

    grad1 = score_net(dup_samples)
    gradv = torch.sum(grad1 * vectors)
    loss1 = torch.sum(grad1 * vectors, dim=-1) ** 2 * 0.5
    grad2 = autograd.grad(gradv, dup_samples, create_graph=True)[0]
    loss2 = torch.sum(vectors * grad2, dim=-1)

    loss1 = loss1.view(n_particles, -1).mean(dim=0)
    loss2 = loss2.view(n_particles, -1).mean(dim=0)

    loss = loss1 + loss2
    return loss.mean(), loss1.mean(), loss2.mean()


def sliced_score_estimation_vr(score_net, samples, n_particles=1):
    dup_samples = samples.unsqueeze(0).expand(n_particles, *samples.shape).contiguous().view(-1, *samples.shape[1:])
    dup_samples.requires_grad_(True)
    vectors = torch.randn_like(dup_samples)

    grad1 = score_net(dup_samples)
    gradv = torch.sum(grad1 * vectors)
    loss1 = torch.sum(grad1 * grad1, dim=-1) / 2.
    grad2 = autograd.grad(gradv, dup_samples, create_graph=True)[0]
    loss2 = torch.sum(vectors * grad2, dim=-1)

    loss1 = loss1.view(n_particles, -1).mean(dim=0)
    loss2 = loss2.view(n_particles, -1).mean(dim=0)

    loss = loss1 + loss2
    return loss.mean(), loss1.mean(), loss2.mean()