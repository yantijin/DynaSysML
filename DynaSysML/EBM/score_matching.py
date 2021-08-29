import torch
import torch.nn as nn
from tqdm import tqdm
import torch.autograd as autograd


'''
Note that exact score matching does not work for deep energy net,
not only because of computational complexity but also because of memory. 
sliced score matching https://arxiv.org/abs/1905.07088
score matching with langevin dynamics https://arxiv.org/abs/1907.05600
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
    '''
    sclied score matching
    s = \nabla_x log p_x
    v^T\nabla_x s = \nabla_x (v^T s)
    loss1 = 0.5 *(v^T s)^2
    loss2 = v^T\nabla_x s v
    :param energy_net: log p(x) = - energy_net(x)
    :param samples: x
    :param n_particles:
    :return:
    '''
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
    '''
    score net  = \nabla_x loq p(x)
    :param score_net:
    :param samples:
    :param n_particles:
    :return:
    '''
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



def dsm(energy_net, samples, sigma=1):
    '''
    denoising score matching
    \tilde x = x + z * sigma
    loss = (sigma^2 * s(\tilde x) + z*sigma)^2
    :param energy_net: log p(x) = - energy_net(x)
    :param samples: input data
    :param sigma: the standard variance
    :return:
    '''
    samples.requires_grad_(True)
    vector = torch.randn_like(samples) * sigma
    perturbed_inputs = samples + vector
    logp = -energy_net(perturbed_inputs)
    dlogp = sigma ** 2 * autograd.grad(logp.sum(), perturbed_inputs, create_graph=True)[0]
    kernel = vector
    loss = torch.norm(dlogp + kernel, dim=-1) ** 2
    loss = loss.mean() / 2.

    return loss


def dsm_score_estimation(scorenet, samples, sigma=0.01):
    '''

    :param scorenet: \nabbla_x \log p(x)
    :param samples:
    :param sigma:
    :return:
    '''
    perturbed_samples = samples + torch.randn_like(samples) * sigma
    target = - 1 / (sigma ** 2) * (perturbed_samples - samples)
    scores = scorenet(perturbed_samples)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1).mean(dim=0)

    return loss


def anneal_dsm_score_estimation(scorenet, samples, labels, sigmas, anneal_power=2.):
    '''
    相比dsm_score_estimation加入了权重lambda(sigma) = sigma^2
    :param scorenet:
    :param samples:
    :param labels: 用于score net的输入，ｌａｂｅｌｓ要和selected sigmas一一对应
    :param sigmas: np.array()
    :param anneal_power:
    :return:
    '''
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:]))) # select used sigmas according to labels
    perturbed_samples = samples + torch.randn_like(samples) * used_sigmas
    target = - 1 / (used_sigmas ** 2) * (perturbed_samples - samples)
    scores = scorenet(perturbed_samples, labels)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power

    return loss.mean(dim=0)