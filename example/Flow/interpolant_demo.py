import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from DynaSysML.Flow.Interpolant import *
from DynaSysML.Flow.inter_utils import make_fc_net
from tqdm import tqdm

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# define base and target distributions
def target(bs):
    x1 = torch.rand(bs) * 4 - 2 # range: [-2, 2]
    x2_ = torch.rand(bs) - torch.randint(2, (bs,)) * 2 # (0, 1) - [0, 1, 2] -> range: [-2, 1]
    x2 = x2_ + (torch.floor(x1) % 2) # [-2, 2]
    return (torch.cat([x1[:, None], x2[:, None]], 1) * 2)

class SimpleNormal(nn.Module):
    def __init__(self, loc, var, requires_grad=False):
        super(SimpleNormal, self).__init__()
        if requires_grad:
            loc.requires_grad_()
            var.requires_grad_()
        self.loc = loc
        self.var = var
        self.dist = torch.distributions.normal.Normal(torch.flatten(self.loc), torch.flatten(self.var))

    def log_prob(self, x):
        logp = self.dist.log_prob(x.reshape(x.shape[0], -1))
        return torch.sum(logp, dim=-1)

    def forward(self, bs):
        x = self.dist.sample((bs, ))
        return torch.reshape(x, (-1, ) + self.loc.shape)

    def rsample(self, bs):
        x = self.dist.rsample((bs, ))
        return torch.reshape(x, (-1,) + self.loc.shape)
base = SimpleNormal(torch.zeros(2).to(device), torch.ones(2).to(device))

# compute likelihood
def compute_likelihood(
        v: nn.Module,
        s: nn.Module,
        interpolant: Interpolant,
        save_num: int,
        sample_num: int,
        eps: int,
        bs: int
):
    '''draw samples from the probability flow and SDE models, and compute likelihoods'''
    probability_flow = pfIntergrater(v, s, interpolant, n_step=5)
    sde_flow = SDEFlow(v, s, interpolant, dt=torch.tensor(1e-2), eps=eps)

    with torch.no_grad():
        x0_tests = base(bs).to(device)
        xfs_sde = sde_flow.rollout_forward(x0_tests, save_num=save_num) # [save_num, bs, dim]
        xf_sde = xfs_sde[-1].squeeze().detach().cpu().numpy()

        x0s_sdeflow, dlogps_sdeflow = sde_flow.rollout_likelihood(xfs_sde[-1], sample_num=sample_num)
        log_p0s = torch.reshape(
            base.log_prob(x0s_sdeflow.reshape(sample_num*bs, 2)),
            (sample_num, bs)
        )
        logpx_sdeflow = torch.mean(log_p0s, dim=0) - dlogps_sdeflow

    # probability flows
    logp0 = base.log_prob(x0_tests)
    xfs_pflow, dlogp_pflow = probability_flow.rollout(x0_tests)
    logpx_pflow = logp0 + dlogp_pflow[-1].squeeze()
    xf_pflow = xfs_pflow[-1].squeeze().detach().cpu().numpy()
    # return xf_sde, logpx_sdeflow, xf_pflow, logpx_pflow
    return xfs_sde.detach().cpu().numpy(), logpx_sdeflow, xfs_pflow.detach().cpu().numpy(), logpx_pflow

# define plot functions
def make_plots(
    v: torch.nn.Module,
    s: torch.nn.Module,
    interpolant: Interpolant,
    n_save: int,
    n_likelihood: int,
    likelihood_bs: int,
    counter: int,
    metrics_freq: int,
    eps: torch.tensor,
    data_dict: dict
) -> None:
    """Make plots to visualize samples and evolution of the likelihood."""
    # compute likelihood and samples for SDE and probability flow.
    xfs_sde, logpx_sdeflow, xfs_pflow, logpx_pflow = compute_likelihood(
        v, s, interpolant, n_save, n_likelihood, eps, likelihood_bs
    )

    xf_sde = xfs_sde[-1]
    xf_pflow = xfs_pflow[-1]


    ### plot the loss, test logp, and samples from interpolant flow
    fig, axes = plt.subplots(1,4, figsize=(16,4))
    # print("EPOCH:", counter)
    # print("LOSS, GRAD:", loss, v_grad, s_grad)


    # plot loss over time.
    nsaves = len(data_dict['losses'])
    epochs = np.arange(nsaves)*metrics_freq
    axes[0].plot(epochs, data_dict['losses'], label=" v + s")
    axes[0].plot(epochs, data_dict['v_losses'], label="v")
    axes[0].plot(epochs, data_dict['s_losses'], label = "s" )
    axes[0].set_title("LOSS")
    axes[0].legend()


    # plot samples from SDE.
    axes[1].scatter(
        xf_sde[:,0], xf_sde[:,1], vmin=0.0, vmax=0.05, alpha = 0.2, c=grab(torch.exp(logpx_sdeflow).detach()))
    axes[1].set_xlim(-5,5)
    axes[1].set_ylim(-6.5, 6.5)
    axes[1].set_title("Samples from SDE", fontsize=14)


    # plot samples from pflow
    axes[2].scatter(
        xf_pflow[:,0], xf_pflow[:,1], vmin=0.0, vmax=0.05, alpha = 0.2, c=grab(torch.exp(logpx_pflow).detach()))
    axes[2].set_xlim(-5,5)
    axes[2].set_ylim(-6.5,6.5)
    axes[2].set_title("Samples from PFlow", fontsize=14)


    # plot likelihood estimates.
    axes[3].plot(epochs, data_dict['logps_pflow'],   label='pflow', color='purple')
    axes[3].plot(epochs, data_dict['logps_sdeflow'], label='sde',   color='red')
    # axes[3].hlines(
    #     y=grab(target_logp_est), xmin=0, xmax=epochs[-1], color='green', linestyle='--', label='exact', linewidth=2
    # )
    axes[3].set_title(r"$\log p$ from PFlow and SDE")
    axes[3].legend(loc='best')
    axes[3].set_ylim(-7,0)


    fig.suptitle(r"$\epsilon = $" + str(grab(eps)) + r" $n_{likelihood} = $" + str(n_likelihood), fontsize=16, y = 1.05)
    plt.savefig('../figures/states_' + str(counter) + '.jpg')

    # probability flows
    plt.figure(figsize=(15,3))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.scatter(xfs_pflow[i, :, 0], xfs_pflow[i, :, 1], vmin=0., vmax=0.05, alpha=0.2, c=grab(torch.exp(logpx_pflow).detach()))
        plt.ylim(-6.5, 6.5)
        plt.xlim(-5, 5)
        plt.title('ODE Flow: step: {}'.format(i+1))

    plt.savefig('../figures/PFLOW_' + str(counter) + '.jpg')

    # sde flows
    plt.figure(figsize=(15,3))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        if i != 0 and i != 4:
            plt.scatter(xfs_sde[2*i+1, :, 0], xfs_sde[2*i+1, :, 1], vmin=0., vmax=0.05, alpha=0.2, c=grab(torch.exp(logpx_sdeflow).detach()))
        elif i ==0:
            plt.scatter(xfs_sde[0, :, 0], xfs_sde[0, :, 1], vmin=0, vmax=0.05, alpha=0.2, c=grab(torch.exp(logpx_sdeflow).detach()))
        else:
            plt.scatter(xfs_sde[-1, :, 0], xfs_sde[-1, :, 1], vmin=0, vmax=0.05, alpha=0.2,
                        c=grab(torch.exp(logpx_sdeflow).detach()))
        plt.ylim(-6.5, 6.5)
        plt.xlim(-5, 5)
        plt.title('SDE Flow: {}%'.format((i+1)/5*100))
    plt.savefig('../figures/SDEFLOW_' + str(counter) + '.jpg')
    plt.show()

# define log metrics
def log_metrics(
        v: torch.nn.Module,
        s: torch.nn.Module,
        interpolant: Interpolant,
        n_save: int,
        n_likelihood: int,
        likelihood_bs: int,
        v_loss: torch.tensor,
        s_loss: torch.tensor,
        loss: torch.tensor,
        v_grad: torch.tensor,
        s_grad: torch.tensor,
        eps: torch.tensor,
        data_dict: dict
) -> None:
    # log loss and gradient data
    v_loss = grab(v_loss).mean()
    data_dict['v_losses'].append(v_loss)
    s_loss = grab(s_loss).mean()
    data_dict['s_losses'].append(s_loss)
    loss = grab(loss).mean()
    data_dict['losses'].append(loss)
    v_grad = grab(v_grad).mean()
    data_dict['v_grads'].append(v_grad)
    s_grad = grab(s_grad).mean()
    data_dict['s_grads'].append(s_grad)

    # compute and log likelihood data
    _, logpx_sdeflow, _, logpx_pflow = compute_likelihood(
        v, s, interpolant, n_save, n_likelihood, eps, likelihood_bs)

    logpx_sdeflow = grab(logpx_sdeflow).mean()
    data_dict['logps_sdeflow'].append(logpx_sdeflow)
    logpx_pflow = grab(logpx_pflow).mean()
    data_dict['logps_pflow'].append(logpx_pflow)

def train_step(
        prior_bs: int,
        target_bs: int,
        N_t: int,
        v: nn.Module,
        s: nn.Module,
        interpolant: Interpolant,
        loss_fac: float,
        opt,
        sched
):
    opt.zero_grad()
    x0s = base(prior_bs).to(device)
    x1s = target(target_bs).to(device)
    ts = torch.rand(size=(N_t, 1)).to(device)

    # compute the loss
    loss_v, loss_s = interpolant_loss(v, s, x0s, x1s, ts, interpolant, loss_fac)
    loss_val = loss_v + loss_s

    loss_val.backward()
    v_grad = torch.tensor([torch.nn.utils.clip_grad_norm_(v.parameters(), float('inf'))])
    s_grad = torch.tensor([torch.nn.utils.clip_grad_norm_(s.parameters(), float('inf'))])
    opt.step()
    sched.step()

    return loss_val.detach(), loss_v.detach(), loss_s.detach(), v_grad.detach(), s_grad.detach()


# 相关参数定义
gamma_type = 'brownian'
path = 'linear'
hiddens = [150, 150, 150, 150]
in_size = 3
out_size = 2
inner_act = 'relu'
final_act = 'none'
lr = 2e-3
eps = torch.tensor(0.4)
N_era = 6
epochs = 500
N_t = 1000 # number of times in batch
plot_bs = 5000
prior_bs = 1000 # number of samples from \rho_0
target_bs = 1000 # number of samples from \rho_1
loss_fac = 4.
sample_num = 20
metrics_freq = 250
plot_freq = 500
save_num = 10

data_dict = {
    'losses': [],
    'v_losses': [],
    's_losses': [],
    'v_grads': [],
    's_grads': [],
    'times': [],
    'logps_pflow': [],
    'logps_sdeflow': []
}

interpolant = Interpolant(path, gamma_type)
v = make_fc_net(hidden_sizes=hiddens, in_size=in_size, out_size=out_size, inner_act=inner_act, final_act=final_act).to(device)
s = make_fc_net(hidden_sizes=hiddens, in_size=in_size, out_size=out_size, inner_act=inner_act, final_act=final_act).to(device)
opt = torch.optim.Adam([*v.parameters(), *s.parameters()], lr=2e-3)
sched = torch.optim.lr_scheduler.StepLR(optimizer=opt, step_size=1500, gamma=0.4)

counter = 0
for i, era in enumerate(range(N_era)):
    with tqdm(range(epochs)) as it:
        for epoch in it:
            # counter += 1
            loss, loss_v, loss_s, grad_v, grad_s = train_step(prior_bs, target_bs, N_t, v, s, interpolant, loss_fac, opt, sched)

            if (counter) % metrics_freq == 0:
                log_metrics(v, s, interpolant, save_num, sample_num, prior_bs, loss_v, loss_s,
                            loss, grad_v, grad_s, eps, data_dict)

            if (counter) % plot_freq == 0:
                make_plots(v, s, interpolant, save_num, sample_num, plot_bs, counter, metrics_freq, eps, data_dict)

            counter += 1
            it.set_postfix(
                ordered_dict={
                    "epoch": counter,
                    "loss": loss.item(),
                    "loss_v": loss_v.item(),
                    "loss_s": loss_s.item(),
                }
            )