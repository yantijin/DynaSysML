import torch
from cons_data import generate_moons, eight_normal_sample
from DynaSysML.Flow.FlowMatching import CFM_vector_field, SB_CFM_vector_field, stochastic_interpolants
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchdiffeq import odeint
import ot # pip install POT
import numpy as np


# datasets
def sample_moons(n):
    x0, _ = generate_moons(n, noise=0.2)
    return x0 * 3 - 1

def sample_8gaussians(n):
    return eight_normal_sample(n, 2, scale=5, var=0.1).float()


class MLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x):
        return self.net(x)

class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x):
        return self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))


def plot_trajectories(traj, saveName=None):
    n = 2000
    plt.figure(figsize=(6, 6))
    plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c="black")
    plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.2, alpha=0.2, c="olive")
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=4, alpha=1, c="blue")
    plt.legend(["Prior sample z(S)", "Flow", "z(0)"])
    plt.xticks([])
    plt.yticks([])
    if saveName != None:
        plt.savefig(saveName)
    # plt.show()

def train_test_CFM(bs=256, dim=2, sigma=0.1, lr=1e-3, epochs=20000):
    model = MLP(dim=dim, time_varying=True)
    cmf_vf = CFM_vector_field(sigma)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    for i in tqdm(range(epochs)):
        optim.zero_grad()
        t = torch.rand(bs, 1)

        x_0 = sample_8gaussians(bs)
        x_1 = sample_moons(bs)

        x_t = cmf_vf.get_x_t(x_1, x_0, t)

        u_t = cmf_vf.get_vf_t(None, x_1, t, x_0)
        v_t = model(torch.cat([x_t, t], dim=-1))

        loss = torch.mean((v_t - u_t)**2)
        loss.backward()
        optim.step()
        if (i+1) % 5000 == 0:
            print(f'{i+1}: loss: {loss.item():0.3f}')
            with torch.no_grad():
                traj = odeint(torch_wrapper(model), sample_8gaussians(1024), 
                              torch.linspace(0, 1, 100)) #[100, 1024, 2]
                num  = (i+1) // 5000
                plot_trajectories(traj, saveName='../figures/CFM_' + str(num) + '.jpg')

def train_test_OT_CFM(bs=256, dim=2, sigma=0.1, lr=1e-3, epochs=20000):
    model = MLP(dim=dim, time_varying=True)
    cmf_vf = CFM_vector_field(sigma)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    for k in tqdm(range(epochs)):
        optim.zero_grad()
        t = torch.rand(bs, 1)

        x_0 = sample_8gaussians(bs)
        x_1 = sample_moons(bs)

        # resample x_0, x_1 according to transport matrix
        a, b = ot.unif(x_0.size()[0]), ot.unif(x_1.size()[0])
        M = torch.cdist(x_0, x_1) ** 2 # 计算距离的平方
        M = M / M.max()
        pi = ot.emd(a, b, M.detach().cpu().numpy()) #[bs, bs]
        p = pi.flatten() # 变成一维， bs*bs
        p = p/ p.sum() # 归一化
        choices = np.random.choice(pi.shape[0]*pi.shape[1], p=p, size=bs)
        i, j = np.divmod(choices, pi.shape[1]) # 商和余数，得到的是对应的index
        x_0 = x_0[i]
        x_1 = x_1[j]

        x_t = cmf_vf.get_x_t(x_1, x_0, t)

        u_t = cmf_vf.get_vf_t(None, x_1, t, x_0)
        v_t = model(torch.cat([x_t, t], dim=-1))

        loss = torch.mean((v_t - u_t) ** 2)
        loss.backward()
        optim.step()
        if (k + 1) % 5000 == 0:
            print(f'{k + 1}: loss: {loss.item():0.3f}')
            with torch.no_grad():
                traj = odeint(torch_wrapper(model), sample_8gaussians(1024),
                              torch.linspace(0, 1, 100))  # [100, 1024, 2]
                num  = (k+1) // 5000
                plot_trajectories(traj, saveName='../figures/ot_CFM_' + str(num) + '.jpg')

def train_test_SB_CFM(bs=256, dim=2, sigma=0.1, lr=1e-3, epochs=20000):
    model = MLP(dim=dim, time_varying=True)
    ot_cmf_vf = SB_CFM_vector_field(sigma)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    for k in tqdm(range(epochs)):
        optim.zero_grad()
        t = torch.rand(bs, 1)

        x_0 = sample_8gaussians(bs)
        x_1 = sample_moons(bs)

        # resample x_0, x_1 according to transport matrix
        a, b = ot.unif(x_0.size()[0]), ot.unif(x_1.size()[0])
        M = torch.cdist(x_0, x_1) ** 2 # 计算距离的平方
        M = M / M.max()
        pi = ot.sinkhorn(a, b, M.detach().cpu().numpy(), reg=2*(sigma**2)) #[bs, bs]

        # sample random interpolations on pi
        p = pi.flatten() # 变成一维， bs*bs
        p = p/ p.sum() # 归一化
        choices = np.random.choice(pi.shape[0]*pi.shape[1], p=p, size=bs)
        i, j = np.divmod(choices, pi.shape[1]) # 商和余数，得到的是对应的index
        x_0 = x_0[i]
        x_1 = x_1[j]

        x_t = ot_cmf_vf.get_x_t(x_1, x_0, t)

        u_t = ot_cmf_vf.get_vf_t(x_t, x_1, t, x_0)
        v_t = model(torch.cat([x_t, t], dim=-1))

        loss = torch.mean((v_t - u_t) ** 2)
        loss.backward()
        optim.step()
        if (k + 1) % 5000 == 0:
            print(f'{k + 1}: loss: {loss.item():0.3f}')
            with torch.no_grad():
                traj = odeint(torch_wrapper(model), sample_8gaussians(1024),
                              torch.linspace(0, 1, 100))  # [100, 1024, 2]
                num = (k+1)//5000
                plot_trajectories(traj, saveName='../figures/sb_CFM_' + str(num) + '.jpg')

def train_test_stochastic_interpolants(bs=256, dim=2, lr=1e-3, epochs=20000, loss_type='fm'):
    model = MLP(dim=dim, time_varying=True)
    sto_int = stochastic_interpolants()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    for k in tqdm(range(epochs)):
        optim.zero_grad()
        t = torch.rand(bs, 1)

        x_0 = sample_8gaussians(bs)
        x_1 = sample_moons(bs)

        x_t = sto_int.get_x_t(x_1, x_0, t)
        u_t = sto_int.get_vf_t(x_t, x_1, t, x_0)
        v_t = model(torch.cat([x_t, t], dim=-1))
        if loss_type == 'fm':
            loss = torch.mean((v_t-u_t)**2)
        elif loss_type == 'sto_int': # 使用原文中的stochastic interpolation的损失，其实两者是一样的，仅差一个常数
            loss = torch.mean(v_t**2 - 2 * u_t * v_t)
        else:
            raise ValueError('loss type must be fm or sto_int')
        loss.backward()
        optim.step()
        if (k + 1) % 5000 == 0:
            print(f'{k + 1}: loss: {loss.item():0.3f}')
            with torch.no_grad():
                traj = odeint(torch_wrapper(model), sample_8gaussians(1024),
                              torch.linspace(0, 1, 100))  # [100, 1024, 2]
                num = (k + 1) // 5000
                plot_trajectories(traj, saveName='../figures/sto_int_' + loss_type + '_' + str(num) + '.jpg')


# train_test_CFM()
# print('='*10 + 'CFM done' + '='*10)

# train_test_OT_CFM()
# print('='*10 + 'OT CFM done' + '='*10)

# train_test_SB_CFM()
# print('='*10 + 'SB CFM done' + '='*10)

train_test_stochastic_interpolants(loss_type='sto_int')

# plt.show()