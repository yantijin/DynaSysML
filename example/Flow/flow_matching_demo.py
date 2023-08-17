import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_circles
from DynaSysML.Flow.FlowMatching import VESDE_vector_field, VPSDE_vector_field, OT_vector_field
from tqdm import tqdm
from DynaSysML.Flow.utils import extract_coeff


def sample_batch(size, factor=0.5, noise=0.01):
    x, y = make_circles(size, factor=factor, noise=noise)
    x *= 5
    return x # [size, 2]

def constrcut_dataset(size, batch, shuffle=True):
    data = sample_batch(size)
    # import pdb; pdb.set_trace()
    dataset = TensorDataset(torch.from_numpy(data).float())
    data_loader = DataLoader(dataset, batch_size=batch, shuffle=shuffle)
    return data_loader

class Model(nn.Module):
    def __init__(self, hdim=256):
        super().__init__()
        self.fc1 = nn.Linear(3, hdim)
        self.fc2 = nn.Linear(hdim, hdim)
        self.fc3 = nn.Linear(hdim, 2)

    def forward(self, x, t):
        x_in = torch.cat((x, extract_coeff(t, x.shape)), dim=1)
        h = F.silu(self.fc1(x_in))
        h = F.silu(self.fc2(h)) + h
        out = self.fc3(h)
        return out

def train_test(lr, batch=16, epoch=200, vf_type='OT', train=True):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = Model().to(device)
    if vf_type == 'OT':
        vf = OT_vector_field(sigma_min=1e-8)
    elif vf_type == 'VESDE':
        vf = VESDE_vector_field()
    elif vf_type == 'VPSDE':
        vf = VPSDE_vector_field()

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    data_loader = constrcut_dataset(10000, batch=batch, shuffle=True)
    if train:
        for i in range(epoch):
            loss_ls = []
            with tqdm(data_loader) as it:
                for dt in data_loader: # [bs, 2]
                    x = dt[0].to(device)
                    # print(x.shape)
                    optim.zero_grad()
                    t = torch.rand(x.shape[0]).to(device)
                    x_t = vf.get_x_t(x, None, t)
                    u_t = vf.get_vf_t(x_t, x, t)
                    v_t = model(x_t, t)
                    loss = (u_t - v_t).abs().mean()
                    loss.backward()
                    optim.step()
                    loss_ls.append(loss.item())
                    it.set_postfix(
                        ordered_dict={
                            'train_loss': round(np.mean(loss_ls), 4),
                            'epoch': i
                        },
                        refresh=False
                    )

        torch.save(model.state_dict(), '../pts/FM_train.pt')
    else:
        model.load_state_dict(torch.load('../pts/FM_train.pt'))

    # 测试
    model.eval()
    print('=======现在开始测试=============')
    ts = np.linspace(0., 1. - 1/1000, num=1000)
    x_t = torch.randn(10000, 2).to(device)
    with torch.no_grad():
        for t in tqdm(ts):
            t = torch.Tensor([t]*x_t.shape[0]).float().to(device)
            # t = t.repeat(x_t.shape[0])
            v_t = model(x_t, t)
            x_t = x_t + 1 / 1000 * v_t
        x_t = x_t.cpu().numpy()
    plt.figure()
    plt.scatter(x_t[:, 0], x_t[:, 1], alpha=0.5, color='red', edgecolors='white', s=40)
    plt.title('sampled data')
    plt.savefig('../figures/FM_' + vf_type + '.jpg')
    # plt.show()

train_test(1e-3, 32, 120, train=True)