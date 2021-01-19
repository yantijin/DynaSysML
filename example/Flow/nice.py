import DynaSysML as dsl
import DynaSysML.Flow as tsf
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import imageio

class pre_scale_and_shift(nn.Module):
    def __init__(self, v_dim):
        super().__init__()
        self.v_dim = v_dim
        self.linear0 = nn.Linear(v_dim, 1000)
        self.linears = nn.ModuleList(nn.Linear(1000, 1000) for i in range(4))
        self.linear5 = nn.Linear(1000, self.v_dim)

    def forward(self,inputs):
        scale = torch.ones_like(inputs)
        hidden1 = torch.relu(self.linear0(inputs))
        for i in range(4):
            hidden1 = torch.relu((self.linears[i](hidden1)))
        shift = torch.relu(self.linear5(hidden1))
        return shift, scale


class Scale(nn.Module):
    def __init__(self, input_shape, **kwargs):
        super(Scale, self).__init__()
        self.input_shape = input_shape
        weights = torch.empty(1, input_shape[-1])
        # self._weight = torch.nn.init.normal_(weights).requires_grad_(True)
        self._weight = nn.Parameter(torch.nn.init.normal_(weights), requires_grad=True)

    def forward(self, inputs):
        return torch.exp(self.weight) * inputs

    @property
    def weight(self):
        return self._weight

    def inverse(self, inputs):
        scale = torch.exp(-self.weight)
        return scale * inputs


class NICE(nn.Module):
    def __init__(self, input_shape, **kwargs):
        super(NICE, self).__init__()
        self.input_shape = input_shape
        self.layer1 = tsf.CouplingLayer(
            pre_scale_and_shift(input_shape[-1] //2),
            axis=-1,
            event_ndims=1,
            scale='linear',
            secondary=False
        )
        self.layer2 = tsf.CouplingLayer(
            pre_scale_and_shift(input_shape[-1]//2),
            scale='linear',
            secondary=True
        )
        self.layer3 = tsf.CouplingLayer(
            pre_scale_and_shift(input_shape[-1]//2),
            scale='linear',
            secondary=False
        )
        self.layer4 = tsf.CouplingLayer(
            pre_scale_and_shift(input_shape[-1]//2),
            scale='linear',
            secondary=True
        )
        self.layer5 = Scale(input_shape)

    def forward(self, inputs):
        hidden, log_det = self.layer1(inputs,
                                       input_log_det=None,
                                       compute_log_det=True)
        hidden, log_det = self.layer2(hidden,
                                      input_log_det=log_det,
                                      compute_log_det=True)
        hidden, log_det = self.layer3(hidden,
                                      input_log_det=log_det,
                                      compute_log_det=True)
        hidden, log_det = self.layer4(hidden,
                                      input_log_det=log_det,
                                      compute_log_det=True)
        out = self.layer5(hidden)
        log_det += torch.sum(self.layer5.weight)
        return out, log_det

    def inverse(self, out):
        hidden = self.layer5.inverse(out)
        hidden = self.layer4(hidden, inverse=True)
        hidden = self.layer3(hidden, inverse=True)
        hidden = self.layer2(hidden, inverse=True)
        recon = self.layer1(hidden, inverse=True)
        return recon


def trainNICE():
    epochs = 20
    batch_size = 128
    x_dim = 784
    input_shape = [batch_size, x_dim]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../../data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)

    model = NICE(input_shape).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(1, epochs+1):
        for batch_idxs, (data, _) in enumerate(train_loader):
            data = data.to(device)
            data = data.view(-1, x_dim)
            optimiser.zero_grad()
            out, log_det = model(data)
            term1 = torch.sum(0.5*out**2, 1)
            loss = torch.mean(term1 - log_det)
            loss.backward()
            optimiser.step()
            if batch_idxs % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idxs * len(data), len(train_loader.dataset),
                           100. * batch_idxs / len(train_loader),
                    loss.item()))

    model.eval()
    n = 15
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    for i in range(n):
        for j in range(n):
            z_sample = torch.tensor(np.array(np.random.randn(1, 784)) * 0.75, dtype=torch.float32)
            # z_sample.to(device)
            x_decoded = model.inverse(z_sample)
            digits = x_decoded[0].reshape(digit_size, digit_size)
            # print(digits.shape)
            figure[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digits.detach().numpy()

    figure = np.clip(figure * 255, 0, 255)
    imageio.imwrite('test.png', figure)



if __name__ == "__main__":
    trainNICE()

