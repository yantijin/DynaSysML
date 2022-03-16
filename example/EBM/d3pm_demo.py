import torch
from tqdm import tqdm
import numpy as np
from unet import logistic_Unet
from DynaSysML.EBM.d3pm import categorical_diffusion
from utils import plot_fig, get_mnist_loaders
import matplotlib.pyplot as plt


# train_loader, test_loader, val_loader = get_mnist_loaders()
# num = 0
# for term, _ in train_loader:
#     if num == 0:
#         term *= 255
#         term = term.long()
#         print(term.shape, term.dtype)
#         print(term[0], torch.max(term), torch.min(term))
#         num += 1

epochs = 5
md = logistic_Unet(dim=16, channels=1, dim_mults=(1, 2, 4)).cuda()
betas = np.linspace(1e-4, 1e-2, 1000)
diffusion = categorical_diffusion(betas, transition_bands=None, method='gaussian', num_bits=8,
                                  loss_type='hybrid', hybrid_coeff=0.001, model_prediction='x_start',
                                  model_output='logistic_pars').cuda()

optim = torch.optim.Adam(md.parameters(), lr=1e-3)
train_loader, test_loader, val_loader = get_mnist_loaders()

num = 0
for epoch in range(epochs):
    with tqdm(train_loader) as it:
        for x, label in it:
            num += 1
            optim.zero_grad()
            x *= 255
            x = x.long().cuda()
            loss = torch.sum(diffusion.training_losses(md, x))
            loss.backward()
            optim.step()
            it.set_postfix(
                ordered_dict={'train_loss': loss.item(), 'epoch': epoch},
                refresh=False
            )

shape = (36, 1, 28, 28)
samples = diffusion.p_sample_loop(md, shape, num_timesteps=None)
plot_fig(samples.detach().cpu().numpy(), show=True)
plt.show()
