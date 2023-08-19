import torch
import numpy as np
import matplotlib.pyplot as plt
from unet import *
from DynaSysML.EBM.DecoupledDPM import *
from tqdm import tqdm
from utils import plot_fig, get_mnist_loaders
from torchvision.utils import save_image


num_steps = 100
TRAIN = True
epochs = 10
eps = 1e-4

model = Decoupled_Unet(dim=16, channels=1, dim_mults=(1, 2, 4)).cuda()
ddm_model = DDM_constant(model, eps=eps).cuda()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
train_loader, _, _ = get_mnist_loaders()

if TRAIN:
    for epoch in range(epochs):
        with tqdm(train_loader) as it:
            for x, _ in it:
                optim.zero_grad()
                loss = ddm_model(x.cuda())
                loss.backward()
                optim.step()

                it.set_postfix(
                    ordered_dict={
                        'train_loss': loss.item(),
                        'epoch': epoch
                    },
                    refresh=False
                )
    
    torch.save(model.state_dict(), '../pts/ddm.pt')
else:
    model.load_state_dict(torch.load('../pts/ddm.pt'))
    print('load model success')

# sampling process
sampling_shape = (36, 1, 28, 28)
res = ddm_model.sample(sampling_shape, num_steps=10, device='cuda:0', denoise=True, clamp=False)
save_image(res, '../figures/decoupled_dm.jpg', nrow=6)