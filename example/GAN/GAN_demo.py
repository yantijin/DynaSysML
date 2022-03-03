from DynaSysML.GAN import Vanilla_GAN
import torch
import torch.nn as nn
from tqdm import tqdm
from example.EBM.utils import plot_fig, get_mnist_loaders
from example.GAN.layers import *
import matplotlib.pyplot as plt
from torchvision.utils import save_image


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def train_GAN(x_dim=28*28, h_dim=128, z_dim=20, epochs=200, batch=100):
    train_loader, test_loader, val_loader = get_mnist_loaders(batch_size=batch)
    gen = MLPGen(z_dim, h_dim, x_dim).to(device)
    dis = MLPDis(x_dim, h_dim).to(device)
    gen_optim = torch.optim.Adam(gen.parameters(), lr=1e-3)
    dis_optim = torch.optim.Adam(dis.parameters(), lr=1e-3)

    gan = Vanilla_GAN(gen, dis)

    step = 0
    g_loss = 0
    d_loss = 0
    for i in range(epochs):
        with tqdm(train_loader) as it:
            for data, _ in it:
                step += 1
                data = data.reshape(batch, -1).to(device)
                if step % 2 == 0:
                    z = torch.rand(batch, z_dim) * 2 -1
                    g_loss = gan.train_gen_step(z.to(device), gen_optim)
                else:
                    z = torch.rand(batch, z_dim) * 2 - 1
                    d_loss = gan.train_dis_step(data, z.to(device), dis_optim)

                it.set_postfix(
                    ordered_dict={
                        'gloss': g_loss,
                        'dloss': d_loss,
                        'epoch': i
                    },
                    refresh=False
                )

                # if step % 500 == 0:
                #     fake = gen(z.to(device))
                #     print("gloss:{}, dloss:{}".format(gan.gloss(fake).item(), gan.dloss(data, fake)))

        if (i+1) % 10 == 0:
            print('epoch reaches: ', i)
            with torch.no_grad():
                z = torch.rand(batch, z_dim) * 2 - 1
                fake = gen(z.to(device)).view(batch, 1, 28, 28).detach().cpu()
                # print(fake[1, 0])
                # plt.imshow(fake[1, 0])
                # plot_fig(res=fake, path='./res' + str(i+1)+'.jpg', n=10)
                save_image(fake, './res' + str(i+1) + '.jpg', nrow=10)


if __name__ == "__main__":
    train_GAN()
