from DynaSysML.GAN import WGANGP
from tqdm import tqdm
from example.EBM.utils import get_mnist_loaders
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from example.GAN.layers import *

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def train_WGANGP(z_dim=100, epochs=200, batch=100):
    train_loader, test_loader, val_loader = get_mnist_loaders(batch_size=batch)
    gen = ConvGen(z_dim).to(device)
    dis = ConvDis().to(device)
    gen_optim = torch.optim.Adam(gen.parameters(), lr=1e-4, betas=(0, 0.9))
    dis_optim = torch.optim.Adam(dis.parameters(), lr=1e-4, betas=(0, 0.9))

    gan = WGANGP(gen, dis)

    step = 0
    g_loss = 0
    d_loss = 0
    for i in range(epochs):
        with tqdm(train_loader) as it:
            for data, _ in it:
                step += 1
                data = data.to(device)
                z = torch.randn(batch, z_dim, 1, 1)
                d_loss = gan.train_dis_step(data, z.to(device), dis_optim)
                if step % 5 == 0:
                    g_loss = gan.train_gen_step(z.to(device), gen_optim)
                it.set_postfix(
                    ordered_dict={
                        'gloss': g_loss,
                        'dloss': d_loss,
                        'epoch': i
                    },
                    refresh=False
                )

        if (i+1) % 10 == 0:
            print('epoch reaches: ', i)
            with torch.no_grad():
                z = torch.randn(batch, z_dim, 1, 1)
                fake = gen(z.to(device)).detach().cpu()
                save_image(fake, './gp_wgan_res' + str(i+1) + '.jpg', nrow=10)


if __name__ == "__main__":
    train_WGANGP()

