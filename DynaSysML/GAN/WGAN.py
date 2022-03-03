import torch
from DynaSysML.GAN import Vanilla_GAN


class WGAN(Vanilla_GAN):
    def __init__(self, gen, dis, wc=0.01):
        super(WGAN, self).__init__(
            gen=gen,
            dis=dis
        )
        self.wc=wc # weight clipping

    def dloss(self, inputs, fake):
        out1 = self.dis(inputs)
        out2 = self.dis(fake)
        dloss = -(torch.mean(out1) - torch.mean(out2))
        return dloss

    def gloss(self, fake):
        out = self.dis(fake)
        gloss = -torch.mean(out)
        return gloss

    def train_dis_step(self, inputs, z, optim):
        optim.zero_grad()
        for p in self.dis.parameters():
            p.data.clamp_(-self.wc, self.wc)
        fake = self.gen(z)
        dloss = self.dloss(inputs, fake)
        dloss.backward()
        optim.step()
        return dloss