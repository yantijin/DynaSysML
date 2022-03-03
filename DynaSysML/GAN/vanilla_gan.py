import torch

class Vanilla_GAN():
    def __init__(self, gen, dis):
        super(Vanilla_GAN, self).__init__()
        self.gen = gen
        self.dis = dis

    def dloss(self, inputs, fake):
        out1 = self.dis(inputs)
        out2 = self.dis(fake)
        dloss = -(torch.mean(torch.log(out1 + 1e-8)) + torch.mean(torch.log(1-out2+1e-8)))
        return dloss

    def gloss(self, fake):
        out = self.dis(fake)
        gloss = -torch.mean(torch.log(out+1e-8))
        return gloss

    def train_gen_step(self, z, optim):
        optim.zero_grad()
        fake = self.gen(z)
        gloss = self.gloss(fake)
        gloss.backward()
        optim.step()
        return gloss

    def train_dis_step(self, inputs, z, optim):
        optim.zero_grad()
        fake = self.gen(z)
        dloss = self.dloss(inputs, fake)
        dloss.backward()
        optim.step()
        return dloss