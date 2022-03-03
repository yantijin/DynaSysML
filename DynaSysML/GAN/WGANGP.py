import torch
from DynaSysML.GAN import Vanilla_GAN


class WGANGP(Vanilla_GAN):
    def __init__(self, gen, dis, w=10):
        super(WGANGP, self).__init__(
            gen=gen,
            dis=dis
        )
        self.lam = w # lambda

    def gradient_penalty(self, fake2):
        out = self.dis(fake2)
        grad = torch.autograd.grad(out, fake2, grad_outputs=torch.ones_like(out), create_graph=True, retain_graph=True)[0]

        res = torch.sqrt(torch.sum(grad**2, dim=list(range(-1, -len(grad.shape), -1))) + 1e-9)
        return torch.mean((res-1)**2)

    def dloss(self, inputs, fake):
        eps = torch.rand_like(fake)
        fake2 = eps * inputs + (1 - eps) * fake
        out1 = self.dis(inputs)
        out2 = self.dis(fake)
        dloss = -(torch.mean(out1) - torch.mean(out2)) + self.lam * self.gradient_penalty(fake2)
        return dloss

    def gloss(self, fake):
        out = self.dis(fake)
        gloss = -torch.mean(out)
        return gloss