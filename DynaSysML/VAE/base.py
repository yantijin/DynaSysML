import torch
import torch.nn as nn
import math


class VAE(nn.Module):
    def __init__(self,
                 in_shape,
                 hidden_dim,
                 encoder,
                 decoder,
                 prior_dist,
                 encode_dist,
                 decode_dist,
                 flow=None,
                 num_samples=None):
        super(VAE, self).__init__()
        # 注意这里encoder, decoder以及prior_dist, flow(optional)已经提前写好了！！
        # 注意decoder中的输入z的形状为[ns, bs, ...]
        self.in_shape = in_shape
        self.hidden_dim = hidden_dim
        self.encoder = encoder
        self.decoder = decoder
        self.prior_dist = prior_dist
        self.enc_dist = encode_dist
        self.dec_dist = decode_dist
        self.flow = flow
        if num_samples is not None:
            self.num_samples = num_samples
        else:
            self.num_samples = 1

    def forward(self, x):
        enc_dist = self.enc_dist(*self.encoder(x))
        z = enc_dist.rsample((self.num_samples,)) # [ns, b, ...]
        if self.flow is not None:
            z, _ = self.flow(z)
        dec_dist = self.dec_dist(*self.dec_dist(z))
        return enc_dist, dec_dist

    def logqz_x(self, enc_dist):
        logqz_x = torch.mean(torch.sum(enc_dist.log_prob(enc_dist.rsample((self.num_samples, ))), dim=list(range(2, len(self.in_shape)))))
        return logqz_x

    def logpx_z(self, dec_dist, x): # 注意这里的x和forward里面的x一定要对应好
        return torch.mean(torch.sum(dec_dist.log_prob(x.expand(self.num_samples, *x.shape)), dim=list(range(2, len(self.in_shape)))))

    def logp_z(self, prior_dist, enc_dist):
        if self.flow is not None:
            z, log_det = self.flow(enc_dist.rsample((self.num_samples, )))
        else:
            z = enc_dist.rsample((self.num_samples, ))
            log_det = None
        logp_zk = torch.mean(torch.sum(prior_dist.log_prob(z), dim=list(range(2, len(self.in_shape)))))
        if log_det:
            logp_zk += torch.mean(log_det)
        return logp_zk

    def base_loss(self, prior_dist, enc_dist, dec_dist, x):
        logqz_x = self.logqz_x(enc_dist)
        logpx_z = self.logpx_z(dec_dist, x)
        logp_z = self.logp_z(prior_dist, enc_dist)
        loss = - (logpx_z + logp_z - logqz_x)
        return loss

    def cygen_loss(self, prior_dist, enc_dist, dec_dist, x):
        # see NeurIPS'21 `on the generative utility of cylic conditionals` https://arxiv.org/abs/2106.15962
        # fitting data loss:
        l1 = torch.sum(dec_dist.log_prob(x.expand((self.num_samples, *x.shape))), dim=list(range(2, len(self.in_shape))))
        loss1 = torch.mean(torch.logsumexp(-l1, dim=0) - math.log(self.num_samples))
        # compatability:
        




