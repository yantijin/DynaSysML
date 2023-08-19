import torch
import torch.nn as nn
import torch.nn.functional as F


class DDM_base(nn.Module):
    '''
    DDM: https://arxiv.org/pdf/2306.13720
    '''
    def __init__(self, net, eps=1e-4):
        super().__init__()
        '''
        net: eps_theta
        eps: used for denoise sampling
        '''
        self.net = net
        self.eps = eps

    def get_denoise_par(self, t, *args, **kwargs):
        res = (t,)
        if args:
            res += args
        if kwargs:
            res += tuple(kwargs.values())
        return res

    def get_H_t(self, t, *phi):
        pass

    def get_phi(self, x_0):
        pass

    def get_noisy_x(self, x_0, t, noise):
        t = t.reshape(t.shape[0], *((1,)*(len(x_0.shape)-1)))
        phi = self.get_phi(x_0=x_0)

        x_t = x_0 + self.get_H_t(t, *phi) + t.sqrt() * noise
        return x_t

    def forward(self, x_0, *args, **kwargs):
        noise = torch.randn_like(x_0)
        t = torch.rand((x_0.shape[0],),).to(x_0.device)
        x_t = self.get_noisy_x(x_0, t, noise)
        denoise_par = self.get_denoise_par(t, *args, **kwargs)
        phi = self.get_phi(x_0)
        phi_theta, eps_theta = self.net(x_t, *denoise_par)
        loss_phi_terms = [F.mse_loss(phi_term, hat_phi_term) for phi_term, hat_phi_term in zip(phi, phi_theta)]
        loss = sum(loss_phi_terms) + F.mse_loss(noise, eps_theta)
        return loss
    
    @torch.no_grad()
    def sample(self, shape, num_steps, device, denoise=True, clamp=True, *args, **kwargs):
        x = torch.randn(shape).to(device)
        x = self.sample_loop(x, num_steps, denoise=denoise,  clamp=clamp, *args, **kwargs)
        return x

    # 根据xt预测x_{t-1}
    def predict_xtm1_xt(self, xt, phi, noise, t, s):
        '''
        Input:
            xt: t时刻的状态
            phi: 计算H_t的coeff, 这里是net输出的第一个参数, 是一个tuple形式 (a, b, c,...)
            noise: net输出的第二个参数, 与xt形状相同
            t: 对应时刻的值 shape: [1]
            s: 步长 shape: [1]
        Return:
            x_{t-1}
        '''
        t = t.reshape(xt.shape[0], *((1,) * (len(xt.shape) - 1) ))
        s = s.reshape(xt.shape[0], *((1,) * (len(xt.shape) - 1) ))

        mean = xt + self.get_H_t(t-s, *phi) - self.get_H_t(t, *phi) - s / t.sqrt() * noise
        sigma = (s * (t - s) / t).sqrt()
        eps = torch.randn_like(mean, device=xt.device)
        return mean + sigma * eps
    
    def pred_x_start(self, x_t, noise, phi, t):
        t = t.reshape(t.shape[0], *((1,)*(len(x_t.shape)-1)))
        x_0 = x_t - self.get_H_t(t, *phi) - t.sqrt() * noise
        return x_0

    # TODO: 当ht是常数时，也就是为-x0时，是对这个做了修正, 其他直接用
    def sample_loop(self, x, num_steps, denoise=True,  clamp=True, *args, **kwargs):
        bs = x.shape[0]

        step = 1. / num_steps
        time_steps = torch.tensor([step]).repeat(num_steps)
        if denoise:
            time_steps = torch.cat((time_steps[:-1], torch.tensor([step - self.eps]), torch.tensor([self.eps])), dim=0)

        cur_time = torch.ones((bs, ), device=x.device)
        for i, time_step in enumerate(time_steps):
            s = torch.full((bs,), time_step, device=x.device)
            if i == time_steps.shape[0] - 1:
                s = cur_time

            denoise_par = self.get_denoise_par(cur_time, *args, **kwargs)
            phi_theta, eps_theta = self.net(x, *denoise_par)

            x = self.predict_xtm1_xt(x, phi_theta, eps_theta, cur_time, s)
            if clamp:
                x.clamp_(-1., 1.)
            cur_time = cur_time - s
       
        return x


class DDM_constant(DDM_base):
    def get_H_t(self, t, *phi):
        C = phi[0]
        t = t.reshape(t.shape[0], *((1,)*(len(C.shape)-1)))
        return t * C

    def get_phi(self, x_0):
        return (- x_0,)    
 

    # 当ht是常数的时候，C=-x0，因此可以先对x_0做预估，而不是直接用net的输出
    def sample_loop(self, x, num_steps, denoise=True,  clamp=True, *args, **kwargs):
        bs = x.shape[0]

        step = 1. / num_steps
        time_steps = torch.tensor([step]).repeat(num_steps)
        if denoise:
            time_steps = torch.cat((time_steps[:-1], torch.tensor([step - self.eps]), torch.tensor([self.eps])), dim=0)

        cur_time = torch.ones((bs, ), device=x.device)
        for i, time_step in enumerate(time_steps):
            s = torch.full((bs,), time_step, device=x.device)
            if i == time_steps.shape[0] - 1:
                s = cur_time

            denoise_par = self.get_denoise_par(cur_time, *args, **kwargs)
            phi_theta, eps_theta = self.net(x, *denoise_par)

            x_0 = self.pred_x_start(x, eps_theta, phi_theta, cur_time)
            if clamp:
                x_0.clamp_(-1., 1.)
            phi_new = (-x_0, )

            x = self.predict_xtm1_xt(x, phi_new, eps_theta, cur_time, s)
            cur_time = cur_time - s
        
        return x