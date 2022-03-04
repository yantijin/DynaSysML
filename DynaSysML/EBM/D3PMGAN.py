import torch
from DynaSysML.EBM import categorical_diffusion


class d3pm_gen(categorical_diffusion):
    def __init__(self, denoise_fn, betas, transition_bands, method, num_bits, loss_type, hybrid_coeff, model_prediction, model_output):
        super(d3pm_gen, self).__init__(
            betas=betas,
            transition_bands=transition_bands,
            method=method,
            num_bits=num_bits,
            loss_type=loss_type,
            hybrid_coeff=hybrid_coeff,
            model_prediction=model_prediction,
            model_output=model_output
        )
        self.denoise_fn = denoise_fn
        self.num_pixel_values = 2 ** num_bits

    def q_sample(self, x_start, t, noise):
        # get q(x_t|x_0), sampling via the gumbel distribution for categorical distribution,
        # see https://en.wikipedia.org/wiki/Categorical_distribution
        zero_idx = t<0
        t[zero_idx] = 0
        logits = torch.log(self.q_probs(x_start, t) + self.eps)
        logits[zero_idx] = 0.

        noise  = torch.clip(noise, min=self.eps, max=1.)
        gumbel_noise = -torch.log(-torch.log(noise)) # noise~Uniform(0, 1)
        return torch.argmax(logits + gumbel_noise, dim=-1)

    def forward(self, x, *args, **kwargs):
        b, device = x.shape[0], x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        noise = torch.rand(x.shape + (self.num_pixel_values,)).to(x.device)
        x_t = self.q_sample(x, t, noise)
        x_t_1 = self.q_sample(x, t-1, noise=noise)
        denoise_par = self.get_denoise_par(t, *args, **kwargs)
        model_output = self.denoise_fn(x_t, *denoise_par)
        if self.model_output == 'logistic_pars':
            loc, log_scale = model_output
            assert torch.sum(torch.isnan((loc))) == 0
            assert torch.sum(torch.isnan(log_scale)) == 0
            x_start_logits = self._get_logits_from_logistic_pars(loc, log_scale)
        elif self.model_output == 'logits':
            assert torch.sum(torch.isnan(model_output)) == 0
            x_start_logits = model_output
        else:
            raise NotImplementedError

        x_0_pred = torch.softmax(x_start_logits, dim=-1)
        x_t_1_logits = self.q_posterior_logits(x_start_logits, x_t, t, x_start_logits=True)
        x_t_1_pred = torch.softmax(x_t_1_logits, dim=-1)
        return x_0_pred, x_t, x_t_1, x_t_1_pred, t

    @torch.no_grad()
    def p_sample(self, model_fn, x, t, noise, *args, **kwargs):
        denoise_par = self.get_denoise_par(t, *args, **kwargs)
        model_output = model_fn(x, *denoise_par)
        if self.model_output == 'logistic_pars':
            loc, log_scale = model_output
            assert torch.sum(torch.isnan((loc))) == 0
            assert torch.sum(torch.isnan(log_scale)) == 0
            x_start_logits = self._get_logits_from_logistic_pars(loc, log_scale)
        elif self.model_output == 'logits':
            assert torch.sum(torch.isnan(model_output)) == 0
            x_start_logits = model_output
        else:
            raise NotImplementedError

        x_t_1_logits = self.q_posterior_logits(x_start_logits, x, t, x_start_logits=True)
        assert noise.shape == x_start_logits.shape
        nonzero_mask = (t != 0).reshape(x.shape[0], *((1,) * len(x.shape))).float()
        noise = torch.clip(noise, self.eps, 1.)
        gumbel_noise = -torch.log(-torch.log(noise))
        sample = torch.argmax(x_t_1_logits + nonzero_mask * gumbel_noise, dim=-1)
        return sample,  torch.softmax(x_start_logits, dim=-1)



class d3pm_gan():
    def __init__(self, dis, denoise_fn, betas, gamma=0.05, device='cpu'):
        self.dis = dis
        self.denoise_fn = denoise_fn
        self.diffusion_gen = (denoise_fn, betas).to(device)
        self.gamma = gamma

    def gloss(self, x, *args, **kwargs):
        x_0_pred, x_t, _, x_t_1_pred, t = self.diffusion_gen(x, *args, **kwargs)
        out = self.dis(x_t, x_t_1_pred, t)
        # print(out.shape)
        gloss = -torch.mean(torch.log(out + 1e-8))
        return gloss

    def dloss(self, x, grad_penal=True, *args, **kwargs):
        x_0_pred, x_t, x_t_1, x_t_1_pred, t = self.diffusion_gen(x, *args, **kwargs)
        out1 = self.dis(x_t, x_t_1, t)
        out2 = self.dis(x_t, x_t_1_pred, t)
        dloss = -(torch.mean(torch.log(out1 + 1e-8)) + torch.mean(torch.log(1-out2+1e-8)))
        if grad_penal:
            dloss += self.gamma / 2 * torch.mean(self.gradient_panalty(x_t, x_t_1, t))
        return dloss

    def gradient_panalty(self, x_t, x_t_1, t):
        x_t_1.requires_grad_(True)
        out = self.dis(x_t, x_t_1, t)
        # print(out.requires_grad, x_t_1.requires_grad)
        grad = torch.autograd.grad(out, x_t_1, grad_outputs=torch.ones_like(out), retain_graph=True, create_graph=True)[0]
        return grad**2

    def train_gen_step(self, x, optim, *args, **kwargs):
        optim.zero_grad()
        gloss = self.gloss(x, *args, **kwargs)
        gloss.backward()
        optim.step()
        return gloss

    def train_dis_step(self, x, optim, *args, **kwargs):
        optim.zero_grad()
        dloss = self.dloss(x, *args, **kwargs)
        dloss.backward()
        optim.step()
        return dloss