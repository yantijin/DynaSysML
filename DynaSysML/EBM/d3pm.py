import torch
import torch.nn as nn
from scipy import special
import numpy as np


def get_transition_mat(beta, t, K, transition_bands, method='uniform'):
    beta_t = beta[t]
    mat = np.full(shape=(K, K), fill_value=beta_t/K, dtype=np.float32)
    if method == 'uniform':
        if transition_bands is None:
            diag_indices = np.diag_indices_from(mat)
            diag_val = 1. - beta_t * (K - 1.)/ K
            mat[diag_indices] = diag_val
            return mat
        else:
            off_diag = np.full(shape=(K-1,),
                               fill_value=beta_t/float(K),
                               dtype=np.float32)
            for j in range(1, transition_bands+1):
                mat += np.diag(off_diag, k=j)
                mat += np.diag(off_diag, k=-j)
                off_diag = off_diag[:-1]
            diag = 1. - mat.sum(1)
            mat += np.diag(diag, k=0)
            return mat
        
    elif method == 'gaussian':
        transition_bands = transition_bands if transition_bands else (K-1)
        values = np.linspace(start=0., stop=255., num=K, endpoint=True, dtype=np.float32)
        values = values * 2 / (K-1)
        values = values[:transition_bands+1]
        values = - values**2 / beta_t

        values = np.concatenate([values[:0:-1], values], axis=0)
        values = special.softmax(values, axis=0)
        values = values[transition_bands:]

        for k in range(1, transition_bands + 1):
            off_diag = np.full(shape=(K - k,),
                               fill_value=values[k],
                               dtype=np.float32)
            mat += np.diag(off_diag, k=k)
            mat += np.diag(off_diag, k=-k)
        
        diag = 1. - mat.sum(1)
        mat += np.diag(diag, k=0)
        return mat
    
    elif method == 'absorb':
        diag = np.full(shape=(K,), fill_value=1.-beta_t, dtype=np.float32)
        mat += np.diag(diag, k=0)
        mat[:, K//2] += beta_t
        return mat
    else:
        raise ValueError('Please check your args `method`, method should be `uniform`, `gaussian` or `absorb`')

def log_min_exp(a, b, epsilon=1e-6):
    # functions: compute log (e^a-e^b) in a numerically stable way
    return a + torch.log1p(-torch.exp(b-a) + epsilon)

def categorical_kl_logits(logits1, logits2, eps=1e-6):
    """KL divergence between categorical distributions.

    Distributions parameterized by logits.

    Args:
    logits1: logits of the first distribution. Last dim is class dim.
    logits2: logits of the second distribution. Last dim is class dim.
    eps: float small number to avoid numerical issues.

    Returns:
    KL(C(logits1) || C(logits2)): shape: logits1.shape[:-1]
    """
    out = (
        torch.softmax(logits1 + eps, dim=-1) *
        (torch.nn.functional.log_softmax(logits1 + eps, dim=-1) -
        torch.nn.functional.log_softmax(logits2 + eps, dim=-1)))
    return torch.sum(out, dim=-1)

def categorical_kl_probs(probs1, probs2, eps=1e-6):
    """KL divergence between categorical distributions.

    Distributions parameterized by logits.

    Args:
    probs1: probs of the first distribution. Last dim is class dim.
    probs2: probs of the second distribution. Last dim is class dim.
    eps: float small number to avoid numerical issues.

    Returns:
    KL(C(probs) || C(logits2)): shape: logits1.shape[:-1]
    """
    out = probs1 * (torch.log(probs1 + eps) - torch.log(probs2 + eps))
    return torch.sum(out, dim=-1)

def categorical_log_likelihood(x, logits):
    """Log likelihood of a discretized Gaussian specialized for image data.

    Assumes data `x` consists of integers [0, num_classes-1].

    Args:
    x: where to evaluate the distribution. shape = (bs, ...), dtype=int32/int64
    logits: logits, shape = (bs, ..., num_classes)

    Returns:
    log likelihoods
    """
    log_probs = torch.nn.functional.log_softmax(logits)
    x_onehot = torch.nn.functional.one_hot(x, logits.shape[-1])
    return torch.sum(log_probs * x_onehot, dim=-1)


class categorical_diffusion(nn.Module):
    def __init__(self, betas, transition_bands, method, num_bits, loss_type, hybrid_coeff, model_prediction, model_output):
        super().__init__()
        self.register_buffer("betas", torch.tensor(betas, dtype=torch.float32))
        self.transition_bands = transition_bands
        self.method = method
        self.num_timesteps = betas.shape[0]
        self.num_bits = num_bits
        self.num_pixel_values = 2**num_bits
        self.eps = 1e-6
        self.model_output = model_output
        self.model_prediction = model_prediction
        self.loss_type = loss_type
        self.hybrid_coeff = hybrid_coeff

        q_one_step_mats = [get_transition_mat(betas, i, self.num_pixel_values, transition_bands, method=method) for i in range(self.num_timesteps)]
        self.register_buffer("q_onestep_mats", torch.tensor(np.stack(q_one_step_mats, axis=0), dtype=torch.float32))

        q_mats = [q_one_step_mats[0]]
        q_mat_t = q_one_step_mats[0]
        for t in range(1, self.num_timesteps):
            q_mat_t = np.tensordot(q_mat_t, q_one_step_mats[t], axes=[[1], [0]])
            q_mats.append(q_mat_t)
        self.register_buffer("q_mats", torch.tensor(np.stack(q_mats, axis=0), dtype=torch.float32))
        self.register_buffer("transpose_q_onestep_mats", self.q_onestep_mats.transpose(1, 2))
        # self.transpose_q_onestep_mats = self.q_onestep_mats.transpose(1, 2)

    
    def _at(self, a, t, x):
        '''
        a: [bs, m, m]
        t: [bs, ]
        x: [bs, height, width, channels]. Should not be of one hot representation, 
            but have integer values representing the class values, long type
        out: [bs, height, width, channels, m]
        '''
        t = t.reshape(t.shape[0], *((1,) * (len(x.shape)-1)))
        # print(a.shape, t.shape, x.shape)
        return a[t, x] #.to(a.device)

    def _at_onehot(self, a, t, x):
        '''
        a: [bs, m, m]
        t: [bs, ]
        x: [bs, height, width, channels, num_pixel_values]. Should be of one hot representation, 
            but have integer values representing the class values float32 type
        out: [bs, height, width, channels, m]
        '''
        # print(x.shape, t.shape, a.shape, a[t].shape, a[t, None, None].shape)
        return torch.matmul(x, a[t, None, None])

    def q_probs(self, x_start, t,):
        return self._at(self.q_mats, t, x_start)

    def q_sample(self, x_start, t, noise):
        # get q(x_t|x_0), sampling via the gumbel distribution for categorical distribution, 
        # see https://en.wikipedia.org/wiki/Categorical_distribution
        logits = torch.log(self.q_probs(x_start, t) + self.eps)
        noise  = torch.clip(noise, min=self.eps, max=1.)
        gumbel_noise = -torch.log(-torch.log(noise)) # noise~Uniform(0, 1)
        return torch.argmax(logits + gumbel_noise, dim=-1)
    
    def q_posterior_logits(self, x_start, x_t, t, x_start_logits):
        # compute logits of q(x_{t-1}|x_t, x_0)
        if x_start_logits:
            assert x_start.shape == x_t.shape + (self.num_pixel_values, )
        else:
            assert x_start.shape == x_t.shape
        
        fact1 = self._at(self.transpose_q_onestep_mats, t, x_t)
        # print(t.device, x_t.device, self.transpose_q_onestep_mats.device)
        if x_start_logits:
            fact2 = self._at_onehot(self.q_mats, t-1, torch.softmax(x_start, dim=-1))
            tzero_logits = x_start
        else:
            fact2 = self._at(self.q_mats, t-1, x_start)
            tzero_logits = torch.log(torch.nn.functional.one_hot(x_start, num_classes=self.num_pixel_values)+self.eps)

        # print(fact1.device, fact2.device)
        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)
        t_broadcast = t.reshape(t.shape[0], *((1,) * (len(out.shape)-1)))
        # print(t_broadcast.dtype, tzero_logits.dtype, out.dtype)
        # print(t_broadcast.shape, tzero_logits.shape, out.shape)
        return torch.where(t_broadcast==0, tzero_logits, out)
    
    def _get_logits_from_logistic_pars(self, loc, log_scale):
        '''
        calculate logit based on an discretization of logistic regression
        loc: location parameters
        log_scale: log_scale par in logistical regression
        p(t)\approxlog(F(t+1/2) - F(t-1/2))
        F(t) = 1/(1+e^{-(x-mu)/scale})
        '''

        # Shift log_scale such that if it's zero the probs have a scale
        # that is not too wide and not too narrow either.
        loc = loc.unsqueeze(dim=-1)
        log_scale = log_scale.unsqueeze(dim=-1)
        inv_scale = torch.exp(- (log_scale - 2.))
        bin_width = 2. / (self.num_pixel_values-1)
        bin_centers = torch.linspace(0., 1., steps=self.num_pixel_values).to(loc.device) # TODO: 这里需要根据数据的上下界来确定linspace上下界
        bin_centers = bin_centers.reshape(*((1,)*(len(loc.shape)-1)), bin_centers.shape[0])
        # print(bin_centers.shape, loc.shape, log_scale.shape)
        bin_centers = bin_centers - loc
        log_cdf_min = torch.nn.functional.logsigmoid(inv_scale*(bin_centers - 0.5*bin_width))
        log_cdf_plus = torch.nn.functional.logsigmoid(inv_scale*(bin_centers + 0.5*bin_width))
        
        logits = log_min_exp(log_cdf_min, log_cdf_plus, self.eps)
        return logits

    def p_logits(self, model_fn, *, x, t):
        # compute logits of p(x_{t-1}|x_t)
        # print(x.dtype, t.dtype)
        model_output = model_fn(x, t)
        if self.model_output == 'logits':
            model_logits = model_output
        elif self.model_output == 'logistic_pars':
            # get logits out of discretized logistic discretization
            loc, log_scale = model_output
            model_logits = self._get_logits_from_logistic_pars(loc, log_scale)
        else:
            raise NotImplementedError(self.model_output)

        
        if self.model_prediction == 'x_start':
            pred_x_start_logits = model_logits

            t_broadcast = t.reshape(t.shape[0], *((1,) * (len(model_logits.shape)-1)))
            model_logits = torch.where(t_broadcast==0,
                                        pred_x_start_logits,
                                        self.q_posterior_logits(pred_x_start_logits, x, t, x_start_logits=True))
        
        elif self.model_prediction == 'xprev':
            pred_x_start_logits = model_logits
            raise NotImplementedError(self.model_prediction)
    
        return model_logits, pred_x_start_logits

    def p_sample(self, model_fn, x, t, noise):
        model_logits, pred_x_start_logits = self.p_logits(model_fn=model_fn, x=x, t=t)
        assert noise.shape == model_logits.shape, noise.shape

        nonzero_mask = (t != 0).reshape(x.shape[0], *((1,)*len(x.shape))).float()
        noise = torch.clip(noise, self.eps, 1.)
        gumbel_noise = -torch.log(-torch.log(noise))
        sample = torch.argmax(model_logits + nonzero_mask*gumbel_noise, dim=-1)
        return sample, torch.softmax(pred_x_start_logits, dim=-1)
    
    def p_sample_loop(self, model_fn, shape, num_timesteps=None):
        '''
        ancestral sampling methods
        '''
        # init_rng, body_rng = np.random.split(rng)
        # del rng
        noise_shape = shape + (self.num_pixel_values, )
        def bofy_func(i, x):
            t = torch.full((shape[0]), self.num_timesteps-1-i, dtype=torch.long)
            x, _ = self.p_sample(
                model_fn=model_fn,
                x=x,
                t=t,
                noise=torch.rand(noise_shape).cuda() #TODO: 这里需要有device
            )
            return x

        if self.method in ['gaussian', 'uniform']:
            x_init = torch.randint(0, self.num_pixel_values, noise_shape)
        elif self.method == 'absorb':
            x_init = torch.full(shape, self.num_pixel_values//2, dtype=torch.long)
        else:
            raise NotImplementedError(self.method)

        
        if num_timesteps is None:
            num_timesteps = self.num_timesteps

        for i in range(0, num_timesteps):
            x_init = bofy_func(i, x_init)
        
        return x_init

    def vb_terms_bpd(self, model_fn, x_start, x_t, t):
        """Calculate specified terms of the variational bound.

            Args:
            model_fn: the denoising network
            x_start: original clean data
            x_t: noisy data
            t: timestep of the noisy data (and the corresponding term of the bound
                to return)

            Returns:
            a pair `(kl, pred_start_logits)`, where `kl` are the requested bound terms
            (specified by `t`), and `pred_x_start_logits` is logits of
            the denoised image.
        """
        true_logits = self.q_posterior_logits(x_start, x_t, t, x_start_logits=False)
        model_logits, pred_x_start_logits = self.p_logits(model_fn, x=x_t, t=t)
        kl = categorical_kl_logits(true_logits, model_logits)
        kl = torch.mean(kl, dim=list(range(1, len(kl.shape)))) / np.log(2.)

        decoder_nll = - categorical_log_likelihood(x_start, model_logits)
        decoder_nll = torch.mean(decoder_nll, dim=list(range(1, len(decoder_nll.shape)))) / np.log(2.)

        return torch.where(t==0, decoder_nll, kl), pred_x_start_logits

    def prior_bpd(self, x_start):
        '''
        KL(q(x_{T-1}|x_start) || U(x_{T-1}|0, num_pixel_vals-1) )
        '''
        q_probs = self.q_probs(x_start=x_start, t=torch.full((x_start.shape[0], ), self.num_timesteps-1, dtype=torch.long))
    
        if self.method in ['guassian', 'uniform']:
            prior_probs = torch.ones_like(q_probs) / self.num_pixel_values
        elif self.method == 'absorb':
            absorb_int = torch.full(q_probs.shape[:-1], self.num_pixel_values//2, dtype=torch.long)
            prior_probs = torch.nn.functional.one_hot(absorb_int, num_classes=self.num_pixel_values).float()
        else:
            raise NotImplementedError(self.method)
    
        
        kl_prior = categorical_kl_probs(q_probs, prior_probs)
        return torch.mean(kl_prior, dim=list(range(1, len(kl_prior.shape)))) / np.log(2.)
    
    def cross_entropy_x_start(self, x_start, pred_x_start_logits):
        '''
        calculate cross entropy between x_start and predicted x_start
        '''
        ce = - categorical_log_likelihood(x_start, pred_x_start_logits)
        ce = torch.mean(ce, dim=list(range(1, len(ce.shape)))) / np.log(2.)
        return ce

    def training_losses(self, model_fn, x_start):
        # print(self.transpose_q_onestep_mats.device, self.q_onestep_mats.device, self.q_mats.device, self.betas.device)
        noise = torch.rand(x_start.shape+(self.num_pixel_values,)).cuda()
        t = torch.randint(0, self.num_timesteps, (x_start.shape[0], )).long().to(x_start.device)
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        if self.loss_type == 'kl':
            losses, _ = self.vb_terms_bpd(model_fn=model_fn, x_start=x_start, x_t=x_t, t=t)
        elif self.loss_type == 'cross_entropy_x_start':
            _, pred_x_start_logits = self.p_logits(model_fn, x=x_t, t=t)
            losses = self.cross_entropy_x_start(x_start=x_start, pred_x_start_logits=pred_x_start_logits)
        
        elif self.loss_type == 'hybrid':
            vb_losses, pred_x_start_logits = self.vb_terms_bpd(model_fn=model_fn, x_start=x_start, x_t=x_t, t=t)
            ce_losses = self.cross_entropy_x_start(x_start=x_start, pred_x_start_logits=pred_x_start_logits)
            losses = vb_losses + self.hybrid_coeff * ce_losses
        else:
            raise NotImplementedError(self.loss_type)
        return losses
    
    def calc_bpd_loop(self, model_fn, *, x_start):
        bs = x_start.shape[0]
        noise_shape = x_start.shape + (self.num_pixel_vals,)
        def map_fn(t):

            t = torch.full((bs,), t).long()
            vb, _ = self.vb_terms_bpd(model_fn=model_fn, x_start=x_start, t=t,
                                    x_t=self.q_sample(x_start=x_start, t=t, noise=torch.rand(noise_shape).cuda()))
            return vb
        
        vbterms_tb = []
        for i in range(self.num_timesteps):
            vbterms_tb.append(map_fn(i))
        vbterms_tb = torch.stack(vbterms_tb, dim=-1)
        assert vbterms_tb.shape == (bs, self.num_timesteps)
        prior_b = self.prior_bpd(x_start=x_start)
        total_b = torch.sum(vbterms_tb, dim=0) + prior_b
        return {
            'total': total_b,
            'vbterms': vbterms_tb,
            'prior': prior_b
        }


if __name__ == "__main__":
    from example.EBM.unet import Unet
    K = 20
    # transition_bands = [None, 3]
    beta = np.linspace(1e-4, 1e-2, 100)
    t =  20
    method_ls = ['uniform', 'gaussian', 'absorb']
    # for method in method_ls:
    #     for tb in transition_bands:
    #         mat = get_transition_mat(beta, t, K, tb, method=method)
    #         print(mat.shape)
    transition_bands = None
    num_bits = 8
    loss_type_ls = ['kl', 'cross_entropy_x_start', 'hybrid']
    model_prediction = 'x_start'
    # model_output_ls = ['logits', ]#
    model_output_ls = ['logistic_pars']
    betas = np.linspace(1e-4, 1e-2, 100)

    inputs = torch.randint(0, 256, (32, 3, 28, 28)).cuda()
    hybrid_coeff = 0.001
    def md_func(x, t):
        res = torch.nn.functional.one_hot(x, 256).float()
        return res

    def md_func1(x, t):
        return x/2, x /255

    for method in method_ls:
        for loss_type in loss_type_ls:
            for model_output in model_output_ls:

                md = categorical_diffusion(betas, transition_bands, method, num_bits, 
                                           loss_type, hybrid_coeff, model_prediction, model_output).cuda()
                loss = md.training_losses(md_func1, x_start=inputs)
                print('+'*80)
                print(loss.shape, torch.mean(loss).item())
                print('+' * 80)