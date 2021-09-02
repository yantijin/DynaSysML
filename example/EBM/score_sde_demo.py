from DynaSysML.EBM.sde_lib import *
from DynaSysML.EBM.sample import *
from DynaSysML.EBM.losses import *
from DynaSysML.EBM.utils import EMA, get_score_fn
from utils import get_mnist_loaders, plot_fig
from unet import cont_Unet
import ml_collections
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt


def config_mnist():
    config = ml_collections.ConfigDict()
    config.model = modeling = ml_collections.ConfigDict()
    modeling.beta_min = 0.1
    modeling.beta_max = 10
    modeling.num_scales = 1000

    config.train = training = ml_collections.ConfigDict()
    training.epochs = 10
    training.ema_decay = 0.999
    training.update_ema_every = 10

    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.predictor = 'ancestral_sampling'
    sampling.corrector = 'none'
    sampling.eps = 1e-3
    sampling.snr = 0.16
    return config


def train(config):
    train_loader, _, _ = get_mnist_loaders()

    model = cont_Unet(dim=16, channels=1, dim_mults=(1, 2, 4)).cuda()
    ema_model = copy.deepcopy(model)
    sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = get_sde_loss_fn(sde, train=True, reduce_mean=True, continuous=True, likelihood_weighting=False)
    ema = EMA(config.train.ema_decay)
    num = 0
    for epoch in range(config.train.epochs):
        with tqdm(train_loader) as it:
            for data, _ in it:
                num += 1
                optim.zero_grad()
                loss = loss_fn(model, data.cuda())
                loss.backward()
                optim.step()
                it.set_postfix(
                    ordered_dict={
                        'train loss': loss.item(),
                        'epoch': epoch
                    },
                    refresh=False
                )
                if num % config.train.update_ema_every == 0:
                    if num < 1000:
                        ema_model.load_state_dict(model.state_dict())
                    else:
                        ema.update_model_average(ema_model, model)


    # sample
    shape = (36, 1, 28, 28)
    predictor = get_predictor(config.sampling.predictor)
    corrector = get_corrector(config.sampling.corrector)
    sampling_fn = get_pc_sampler(sde, shape,
                                 predictor=predictor,
                                 corrector=corrector,
                                 inverse_scaler=lambda x: x,
                                 snr=config.sampling.snr,
                                 eps=config.sampling.eps)

    samples, n = sampling_fn(model)
    samples = samples.detach().cpu().numpy()
    plot_fig(samples)
    plt.show()

config = config_mnist()
train(config)