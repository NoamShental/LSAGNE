import torch
import torch.utils.data

from torch.distributions import Normal, Categorical, MultivariateNormal, kl_divergence, Bernoulli


# TODO support frozen epsilon, in order to get same encoding during post training evaluations

def compute_log_alpha(mu, logvar):
    # clamp because dropout rate p in 0-99%, where p = alpha/(alpha+1)
    return (logvar - 2 * torch.log(torch.abs(mu) + 1e-8)).clamp(min=-8, max=8)


def compute_logvar(mu, log_alpha):
    return log_alpha + 2 * torch.log(torch.abs(mu) + 1e-8)

def compute_ll(p, x):
    """
    :param p: Normal: p.loc.shape = (n_obs, n_feats)
    :param x:
    :return: log-likelihood compatible with the distribution p
    """
    if isinstance(p, Normal):
        ll = p.log_prob(x).sum(1, keepdims=True)
    elif isinstance(p, Categorical):
        ll = p.log_prob(x.view(-1))
    elif isinstance(p, MultivariateNormal):
        ll = p.log_prob(x).unsqueeze(1)  # MultiVariate already sums over dimensions
    else:
        raise NotImplementedError

    return ll.mean(0)

def KL_log_uniform(mu, logvar):
    """
    Paragraph 4.2 from:
    Variational Dropout Sparsifies Deep Neural Networks
    Molchanov, Dmitry; Ashukha, Arsenii; Vetrov, Dmitry
    https://arxiv.org/abs/1701.05369
    https://github.com/senya-ashukha/variational-dropout-sparsifies-dnn/blob/master/KL%20approximation.ipynb
    """
    log_alpha = compute_log_alpha(mu, logvar)
    k1, k2, k3 = 0.63576, 1.8732, 1.48695
    neg_KL = k1 * torch.sigmoid(k2 + k3 * log_alpha) - 0.5 * torch.log1p(torch.exp(-log_alpha)) - k1
    return -neg_KL

def compute_kl(p1, p2=None, sparse=False):
    """
    :param p1: Normal distribution with p1.loc.shape = (n_obs, n_lat_dims)
    :param p2: same as p1
    :param sparse:
    :return: scalar value
    """
    if sparse:
        kl = KL_log_uniform(mu=p1.loc, logvar=p1.scale.pow(2).log())
    else:
        kl = kl_divergence(p1, p2)

    return kl.sum(1, keepdims=True).mean(0)



class SparseVaeNet(torch.nn.Module):

    def __init__(
            self,
            latent_dim,
            input_dim,
            # encode_dims,
            # decode_dims,
            # beta=1.0,  # for beta-VAE (kl weight coefficient)
            sparse=False,
            log_alpha=None,
            noise_init_logvar=-3,
            noise_fixed=False,
            # bias_enc=True,
            # bias_dec=True,
    ):
        super().__init__()
        self.input_dim = input_dim
        # self.beta = beta
        self.latent_dim = latent_dim
        self.sparse = sparse
        self.log_alpha = log_alpha
        self.noise_init_logvar = noise_init_logvar
        self.noise_fixed = noise_fixed
        # self.bias_enc = bias_enc
        # self.bias_dec = bias_dec
        # self.encode_dims = encode_dims
        # self.decode_dims = decode_dims
        self.act_fun = torch.nn.LeakyReLU
        self.hidden_dim = (self.input_dim + self.latent_dim) // 2

        self.init_encoder()
        self.init_decoder()

    def init_encoder(self):
        # Encoders: random initialization of weights
        self.W_mu = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.hidden_dim),
            self.act_fun(),
            # torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            # self.act_fun(),
            torch.nn.Linear(self.hidden_dim, self.latent_dim),
        )
        if self.sparse:
            if self.log_alpha is None:
                self.log_alpha = torch.nn.Parameter(torch.FloatTensor(1, self.latent_dim).normal_(0, 0.01))
        else:
            self.W_logvar = torch.nn.Sequential(
                torch.nn.Linear(self.input_dim, self.hidden_dim),
                self.act_fun(),
                # torch.nn.Linear(self.hidden_dim, self.hidden_dim),
                # self.act_fun(),
                torch.nn.Linear(self.hidden_dim, self.latent_dim),
            )

    def init_decoder(self):
        # Decoders: random initialization of weights
        self.W_out = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, self.hidden_dim),
            self.act_fun(),
            # torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            # self.act_fun(),
            torch.nn.Linear(self.hidden_dim, self.input_dim),
        )
        tmp_noise_par = torch.FloatTensor(1, self.input_dim).fill_(self.noise_init_logvar)
        if self.noise_fixed:
            self.W_out_logvar = torch.nn.Parameter(data=tmp_noise_par, requires_grad=False)
        else:
            self.W_out_logvar = torch.nn.Parameter(data=tmp_noise_par, requires_grad=True)

        del tmp_noise_par

    def encode(self, x):
        """
        :param x: list of datasets (Obs x Feats)
        :return: list of encoded distributions (one list element per dataset)
        """
        mu = self.W_mu(x)
        if not self.sparse:
            logvar = self.W_logvar(x)
        else:
            logvar = compute_logvar(mu, self.log_alpha)

        return Normal(loc=mu, scale=logvar.exp().pow(0.5))

    def decode(self, z):
        pi = Normal(
            loc=self.W_out(z),
            scale=self.W_out_logvar.exp().pow(0.5)
        )
        return pi

    def forward(self, x, y=None):
        q = self.encode(x)
        posterior = q

        if self.training:
            z = posterior.rsample()
        else:
            z = posterior.loc
        p = self.decode(z)

        fwd_ret = {
            'y': x if y is None else y,
            'z': z,
            'posterior': posterior,
            'p': p,
        }

        return fwd_ret

    def compute_kl(self, posterior):

        return compute_kl(p1=posterior, p2=Normal(0, 1), sparse=self.sparse)

    def loss_function(self, fwd_ret, beta):
        y = fwd_ret['y']
        posterior = fwd_ret['posterior']
        p = fwd_ret['p']

        kl = (beta * self.compute_kl(posterior)).squeeze()
        ll = compute_ll(p, y).squeeze()

        total = kl - ll

        losses = {
            'total': total,
            'kl': kl,
            'll': ll,
        }

        return losses

    @staticmethod
    def p_to_prediction(p):

        if isinstance(p, list):
            return [SparseVaeNet.p_to_prediction(_) for _ in p]

        if isinstance(p, Normal):
            pred = p.loc
        elif isinstance(p, Categorical):
            pred = p.logits.argmax(dim=1)
        elif isinstance(p, Bernoulli):
            pred = p.probs
        else:
            raise NotImplementedError

        return pred

    @staticmethod
    def p_to_expected_value(p):

        if isinstance(p, list):
            return [SparseVaeNet.p_to_prediction(_) for _ in p]

        if isinstance(p, Normal):
            pred = p.loc
        elif isinstance(p, Categorical):
            pred = p.probs  #  expected value is not defined for a Cat distrib. Careful when using this method
        elif isinstance(p, Bernoulli):
            pred = p.probs
        else:
            raise NotImplementedError

        return pred

    def reconstruct(self, x, sample=False):
        with torch.no_grad():
            q = self.encode(x)
            posterior = q
            if sample:
                z = posterior.sample()
            else:
                z = posterior.loc
            p = self.decode(z)

        return self.p_to_prediction(p)

    def predict(self, *args, **kwargs):
        return self.reconstruct(*args, **kwargs)

    def generate(self):

        if self.sparse:
            raise NotImplementedError

        z = Normal(torch.zeros(1, self.latent_dim), 1).sample()
        p = self.decode(z)

        if isinstance(p, Normal):
            return p.loc
        elif isinstance(p, Categorical):
            return p.logits.argmax(dim=1)
        else:
            raise NotImplementedError

    @property
    def dropout(self):
        if self.sparse:
            alpha = torch.exp(self.log_alpha.detach())
            return alpha / (alpha + 1)
        else:
            raise NotImplementedError

    def extra_repr(self) -> str:
        extrapars = ['W_out_logvar', 'log_alpha']
        ret = ''
        for ep in extrapars:
            if isinstance(getattr(self, ep), torch.nn.Parameter):
                ret += f'({ep}): Parameter(shape={getattr(self, ep).shape})\n'

        return ret.rstrip('\n')
