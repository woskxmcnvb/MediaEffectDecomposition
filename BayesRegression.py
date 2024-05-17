import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from jax import random
import jax.numpy as jnp

import arviz as az

import numpyro
import numpyro.distributions as dist

from numpyro.infer import MCMC, NUTS, Predictive

from scipy.special import expit

def CompareModels(*args):
    models_dict = {m.name: m.ToArviZ() for m in args}
    compare_results = az.compare(models_dict, ic='waic')
    print(compare_results)
    az.plot_compare(compare_results)
    az.plot_forest(list(models_dict.values()), var_names=['beta'])
    plt.show()

class BernoulliRegression:
    name: str = None
    mcmc: MCMC = None
    pred_function = None
    fit_with_sample_size = None
    fit_with_data = {}
    seed = 127

    def __init__(self, name) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name

    """@staticmethod
    def Model(covs, y=None):
        if len(covs.shape) == 1: 
            covs = covs[:, jnp.newaxis]
        covs_dim = covs.shape[-1]

        intercept =   numpyro.sample("intercept", dist.Normal(0, 5))
        alpha = 20 * numpyro.sample("alpha", dist.Beta(1, 2).expand([covs_dim]).to_event(1))
        gamma = 10 * numpyro.sample("gamma", dist.Beta(1, 2).expand([covs_dim]).to_event(1))
        camp_beta = numpyro.sample("camp_beta", dist.Normal(0, 1).expand([covs_dim]).to_event(1))
        
        # hills transformation
        covs_pow_gamma = jnp.power(covs, gamma)
        covs_saturated = covs_pow_gamma / (covs_pow_gamma + jnp.power(alpha, gamma))
        log_prob = numpyro.deterministic("prob", intercept + (camp_beta * covs_saturated).sum(axis=1))

        numpyro.sample("obs", dist.BinomialLogits(log_prob), obs=y)"""

    """@staticmethod
    def Model(media_freq, non_media, y=None):
        intercept = numpyro.sample("intercept", dist.Normal(0, 5))
        
        if media_freq is None:
            media_part = 0
        else:
            if len(media_freq.shape) == 1:
                media_freq = media_freq[:, jnp.newaxis]
            media_dims = media_freq.shape[-1]
            
            alpha = 20 * numpyro.sample("alpha", dist.Beta(1, 2).expand([media_dims]).to_event(1))
            gamma = 10 * numpyro.sample("gamma", dist.Beta(1, 2).expand([media_dims]).to_event(1))
            media_beta = numpyro.sample("media_beta", dist.Normal(0, 1).expand([media_dims]).to_event(1))
            
            # hills transformation for media
            media_pow_gamma = jnp.power(media_freq, gamma)
            media_saturated = media_pow_gamma / (media_pow_gamma + jnp.power(alpha, gamma))
            media_part = (media_beta * media_saturated).sum(axis=1)

        if non_media is None:
            non_media_part = 0
        else:
            if len(non_media.shape) == 1:
                non_media = non_media[:, np.newaxis]
            non_media_dims = non_media.shape[-1]
            non_media_beta = numpyro.sample("non_media_beta", dist.Normal(0, 1).expand([non_media_dims]).to_event(1))
            non_media_part = (non_media * non_media_beta).sum(axis=1)
        
        log_prob = numpyro.deterministic("prob", intercept + media_part + non_media_part)
        numpyro.sample("obs", dist.BinomialLogits(log_prob), obs=y)"""

    """@staticmethod
    def ModelSplit_naive(media_freq, non_media, split_var, y=None):
        intercept_one = numpyro.sample("intercept_one", dist.Normal(0, 5))
        intercept_two = numpyro.sample("intercept_two", dist.Normal(0, 5))
        intercept = intercept_one + intercept_two * split_var
        
        if len(split_var.shape) == 1:
            split_var = split_var[:, jnp.newaxis]
        
        if media_freq is None:
            media_part = 0
        else:
            if len(media_freq.shape) == 1:
                media_freq = media_freq[:, jnp.newaxis]
            media_dims = media_freq.shape[-1]
            
            alpha = 20 * numpyro.sample("alpha", dist.Beta(1, 2).expand([media_dims]).to_event(1))
            gamma = 10 * numpyro.sample("gamma", dist.Beta(1, 2).expand([media_dims]).to_event(1))
            media_beta_one = numpyro.sample("media_beta_one", dist.Normal(0, 1).expand([media_dims]).to_event(1))
            media_beta_two = numpyro.sample("media_beta_two", dist.Normal(0, 1).expand([media_dims]).to_event(1))

            
            # hills transformation for media
            media_pow_gamma = jnp.power(media_freq, gamma)
            media_saturated = media_pow_gamma / (media_pow_gamma + jnp.power(alpha, gamma))
            media_beta = media_beta_one + media_beta_two * split_var
            media_part = (media_beta * media_saturated).sum(axis=1)

        if non_media is None:
            non_media_part = 0
        else:
            if len(non_media.shape) == 1:
                non_media = non_media[:, jnp.newaxis]
            non_media_dims = non_media.shape[-1]
            non_media_beta_one = numpyro.sample("non_media_beta_one", dist.Normal(0, 1).expand([non_media_dims]).to_event(1))
            non_media_beta_two = numpyro.sample("non_media_beta_two", dist.Normal(0, 1).expand([non_media_dims]).to_event(1))
            non_media_beta = non_media_beta_one + non_media_beta_two * split_var
            
            non_media_part = (non_media * non_media_beta).sum(axis=1)
        
        log_prob = numpyro.deterministic("prob", intercept + media_part + non_media_part)
        numpyro.sample("obs", dist.BinomialLogits(log_prob), obs=y)"""



    """@staticmethod
    def ModelSplit_index(media_freq, non_media, split_var, y=None):
        if split_var is None:
            n_groups = 1
        else:
            n_groups = len(np.unique(split_var))

        with numpyro.plate("sample_split", n_groups):
                intercept = numpyro.sample("intercept", dist.Normal(0, 5))
        
        if media_freq is None:
            media_part = 0
        else:
            if len(media_freq.shape) == 1:
                media_freq = media_freq[:, jnp.newaxis]
            media_dims = media_freq.shape[-1]

            alpha = 20 * numpyro.sample("alpha", dist.Beta(1, 2).expand([media_dims]).to_event(1))
            gamma = 10 * numpyro.sample("gamma", dist.Beta(1, 2).expand([media_dims]).to_event(1))
        
            with numpyro.plate("sample_split", n_groups):
                media_beta = numpyro.sample("media_beta", dist.Normal(0, 1).expand([media_dims]).to_event(1))
            
            # hills transformation for media
            media_pow_gamma = jnp.power(media_freq, gamma)
            media_saturated = media_pow_gamma / (media_pow_gamma + jnp.power(alpha, gamma))
            media_part = (media_beta[split_var] * media_saturated).sum(axis=1)

        if non_media is None:
            non_media_part = 0
        else:
            if len(non_media.shape) == 1:
                non_media = non_media[:, jnp.newaxis]
            non_media_dims = non_media.shape[-1]

            with numpyro.plate("sample_split", n_groups):
                non_media_beta = numpyro.sample("non_media_beta", dist.Normal(0, 1).expand([non_media_dims]).to_event(1))
            
            non_media_part = (non_media * non_media_beta[split_var]).sum(axis=1)


        log_prob = numpyro.deterministic("prob", intercept[split_var] + media_part + non_media_part)
        numpyro.sample("obs", dist.BinomialLogits(log_prob), obs=y)"""

    def __GetSampleSize(self, media_freq=None, non_media=None, split_var=None, y=None): 
        if media_freq: 
            return media_freq.shape[0]
        if non_media:
            return non_media.shape[0]
        if split_var:
            return split_var.shape[0]
        if y:
            return y.shape[0]
        if self.fit_with_sample_size:
            return self.fit_with_sample_size
        raise RecursionError("No sample size defined")
        

    def Model(self, media_freq=None, non_media=None, split_var=None, y=None):
        # media saturaion if in model
        if media_freq is not None: 
            alpha = 20 * numpyro.sample("alpha", dist.Beta(1, 2).expand([media_freq.shape[-1]]).to_event(1)) # dims: (media_dims) 
            gamma = 10 * numpyro.sample("gamma", dist.Beta(1, 2).expand([media_freq.shape[-1]]).to_event(1))
            media_pow_gamma = jnp.power(media_freq, gamma)
            media_freq = media_pow_gamma / (media_pow_gamma + jnp.power(alpha, gamma)) # dims: (sample, media_dims) 
        
        covs = jnp.column_stack([c for c in [media_freq, non_media] if (c is not None)] + [np.ones((self.__GetSampleSize(), 1))])

        if split_var is None:
            n_groups, split_var = 1, 0
        else:
            n_groups = len(np.unique(split_var))

        with numpyro.plate("sample_split", n_groups):
            betas = numpyro.sample("beta", dist.Normal(0, 1).expand([covs.shape[-1]]).to_event(1))

        log_prob = numpyro.deterministic("prob", (betas[split_var] * covs).sum(axis=1))
        numpyro.sample("obs", dist.BinomialLogits(log_prob), obs=y)

    """@staticmethod
    def ModelSplit_index_new(media_freq, non_media, split_var=None, y=None):
        if media_freq is not None: 
            alpha = 20 * numpyro.sample("alpha", dist.Beta(1, 2).expand([media_freq.shape[-1]]).to_event(1)) # dims: (media_dims) 
            gamma = 10 * numpyro.sample("gamma", dist.Beta(1, 2).expand([media_freq.shape[-1]]).to_event(1))
            media_pow_gamma = jnp.power(media_freq, gamma)
            media_saturated = media_pow_gamma / (media_pow_gamma + jnp.power(alpha, gamma)) # dims: (sample, media_dims) 
        
        covs = jnp.column_stack([c for c in [media_saturated, non_media] if (c is not None)])

        if split_var is None:
            n_groups, split_var = 1, 0
        else:
            n_groups = len(np.unique(split_var))

        mu_intercept = numpyro.sample("mu intercept", dist.Normal(0, 1))
        si_intercept = numpyro.sample("si intercept", dist.HalfCauchy(1))

        mu_beta = numpyro.sample("mu beta", dist.Normal(0, 1))
        si_beta = numpyro.sample("si beta", dist.HalfCauchy(1))

        with numpyro.plate("sample_split", n_groups):
            intercept = numpyro.sample("intercept", dist.Normal(mu_intercept, si_intercept))
            betas = numpyro.sample("media_beta", dist.Normal(mu_beta, si_beta).expand([covs.shape[-1]]).to_event(1))

        log_prob = numpyro.deterministic("prob", intercept[split_var] + (betas[split_var] * covs).sum(axis=1))
        numpyro.sample("obs", dist.BinomialLogits(log_prob), obs=y)"""


    def Fit(self, media: np.array, non_media: np.array, split: np.array, y: np.array, show_trace=False):
        self.fit_with_sample_size = y.shape[0]
        #self.fit_with_data['media'] = PrepareInput(media)
        #self.fit_with_data['non_media'] = PrepareInput(non_media)
        #self.fit_with_data['split'] = (None if split is None else split.copy())
        
        rng_key = random.PRNGKey(self.seed)
        self.mcmc = MCMC(NUTS(self.Model), num_warmup=1000, num_samples=2000)
        self.mcmc.run(rng_key, 
                      media_freq=media, #self.fit_with_data['media'], 
                      non_media=non_media, #self.fit_with_data['non_media'], 
                      split_var=split, #self.fit_with_data['split'], 
                      y=y)
        self.pred_function = Predictive(self.Model, posterior_samples=self.GetPosterior(drop_deterministic=True))
        if show_trace:
            az_data = self.ToArviZ()
            vars_to_plot = [v for v in ['alpha', 'gamma', 'beta'] if v in az_data['posterior']]
            az.plot_trace(az_data, vars_to_plot)
            plt.show()
        return self
    
    def ToArviZ(self) -> az.InferenceData:
        assert self.mcmc
        return az.from_numpyro(self.mcmc)
    
    def GetPosterior(self, drop_deterministic=True) -> dict:
        assert self.mcmc, "Run .Fit first"
        posterior = self.mcmc.get_samples().copy()
        if drop_deterministic:
            del posterior['prob']
        return posterior
    
    def PredictionFuncion(self):
        assert self.mcmc, "Run .Fit first"
        return self.pred_function

    def Contributions(self, media: np.array, non_media: np.array, split: np.array) -> np.array: 
        # возвращает вклады в разрезе base / non_media / media
        # по респондентам
        # dims = (resps,  base / non_media / media)
        assert self.mcmc, "Run .Fit first"
        
        # вклады, но на каждом уровне у нас пока не вклад, а накопленная сумма с самого начала
        contributions = []
        
        # base: нули для non-campaign и для campaign
        # dims: (N respondents)
        contributions.append(
            expit(self.pred_function(
                random.PRNGKey(127),
                media_freq=(None if media is None else np.zeros_like(media)), 
                non_media=(None if non_media is None else np.zeros_like(non_media)), 
                split_var=split
            )['prob']).mean(axis=0)
        )
        
        # non campaign: вернули значения для non-campaign
        if non_media is not None:
            contributions.append(
                expit(self.pred_function(
                    random.PRNGKey(127),
                    media_freq=(None if media is None else np.zeros_like(media)), 
                    non_media=non_media, 
                    split_var=split
                )['prob']).mean(axis=0)
            )

        # campaign: возвращаем значения для кампаний по очереди
        if media is not None:
            Xm = np.zeros_like(media)
            for i in range(Xm.shape[-1]):
                Xm[..., i] = media[..., i]
                contributions.append(
                    expit(self.pred_function(
                        random.PRNGKey(127),
                        media_freq=Xm, 
                        non_media=non_media, 
                        split_var=split
                    )['prob']).mean(axis=0)
                )
        
        # собираем все в одну таблицу. dims: (N respondents, компоненты модели)
        # находим послойную разницу, чтобы найти именно вклады
        return np.diff(np.column_stack(contributions), axis=1, prepend=0)
