import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

import arviz as az

import numpyro
import numpyro.distributions as dist

numpyro.set_host_device_count(4)

from numpyro.infer import MCMC, NUTS, Predictive

@jax.jit
def _media_preprocess(X: jax.Array, alpha: jax.Array, gamma: jax.Array, beta=None) -> jax.Array:
    data_pow_alpha = jnp.power(X, alpha)
    return data_pow_alpha / (data_pow_alpha + jnp.power(gamma, alpha))

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
        raise ValueError("No sample size defined")
        
    def Model(self, media_freq: jax.Array=None, non_media: jax.Array=None, split_var: jax.Array=None, y: jax.Array=None):
        # media saturaion if in model
        if media_freq is not None: 
            alpha = 20 * numpyro.sample("alpha", dist.Beta(1, 2).expand([media_freq.shape[-1]]).to_event(1)) # dims: (media_dims) 
            gamma = 10 * numpyro.sample("gamma", dist.Beta(1, 2).expand([media_freq.shape[-1]]).to_event(1)) # dims: (media_dims) 
            media_freq = _media_preprocess(media_freq, gamma, alpha) # dims: (sample, media_dims) 
        
        covs = jnp.column_stack([c for c in [media_freq, non_media] if (c is not None)] + [jnp.ones((self.__GetSampleSize(), 1))])

        with numpyro.plate("sample_split", self.num_sample_splits):
            betas = numpyro.sample("beta", dist.Normal(0, 1).expand([covs.shape[-1]]).to_event(1))

        log_prob = numpyro.deterministic("prob", (betas[split_var] * covs).sum(axis=1))
        numpyro.sample("obs", dist.BinomialLogits(log_prob), obs=y)

    def Fit(self, media: jax.Array, non_media: jax.Array, split: jax.Array, y: jax.Array, show_trace=False, num_samples=2000, num_chains=1):
        self.fit_with_sample_size = y.shape[0]
        if split is None:    
            self.num_sample_splits = 1
            split = jnp.array(0)
        else:
            self.num_sample_splits = split.max().item() + 1
        
        rng_key = jax.random.PRNGKey(self.seed)
        self.mcmc = MCMC(NUTS(self.Model), num_warmup=1000, num_samples=num_samples, num_chains=num_chains)
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

    def Predictions(self, media: jax.Array, non_media: jax.Array, split: jax.Array) -> jax.Array:
        return jax.scipy.special.expit(self.pred_function(
                jax.random.PRNGKey(127),
                media_freq=media, 
                non_media=non_media, 
                split_var=split
            )['prob']).mean(axis=0)
    
    def Contributions(self, media: jax.Array, non_media: jax.Array, split: jax.Array) -> jax.Array: 
        # возвращает вклады в разрезе base / non_media / media
        # по респондентам
        # dims = (resps,  base / non_media / media)
        assert self.mcmc, "Run .Fit first"
        
        # вклады, но на каждом уровне у нас пока не вклад, а накопленная сумма с самого начала
        contributions = []
        
        # base: нули для non-campaign и для campaign
        # dims: (N respondents)
        contributions.append(
            jax.scipy.special.expit(self.pred_function(
                jax.random.PRNGKey(127),
                media_freq=(None if media is None else jnp.zeros_like(media)), 
                non_media=(None if non_media is None else jnp.zeros_like(non_media)), 
                split_var=split
            )['prob']).mean(axis=0)
        )
        
        # non campaign: вернули значения для non-campaign
        if non_media is not None:
            contributions.append(
                jax.scipy.special.expit(self.pred_function(
                    jax.random.PRNGKey(127),
                    media_freq=(None if media is None else jnp.zeros_like(media)), 
                    non_media=non_media, 
                    split_var=split
                )['prob']).mean(axis=0)
            )

        # campaign: возвращаем значения для кампаний по очереди
        if media is not None:
            Xm = jnp.zeros_like(media)
            for i in range(Xm.shape[-1]):
                #Xm[..., i] = media[..., i]
                Xm.at[..., i].set(media[..., i])
                contributions.append(
                    jax.scipy.special.expit(self.pred_function(
                        jax.random.PRNGKey(127),
                        media_freq=Xm, 
                        non_media=non_media, 
                        split_var=split
                    )['prob']).mean(axis=0)
                )
        
        # собираем все в одну таблицу. dims: (N respondents, компоненты модели)
        # находим послойную разницу, чтобы найти именно вклады
        return jnp.diff(jnp.column_stack(contributions), axis=1, prepend=0)
