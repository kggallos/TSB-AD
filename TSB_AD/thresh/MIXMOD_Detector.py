# -*- coding: utf-8 -*-
# Author: Konstantinos Gallos <kggallos@gmail.com>
# License: Apache-2.0 License
"""
This code is adapted from [pythresh] by [KulikDM]
Original source: [https://github.com/KulikDM/pythresh]
"""

from itertools import combinations
import pandas as pd
import numpy as np
import scipy.stats as stats
import argparse, time
import scipy.optimize as opt
from scipy.special import digamma

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank

from .thresholding_utils import check_scores, normalize

class MIXMOD():
    r"""MIXMOD class for the Normal & Non-Normal Mixture Models thresholder.

       Use normal & non-normal mixture models to find a non-parametric means
       to threshold scores generated by the decision_scores, where outliers
       are set to any value beyond the posterior probability threshold
       for equal posteriors of a two distribution mixture model.
       See :cite:`veluw2023mixmod` for details

       Parameters
       ----------

       method : str, optional (default='mean')
            Method to evaluate selecting the best fit mixture model. Default
            'mean' sets this as the closest mixture models to the mean of the posterior
            probability threshold for equal posteriors of a two distribution mixture model
            for all fits. Setting 'ks' uses the two-sample Kolmogorov-Smirnov test for
            goodness of fit.

       tol : float, optional (default=1e-5)
            Tolerance for convergence of the EM fit

       max_iter : int, optional (default=250)
            Max number of iterations to run EM during fit

       random_state : int, optional (default=1234)
            Random seed for the random number generators of the thresholders. Can also
            be set to None.

       Attributes
       ----------

        threshold_ : float
            The threshold value that separates inliers from outliers.

        decision_scores_: ndarray of shape (n_samples,) #TODO
            Not actually used, present for API consistency by convention.
            It contains 0s and 1s because this is a thresholding method.

        mixture_ : fitted mixture model class of the selected model used for thresholding

       Notes
       -----

       The Normal & Non-Normal Mixture Models thresholder is constructed by searching
       all possible two component combinations of the following distributions (normal,
       lognormal, uniform, student's t, pareto, laplace, gamma, fisk, and exponential).
       Each two component combination mixture is is fit to the data using
       expectation-maximization (EM) using the corresponding maximum likelihood estimation
       functions (MLE) for each distribution. From this the posterior probability threshold
       is obtained as the point where equal posteriors of a two distribution mixture model
       exists.

    """

    def __init__(self, method='mean', tol=1e-5, max_iter=250,
                  random_state=1234, normalize=True):

        dists = [stats.expon, stats.fisk, stats.gamma, stats.laplace, stats.t,
                 stats.lognorm, stats.norm, stats.uniform, stats.pareto]

        self.combs = list(combinations(dists, 2))

        self.method = method
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.normalize = normalize

    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        pass

    def decision_function(self, X):
        """    
        Not used, present for API consistency by convention.
        """
        pass
        
    def predict(self, X):
        """
        Predict anomalies in a batch of data points.

        Parameters
        ----------
        X : numpy array of shape (n_samples,)
            The input data points.

        Returns
        -------
        preds : numpy array of shape (n_samples,)
            Predictions (1 for anomaly, 0 for normal).
        """
        n_samples, n_features = X.shape

        X = check_scores(X, random_state=self.random_state)

        if self.normalize: X = normalize(X)

        mix_scores = X + 1

        # Create a KDE of the decision scores
        points = max(len(X)*3, 1000)
        x = np.linspace(1, 2, points)

        if self.method == 'ks':
            kde = stats.gaussian_kde(mix_scores, bw_method=0.1)

        mixtures = []
        scores = []
        crossing = []

        # Fit all possible combinations of dists to the scores
        for comb in self.combs:

            mix = MixtureModel([comb[0], comb[1]], self.tol, self.max_iter)
            try:
                mix.fit(mix_scores)
            except Exception:
                continue

            # Get the posterior probability threshold for equal posteriors
            y = mix.posterior(x)

            y_diff = np.sign(y[1] - y[0])
            crossings = np.where(np.diff(y_diff) != 0)[0]

            if len(crossings) == 0:
                continue

            # Evaluate the fit
            if self.method == 'ks':
                stat, _ = stats.ks_2samp(kde(x), mix.pdf(x))
            else:
                stat = x[crossings[-1]]

            mixtures.append(mix)
            scores.append(stat)
            crossing.append(crossings[-1])

        # Use the highest fit score
        if self.method == 'ks':
            max_stat = np.argmax(scores)
        else:
            diff = np.mean(scores) - np.array(scores)
            max_stat = np.argmin(np.abs(diff))

        mixture = mixtures[max_stat]
        cross = crossing[max_stat]

        limit = x[cross] if len(crossing) > 0 else 2

        self.threshold_ = limit - 1
        self.mixture_ = mixture

        # return cut(mix_scores, limit) #TODO remove this

        preds = np.zeros(n_samples, dtype=int)
        preds[mix_scores >= limit] = 1

        return preds
    
    
# This portion of code is derived from the GitHub repository marcsingleton/mixmod
# which is licensed under the MIT License. Copyright (c) 2022 Marc Singleton


class MixtureModel:
    """Class for performing calculations with mixture models."""

    def __init__(self, components, tol, max_iter):

        params = [{} for _ in components]
        weights = [1 / len(components) for _ in components]

        self.components = components
        self.params = params
        self.weights = weights
        self.tol = tol
        self.max_iter = max_iter
        self.converged = False

    def fit(self, data):
        """Fit the free parameters of the mixture model with EM algorithm."""

        weights_opt = self.weights.copy()
        params_opt = []

        # Get closed-form estimator for initial estimation
        for component, param in zip(self.components, self.params):

            cfe = MLES().cfes[component.name]
            param_init = {**cfe(data), **param}
            params_opt.append(param_init)

        ll0 = self._get_loglikelihood(data, self.components,
                                      params_opt, weights_opt)

        # Apply Expectation-Maximization
        for numiter in range(1, self.max_iter + 1):

            expts = self._get_posterior(
                data, self.components, params_opt, weights_opt)
            weights_opt = expts.sum(axis=1) / expts.sum()

            for component, param_opt, expt in zip(self.components, params_opt, expts):

                # Get MLE function and update parameters
                mle = MLES().mles[component.name]
                opt = mle(data, expt=expt, initial=param_opt)
                param_opt.update(opt)

            ll = self._get_loglikelihood(
                data, self.components, params_opt, weights_opt)

            # Test numerical exception then convergence
            if np.isnan(ll) or np.isinf(ll):
                break
            if abs(ll - ll0) < self.tol:
                self.converged = True
                break

            ll0 = ll

        self.params = params_opt
        self.weights = weights_opt.tolist()

        return numiter, ll

    def loglikelihood(self, data):
        """Return log-likelihood of data according to mixture model."""

        return self._get_loglikelihood(data, self.components, self.params, self.weights)

    def posterior(self, data):
        """Return array of posterior probabilities of data for each component of mixture model."""

        return self._get_posterior(data, self.components, self.params, self.weights)

    def pdf(self, x, component='sum'):
        """Return pdf evaluated at x."""

        ps = self._get_pdfstack(x, self.components, self.params, self.weights)
        return ps.sum(axis=0)

    def _get_loglikelihood(self, data, components, params, weights):
        """Return log-likelihood of data according to mixture model."""

        p = 0
        model_params = zip(components, params, weights)
        for component, param, weight in model_params:
            pf = getattr(component, 'pdf')
            p += weight * pf(data, **param)
        return np.log(p).sum()

    def _get_posterior(self, data, components, params, weights):
        """Return array of posterior probabilities of data for each component of mixture model."""

        ps = self._get_pdfstack(data, components, params, weights)

        return ps / ps.sum(axis=0)

    def _get_pdfstack(self, data, components, params, weights):
        """Return array of pdfs evaluated at data for each component of mixture model."""

        model_params = zip(components, params, weights)
        ps = [weight * component.pdf(data, **param)
              for component, param, weight in model_params]

        return np.stack(ps, axis=0)


class MLES:
    """Class of maximum likelihood estimation functions."""

    def __init__(self):

        pass

    def create_fisk_scale(data, expt=None):
        expt = np.full(len(data), 1) if expt is None else expt

        def fisk_scale(scale):
            # Compute sums
            e = expt.sum()
            q = ((expt * data) / (scale + data)).sum()

            return 2 * q - e

        return fisk_scale

    def create_fisk_shape(data, expt=None, scale=1):
        expt = np.full(len(data), 1) if expt is None else expt

        def fisk_shape(c):
            # Compute summands
            r = data / scale
            s = 1 / c + np.log(r) - 2 * np.log(r) * r ** c / (1 + r ** c)

            return (expt * s).sum()

        return fisk_shape

    def create_gamma_shape(data, expt=None):
        expt = np.full(len(data), 1) if expt is None else expt

        def gamma_shape(a):
            # Compute sums
            e = expt.sum()
            ed = (expt * data).sum()
            elogd = (expt * np.log(data)).sum()

            return elogd - e * np.log(ed / e) + e * (np.log(a) - digamma(a))

        return gamma_shape

    def mm_fisk(data, expt=None, **kwargs):
        """Method of moment estimator for a fisk distribution."""

        expt = np.full(len(data), 1) if expt is None else expt
        ests = {}

        # Moments
        logdata = np.log(data)
        m1 = (logdata * expt).sum() / expt.sum()
        m2 = (logdata ** 2 * expt).sum() / expt.sum()

        # Estimators
        ests['c'] = np.pi / np.sqrt(3 * (m2 - m1 ** 2))
        ests['scale'] = np.exp(m1)

        return ests

    def mm_gamma(data, expt=None, **kwargs):
        """Method of moment estimator for a gamma distribution."""

        expt = np.full(len(data), 1) if expt is None else expt
        ests = {}

        # Moments
        m1 = (data * expt).sum() / expt.sum()
        m2 = (data ** 2 * expt).sum() / expt.sum()

        # Estimators
        ests['a'] = m1 ** 2 / (m2 - m1 ** 2)
        ests['scale'] = (m2 - m1 ** 2) / m1

        return ests

    def mle_expon(data, expt=None, **kwargs):
        """MLE for an exponential distribution."""

        expt = np.full(len(data), 1) if expt is None else expt
        ests = {}

        # Scale parameter estimation
        e = expt.sum()
        ed = (expt * data).sum()
        scale = ed / e
        ests['scale'] = scale

        return ests

    def mle_fisk(data, expt=None, initial=None):
        """MLE for a fisk distribution."""

        expt = np.full(len(data), 1) if expt is None else expt
        initial = MLES.mm_fisk(data) if initial is None else initial
        ests = {}

        # Scale parameter estimation
        fisk_scale = MLES.create_fisk_scale(data, expt)
        scale = opt.newton(fisk_scale, initial['scale'])
        ests['scale'] = scale

        # Shape parameter estimation
        fisk_shape = MLES.create_fisk_shape(data, expt, scale)
        c = opt.newton(fisk_shape, initial['c'])
        ests['c'] = c

        return ests

    def mle_gamma(data, expt=None, initial=None):
        """MLE for a gamma distribution."""

        expt = np.full(len(data), 1) if expt is None else expt
        initial = MLES.mm_gamma(data) if initial is None else initial
        ests = {}

        # Shape parameter estimation
        gamma_shape = MLES.create_gamma_shape(data, expt)
        try:
            a = opt.newton(gamma_shape, initial['a'])
        except ValueError:
            lower = initial['a'] / 2
            upper = initial['a'] * 2
            while np.sign(gamma_shape(lower)) == np.sign(gamma_shape(upper)):
                lower /= 2
                upper *= 2
            a = opt.brentq(gamma_shape, lower, upper)
        ests['a'] = a

        # Scale parameter estimation
        scale = (expt * data).sum() / (a * expt.sum())
        ests['scale'] = scale

        return ests

    def mle_laplace(data, expt=None, **kwargs):
        """MLE for a laplace distribution."""

        expt = np.full(len(data), 1) if expt is None else expt[data.argsort()]
        data = np.sort(data)
        ests = {}

        # Location parameter estimation
        cm = expt.sum() / 2
        e_cum = expt.cumsum()
        idx = np.argmax(e_cum > cm)

        if data[idx] == data[idx - 1]:
            loc = data[idx]
        else:
            m = (e_cum[idx] - e_cum[idx - 1]) / (data[idx] - data[idx - 1])
            b = e_cum[idx] - m * data[idx]
            loc = (cm - b) / m
        ests['loc'] = loc

        # Scale parameter estimation
        e = expt.sum()
        d_abserr = abs(data - loc)
        scale = (expt * d_abserr).sum() / e
        ests['scale'] = scale

        return ests

    def mle_lognorm(data, expt=None, **kwargs):
        """MLE for a log-normal distribution."""

        expt = np.full(len(data), 1) if expt is None else expt
        ests = {}

        # Scale parameter estimation
        e = expt.sum()
        elogd = (expt * np.log(data)).sum()
        scale = np.exp(elogd / e)
        ests['scale'] = scale

        # Shape parameter estimation
        e = expt.sum()
        logd_sqerr = (np.log(data) - np.log(scale)) ** 2
        s = np.sqrt((expt * logd_sqerr).sum() / e)
        ests['s'] = s

        return ests

    def mle_norm(data, expt=None, **kwargs):
        """MLE for a normal distribution."""

        expt = np.full(len(data), 1) if expt is None else expt
        ests = {}

        # Location parameter estimation
        e = expt.sum()
        ed = (expt * data).sum()
        loc = ed / e
        ests['loc'] = loc

        # Scale parameter estimation
        e = expt.sum()
        d_sqerr = (data - loc) ** 2
        scale = np.sqrt((expt * d_sqerr).sum() / e)
        ests['scale'] = scale

        return ests

    def mle_pareto(data, expt=None, **kwargs):
        """MLE for a pareto distribution."""

        expt = np.full(len(data), 1) if expt is None else expt
        ests = {}

        # Scale parameter estimation
        scale = min(data)
        ests['scale'] = scale

        # Shape parameter estimation
        e = expt.sum()
        elogd = (expt * np.log(data)).sum()
        b = e / (elogd - e * np.log(scale))
        ests['b'] = b

        return ests

    def mle_uniform(data, **kwargs):
        """MLE for a uniform distribution."""

        ests = {}

        # Location parameter estimation
        loc = min(data)
        ests['loc'] = loc

        # Scale parameter estimation
        scale = max(data) - loc
        ests['scale'] = scale

        return ests

    def mle_t(data, expt=None, **kwargs):
        """MLE for an student-t distribution."""

        expt = np.full(len(data), 1) if expt is None else expt
        ests = {}

        # Location parameter estimation
        e = expt.sum()
        ed = (expt * data).sum()
        loc = ed / e
        ests['loc'] = loc

        # Scale parameter estimation
        e = expt.sum()
        w_data = data - loc
        scale = np.sqrt((expt * w_data**2).sum() / e)
        ests['scale'] = scale

        # Effective degrees of freedom estimation
        w_sum_squares = (expt**2).sum()
        w_sum = expt.sum()
        df = w_sum**2 / w_sum_squares
        ests['df'] = df

        return ests

    mles = {'expon': mle_expon,
            'fisk': mle_fisk,
            'gamma': mle_gamma,
            'laplace': mle_laplace,
            'lognorm': mle_lognorm,
            'norm': mle_norm,
            'pareto': mle_pareto,
            'uniform': mle_uniform,
            't': mle_t}

    cfes = {**mles,
            'fisk': mm_fisk,
            'gamma': mm_gamma}


if __name__ == '__main__':

    Start_T = time.time()
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Running MIXMOD')
    parser.add_argument('--filename', type=str, default='001_NAB_id_1_Facility_tr_1007_1st_2014.csv')
    parser.add_argument('--data_direc', type=str, default='Datasets/TSB-AD-U/')
    parser.add_argument('--AD_Name', type=str, default='MIXMOD')
    args = parser.parse_args()

    # multivariate
    # parser.add_argument('--filename', type=str, default='057_SMD_id_1_Facility_tr_4529_1st_4629.csv')
    # parser.add_argument('--data_direc', type=str, default='Datasets/TSB-AD-M/')

    Custom_AD_HP = {
        'random_state': 1234,   # not related to method itself, but to formatting input
        'method': 'mean',
        'tol': 1e-5,
        'max_iter': 250,
    }

    df = pd.read_csv(args.data_direc + args.filename).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()
    print('data: ', data.shape)
    print('label: ', label.shape)

    slidingWindow = find_length_rank(data, rank=1)

    Custom_AD_HP['method'] = 'ks'
    clf = MIXMOD(**Custom_AD_HP)
    output = clf.predict(data)
    pred = output
    evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow, pred=pred)
    print('Evaluation Result: ', evaluation_result)
