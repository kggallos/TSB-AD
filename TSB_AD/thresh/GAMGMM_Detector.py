# -*- coding: utf-8 -*-
# Author: Konstantinos Gallos <kggallos@gmail.com>
# License: Apache-2.0 License
"""
This code is adapted from [pythresh] by [KulikDM]
Original source: [https://github.com/KulikDM/pythresh]
"""

import warnings
import pandas as pd
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import beta, dirichlet, multivariate_normal, wishart
from sklearn.decomposition import TruncatedSVD
from sklearn.mixture import BayesianGaussianMixture
from sklearn.utils import check_array
import argparse, time

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.models.base import BaseDetector



from .thresholding_utils import check_scores, normalize

class GAMGMM(BaseDetector):
    r"""GAMGMM class for gammaGMM thresholder.

       Use a Bayesian method for estimating the posterior distribution
       of the contamination factor (i.e., the proportion of anomalies)
       for a given unlabeled dataset. The threshold is set such
       that the proportion of predicted anomalies equals the
       contamination factor. See :cite:`perini2023gamgmm` for details.

       Parameters
       ----------

       n_contaminations : int, optional (default=1000)
            number of samples to draw from the contamination posterior distribution

       n_draws : int, optional (default=50)
            number of samples simultaneously drawn from each DPGMM component

       p0 : float, optional (default=0.01)
            probability that no anomalies are in the data

       phigh : float, optional (default=0.01)
            probability that there are more than high_gamma anomalies

       high_gamma : float, optional (default=0.15)
            sensibly high number of anomalies that has low probability to occur

       gamma_lim : float, optional (default=0.5)
            Upper gamma/proportion of anomalies limit

       K : int, optional (default=100)
            number of components for DPGMM used to approximate the Dirichlet Process

       skip : bool, optional (default=False)
            skip optimal hyperparameter test (this may return a sub-optimal solution)

       steps : int, optional (default=100)
            number of iterations to test for optimal hyperparameters

       random_state : int, optional (default=1234)
            Random seed for the random number generators of the thresholders. Can also
            be set to None.

       verbose : bool, optional (default=False)
            20 iterations step printout of the DPGMM process

       Attributes
       ----------

        threshold_ : float
            The threshold value that separates inliers from outliers.

        decision_scores_: ndarray of shape (n_samples,) #TODO
            Not actually used, present for API consistency by convention.
            It contains 0s and 1s because this is a thresholding method.
       Notes
       -----

       This implementation deviates from that in :cite:`perini2023gamgmm` only
       in the post-processing page. These deviations include: if a single outlier
       detector likelihood score set is passed a dummy score set of zeros will be
       added such that GAMGMM method can function correctly, if multiple outlier
       detector likelihood score sets are passed a TruncatedSVD 1D decomposed will
       be thresholded but not used to determine the gamma contamination. However,
       if you wish to follow the original implementation please go to
       `GammaGMM <https://github.com/Lorenzo-Perini/GammaGMM>`_

    """

    def __init__(self, random_state=1234, normalize=True):
        super().__init__()
        self.random_state = random_state
        self.normalize = normalize

    def __init__(self,
                 n_contaminations=1000,
                 n_draws=50,
                 p0=0.01,
                 phigh=0.01,
                 high_gamma=0.15,
                 gamma_lim=0.5,
                 K=100,
                 skip=False,
                 steps=100,
                 random_state=1234,
                 verbose=False,
                 normalize=True):

        self.n_contaminations = n_contaminations
        self.n_draws = n_draws
        self.p0 = p0
        self.phigh = phigh
        self.high_gamma = high_gamma
        self.gamma_lim = gamma_lim
        self.K = K
        self.skip = skip
        self.steps = steps
        self.random_state = random_state
        self.verbose = verbose
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

        if (np.asarray(X).ndim == 2) & (np.atleast_2d(X).shape[1] > 1):

            X = check_array(X, ensure_2d=True)
            score_space = self._augment_space(X)

            # Decompose decision scores to 1D for thresholding
            decomp = TruncatedSVD(n_components=1,
                                  random_state=self.random_state)

            X = decomp.fit_transform(normalize(X))

        else:

            X = check_array(X, ensure_2d=False).squeeze()
            X = np.atleast_2d(X).T

            score_space = self._augment_space(X).squeeze()
            score_space = np.vstack(
                [score_space, np.zeros_like(score_space)]).T

            warnings.warn('''Using one set of outlier detection likelihood scores may
                             lead to a suboptimal or no solution. Please consider increasing
                             the the number of outlier detection score sets''')

        X = normalize(X.squeeze())
        self.dscores_ = X.copy()

        # Compute the gamma posterior and threshold
        gamma_posterior_sample = self._compute_gamma_posterior(score_space)

        gamma_mean = np.mean(gamma_posterior_sample)

        self.threshold_ = np.percentile(X, 100 * (1 - gamma_mean))

        labels = (X > self.threshold_).astype('int').ravel() #TODO CHANGE TO preds
        return labels


    def _compute_gamma_posterior(self, decision):

        # Handle overflow of components versus samples.
        self.K = decision.shape[0] - \
            1 if np.shape(decision)[0] < self.K else self.K

        itv = 0
        random_start = 1234 if not self.random_state else self.random_state

        # Repeat the loop until a valid gamma sample is found else return unassigned gammas
        while True:

            whileseed = random_start + 100*itv
            np.random.seed(whileseed)

            # Fit the DPGMM
            bgm = BayesianGaussianMixture(weight_concentration_prior_type='dirichlet_process', n_components=self.K,
                                          weight_concentration_prior=0.01, max_iter=1500, random_state=whileseed,
                                          verbose=self.verbose, verbose_interval=20, reg_covar=1e-4).fit(decision)

            # Drop components with less than 2 instances assigned
            filter_idx = np.where(bgm.weight_concentration_[0] >= 2)[0]

            tot_concentration = np.sum(bgm.weight_concentration_[0])

            partial_concentration = np.sum(
                bgm.weight_concentration_[0][filter_idx])

            means = bgm.means_[filter_idx]
            covariances = bgm.covariances_[filter_idx, :, :]

            alphas = bgm.weight_concentration_[0][filter_idx]
            mean_precs = bgm.mean_precision_[filter_idx]
            dgf = bgm.degrees_of_freedom_[filter_idx]

            idx_sortcomponents, meanstd = self._order_components(
                means, mean_precs, covariances, dgf)

            # Redistribute the lost mass (after cutting off some components)
            alphas = alphas[idx_sortcomponents] + \
                (tot_concentration-partial_concentration)/len(filter_idx)

            # Solve the optimization problem to find the hyperparameters delta and tau
            res = least_squares(self._find_delta_tau, x0=(-2, 1), args=(meanstd, alphas),
                                bounds=((-50, -1), (0, 50)))

            delta, tau = res.x

            # Check that delta and tau are properly found, allowing for a 10% error
            p0Est, phighEst = self._check_delta_tau(
                delta, tau, meanstd, alphas)

            if (((p0Est < self.p0*1.1) & (p0Est > self.p0*0.9) & (phighEst < self.phigh*1.1) &
                    (phighEst > self.phigh*0.9)) or (self.skip)):

                # If hyperparameters are OK, break the loop. Otherwise repeat it with different seeds
                if self.verbose:
                    print('Optimal hyperparameters were found')
                break

            elif self.verbose:
                print(
                    'Optimal hyperparameters were not found. Rerunning the model on a new seed.')

            itv += 1
            if itv > self.steps:
                print('No solution found! Returning unassigned gammas')
                return np.zeros(self.n_contaminations, np.float32)

        # Sort the components values and extract the parameters posteriors
        means = means[idx_sortcomponents]
        covariances = covariances[idx_sortcomponents, :, :]

        mean_precs = bgm.mean_precision_[idx_sortcomponents]
        dgf = bgm.degrees_of_freedom_[idx_sortcomponents]

        # Compute the cumulative sum of the mixing proportion (GMM weights)
        gmm_weights = np.cumsum(dirichlet(alphas).rvs(
            self.n_contaminations), axis=1)

        w = {}
        for k in range(len(filter_idx)):
            w[k+1] = gmm_weights[:, k]

        # Sample from gamma's posterior by computing the probabilities
        gamma = self._sample_withexactprobs(
            means, mean_precs, covariances, dgf, delta, tau, w)

        # Clip gammas to upper limit
        gamma = gamma[gamma < self.gamma_lim]

        if np.all(gamma == 0):
            return np.zeros(self.n_contaminations, np.float32)

        if len(gamma) < self.n_contaminations:
            gamma = np.concatenate((gamma, np.random.choice(
                gamma[gamma > 0.0], self.n_contaminations - len(gamma), replace=True)))

        # return the sample from gamma's posterior
        return gamma

    def _order_components(self, means, mean_precs, covariances, dgf):

        K, M = np.shape(means)
        meanstd = np.zeros(K, np.float32)
        mean_std = np.sqrt(1/mean_precs)

        for k in range(K):

            sample_mean_component = multivariate_normal.rvs(mean=means[k, :], cov=mean_std[k]**2,
                                                            size=1000, random_state=self.random_state)
            sample_covariance = wishart.rvs(
                df=dgf[k], scale=covariances[k]/dgf[k], size=1000, random_state=self.random_state)

            var = np.array([np.diag(sample_covariance[i])
                           for i in range(1000)])

            meanstd[k] = np.mean([np.mean(sample_mean_component[:, m].reshape(-1) /
                                  (1 + np.sqrt(var[:, m].reshape(-1))))
                                  for m in range(M)])

        idx_components = np.argsort(-meanstd)
        meanstd = meanstd[idx_components]

        return idx_components, np.array(meanstd)

    def _find_delta_tau(self, params, *args):

        delta, tau = params
        meanstd, alphas = args

        first_eq = delta - (np.log(self.p0/(1 - self.p0)) - tau)/meanstd[0]

        prob_ck = self._sigmoid(delta, tau, meanstd)
        prob_c1ck = self._derive_jointprobs(prob_ck)

        a = np.cumsum(alphas)
        b = sum(alphas) - np.cumsum(alphas)

        probBetaGreaterT = np.nan_to_num(
            beta.sf(self.high_gamma, a, b), nan=1.0)

        second_eq = np.sum(probBetaGreaterT * prob_c1ck) - self.phigh

        return (first_eq, second_eq)

    def _check_delta_tau(self, delta, tau, meanstd, alphas):
        """Check that delta and tau are properly set.

        Return the p_0 and p_high estimated using the given delta and tau
        """

        prob_ck = self._sigmoid(delta, tau, meanstd)
        p0Est = 1 - prob_ck[0]

        prob_c1ck = self._derive_jointprobs(prob_ck)

        a = np.cumsum(alphas)
        b = sum(alphas) - np.cumsum(alphas)

        probBetaGreaterT = np.nan_to_num(
            beta.sf(self.high_gamma, a, b), nan=1.0)

        phighEst = np.sum(probBetaGreaterT * prob_c1ck)

        return p0Est, phighEst

    def _sigmoid(self, delta, tau, x):
        """Transforms scores into probabilities using a sigmoid function."""

        return 1/(1+np.exp(tau+delta*x))

    def _derive_jointprobs(self, prob_ck):
        """Obtain the joint probabilities given the conditional probabilities."""

        cumprobs = np.cumprod(prob_ck)
        negprobs = np.roll(1-prob_ck, -1)
        negprobs[-1] = 1

        prob_c1ck = cumprobs*negprobs

        return prob_c1ck

    def _sample_withexactprobs(self, means, mean_precs, covariances, dgf, delta, tau, w):
        """This function computes the joint probabilities and use them to get a sample from gamma's posterior."""

        K = np.shape(means)[0]
        mean_std = np.sqrt(1/mean_precs)
        samples = np.array([])

        i = 0
        random_start = 1234 if not self.random_state else self.random_state

        while len(samples) < self.n_contaminations * (1-self.p0):

            prob_ck = np.zeros(K, np.float32)
            for k in range(K):

                rnd = (i+1) * (k+1)

                sample_mean_component = multivariate_normal.rvs(mean=means[k, :], cov=mean_std[k]**2,
                                                                size=1, random_state=10*random_start + rnd)

                sample_covariance = wishart.rvs(df=dgf[k], scale=covariances[k]/dgf[k],
                                                size=1, random_state=10*random_start + rnd)

                var = np.diag(sample_covariance)
                meanstd = np.mean((sample_mean_component)/(1+np.sqrt(var)))

                prob_ck[k] = self._sigmoid(delta, tau, meanstd)

            prob_c1ck = self._derive_jointprobs(prob_ck)

            for k in range(K):
                ns = int(np.round(self.n_draws * prob_c1ck[k], 0))
                if ns > 0:
                    samples = np.concatenate(
                        (samples, np.random.choice(w[k+1], ns, replace=False)))
            i += 1

        if len(samples) > self.n_contaminations*(1-self.p0):
            samples = np.random.choice(samples, int(
                self.n_contaminations*(1-self.p0)), replace=False)

        samples = np.concatenate(
            (samples, np.zeros(self.n_contaminations - len(samples), np.float32)))

        return samples

    def _augment_space(self, decision):
        """Map outlier likelihood scores into the positive axis, take the log and z normalize the final values."""

        norm_scores = np.zeros_like(decision)

        for i, scores in enumerate(decision.T):

            minx = np.min(scores)
            x = np.log(scores - minx + 0.01)

            meanx = np.mean(x)
            stdx = np.std(x)

            x = (x - meanx)/stdx if stdx > 0 else x

            norm_scores[:, i] = x

        return norm_scores




if __name__ == '__main__':

    Start_T = time.time()
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Running MAD')
    parser.add_argument('--filename', type=str, default='001_NAB_id_1_Facility_tr_1007_1st_2014.csv')
    parser.add_argument('--data_direc', type=str, default='Datasets/TSB-AD-U/')
    parser.add_argument('--AD_Name', type=str, default='MAD')
    args = parser.parse_args()

    # multivariate
    # parser.add_argument('--filename', type=str, default='057_SMD_id_1_Facility_tr_4529_1st_4629.csv')
    # parser.add_argument('--data_direc', type=str, default='Datasets/TSB-AD-M/')

    Custom_AD_HP = {
        'random_state': 1234,   # not related to method itself, but to formatting input
        "n_contaminations": 1000,
        "n_draws": 50,
        "p0": 0.01,
        "phigh": 0.01,
        "high_gamma": 0.15,
        "gamma_lim": 0.5,
        "K": 100,
        "skip": False,
        "steps": 100,
        "random_state": 1234,
        "verbose": False
    }

    df = pd.read_csv(args.data_direc + args.filename).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()
    print('data: ', data.shape)
    print('label: ', label.shape)

    slidingWindow = find_length_rank(data, rank=1)
    train_index = args.filename.split('.')[0].split('_')[-3]
    data_train = data[:int(train_index), :]
    data_test = data[int(train_index):, :]
    label_test = label[int(train_index):]

    start_time = time.time()

    print("------- ON TEST DATA -------")
    clf = GAMGMM(**Custom_AD_HP)
    # clf.fit(data_train)
    output = clf.predict(data_test)
    pred = output   # output has already the predictions

    end_time = time.time()
    run_time = end_time - start_time

    evaluation_result = get_metrics(output, label_test, slidingWindow=slidingWindow, pred=pred)
    print('Evaluation Result: ', evaluation_result)
    # THIS IS THE OUTPUT, TODO maybe check and modify not to get np.float64?
    # Evaluation Result:  {'AUC-PR': np.float64(0.1185928941910645), 'AUC-ROC': np.float64(0.5093895820170664), 'VUS-PR': np.float64(0.12024946396753125), 
    # 'VUS-ROC': np.float64(0.5098574578334355), 'Standard-F1': np.float64(0.04838709677419355), 'PA-F1': 0.9716713881019831, 
    # 'Event-based-F1': np.float64(0.47368421052631543), 'R-based-F1': 0.23175435727792243, 'Affiliation-F': 0.7845131472572064}

    ####!
    # print("------- ON WHOLE DATA -------")
    # clf = GAMGMM(**Custom_AD_HP)
    # # clf.fit(data)
    # output = clf.predict(data)
    # pred = output
    # evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow, pred=pred)
    # print('Evaluation Result: ', evaluation_result)
