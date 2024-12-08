# -*- coding: utf-8 -*-
# Author: Konstantinos Gallos <kggallos@gmail.com>
# License: Apache-2.0 License
"""
This code is adapted from [peak-over-threshold] by [cbhua]
Original source: [https://github.com/cbhua/peak-over-threshold]
"""

import pandas as pd
import numpy as np
import argparse, time
from math import log
from sklearn.preprocessing import MinMaxScaler

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.models.base import BaseDetector
from .grimshaw import grimshaw

from .thresholding_utils import normalize


class POT(BaseDetector):
    """
    POT class for Peak-Over-Threshold thresholder.

    Parameters
    ----------
    risk : float
        Detection level

    init_level: int
        Probability associated with the initial threshold
    
    num_candidates: int
        The maximum number of nodes we choose as candidates
        
    epsilon: float
        Numerical parameter to perform
        
    Attributes
    ----------
    threshold_ : float
        The threshold value that separates inliers from outliers.

    anomaly_indices : ndarray
        1D array with indices of anomalies. Computed when predict() is called.

    decision_scores_: ndarray of shape (n_samples,) #TODO
        Not actually used, present for API consistency by convention.
        It contains 0s and 1s because this is a thresholding method.

    Notes
    -----
    Implements the Streaming Peak-Over-Threshold algorithm as described in:
    Siffer, Alban, et al. "Anomaly detection in streams with extreme value theory."
    Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery
    and Data Mining. 2017.
    """

    def __init__(self, risk=1e-4, init_level=0.98,
                  num_candidates=10, epsilon=1e-8, normalize=True):
        super().__init__()
        self.risk = risk
        self.init_level = init_level
        self.num_candidates = num_candidates
        self.epsilon = epsilon
        self.normalize = normalize

        self.threshold_ = None
        self.anomaly_indices = []

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

        n_samples, n_features = X.shape

        X = self._check_dimensions(X)

        if self.normalize: X = normalize(X)

        # Set init threshold
        t = np.sort(X)[int(self.init_level * n_samples)]
        peaks = X[X > t] - t

        # Grimshaw
        gamma, sigma = grimshaw(peaks=peaks, 
                                threshold=t, 
                                num_candidates=self.num_candidates, 
                                epsilon=self.epsilon
                                )

        # Calculate Threshold
        r = n_samples * self.risk / peaks.size
        if gamma != 0:
            z = t + (sigma / gamma) * (pow(r, -gamma) - 1)
        else: 
            z = t - sigma * log(r)

        self.threshold_ = z

        #TODO should we keep this?
        # or maybe set decision_scores_ with 0s and 1s based on the method?
        self.decision_scores_ = np.zeros(n_samples) 

        return self

    def decision_function(self, X):
        # return self.predict(X) #TODO maybe we can keep this? Or is it inconsistent?
        pass

    def predict(self, X):
        """Predict outliers of X using the fitted thresholding method.

        The anomalies of an input sample is computed based on different
        thresholding algorithms. Normal data points are defined as 0 
        while outliers are defined as 1.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        predictions : numpy array of shape (n_samples,)
            The predictions of the input samples.
        """
        X = self._check_dimensions(X)

        n_samples = len(X)

        # n_samples, n_features = X.shape
        preds = np.zeros(n_samples)
        preds[X >= self.threshold_] = 1

        self.anomaly_indices = np.where(preds == 1)[0]

        return preds
    
    def get_thresholds(self):
        """
        Get threshold of fitted data.
        """
        return self.threshold_

    def get_anomaly_indices(self):
        """
        Get anomalies indices of predicted data.
        If not fitted, returns empty list."""
        return self.anomaly_indices
    
    def _check_dimensions(self, X):
        """
        Ensures that the input is univariate and reshapes it to a 1D array.

        Parameters
        ----------
        X : numpy array
            The input data points.

        Returns
        -------
        numpy array of shape (n_samples,)
            The reshaped univariate input data.

        Raises
        ------
        ValueError
            If X is not univariate or not a valid numpy array.
        """
        X = np.asarray(X)  # ensure X is a numpy array
        
        if X.ndim == 1: # X is already 1D
            return X
        
        if X.ndim == 2 and X.shape[1] == 1:
            return X.reshape(-1)
        
        raise ValueError(
            f"Expected a univariate array (1D or 2D with one column). "
            f"Got array with shape {X.shape}."
        )



# def run_Custom_AD_Unsupervised(data, 
#             risk, init_level, num_candidates, epsilon):
    # clf = POT(risk=risk, init_level=init_level,
    #            num_candidates=num_candidates, epsilon=epsilon)
def run_Custom_AD_Unsupervised(data, HP):
    clf = POT(**HP)
    clf.fit(data)
    score = clf.predict(data)
    # score = clf.decision_scores_
    # score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score


if __name__ == '__main__':

    Start_T = time.time()
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Running POT')
    parser.add_argument('--filename', type=str, default='001_NAB_id_1_Facility_tr_1007_1st_2014.csv')
    parser.add_argument('--data_direc', type=str, default='Datasets/TSB-AD-U/')
    parser.add_argument('--AD_Name', type=str, default='POT')

    # mutlivariate
    # parser.add_argument('--filename', type=str, default='057_SMD_id_1_Facility_tr_4529_1st_4629.csv')
    # parser.add_argument('--data_direc', type=str, default='Datasets/TSB-AD-M/')

    args = parser.parse_args()

    Custom_AD_HP = {
        'risk': 1e-4,
        'init_level': 0.98,
        'num_candidates': 10,
        'epsilon': 1e-8,
    }

    df = pd.read_csv(args.data_direc + args.filename).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()
    print('data: ', data.shape)
    print('label: ', label.shape)

    slidingWindow = find_length_rank(data, rank=1)
    train_index = args.filename.split('.')[0].split('_')[-3]
    data_train = data[:int(train_index), :]

    start_time = time.time()

    # output = run_Custom_AD_Unsupervised(data, **Custom_AD_HP)
    output = run_Custom_AD_Unsupervised(data, Custom_AD_HP)

    end_time = time.time()
    run_time = end_time - start_time

    pred = output   #NOTE output has already the predictions
    # pred = output > (np.mean(output)+3*np.std(output))
    evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow, pred=pred)
    print('Evaluation Result: ', evaluation_result)