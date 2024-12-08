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
from sklearn.preprocessing import MinMaxScaler

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.models.base import BaseDetector
from .grimshaw import grimshaw

import numpy as np
from .POT_Detector import POT


class DSPOT(BaseDetector):
    """
    DSPOT class for Drift Streaming Peak-Over-Threshold thresholder.

    Parameters
    ----------
    depth: int
        Number of data point selected to detect drift

    num_init : int
        Number of data points used to initialize the threshold

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
        Not used, present for API consistency by convention.

    thresholds : list
        Contains the dynamic thresholds for the whole time series.

    initial_threshold_: float
        Initial threshold obtained with POT
    
    anomaly_indices : ndarray/list
        1D array with indices of anomalies. Computed when predict() is called.
    
    decision_scores_: ndarray of shape (n_samples,) #TODO
        Not actually used, present for API consistency by convention.
        It contains 0s and 1s because this is a thresholding method.

    peaks_: list
        Data points greater that initial_threshold_
    
    Notes
    -----
    Implements the Streaming Peak-Over-Threshold algorithm as described in:
    Siffer, Alban, et al. "Anomaly detection in streams with extreme value theory."
    Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery
    and Data Mining. 2017.
    """

    def __init__(self, num_init=None, risk=1e-4, depth=100,
                 init_level=0.98, num_candidates=10, epsilon=1e-8):
        self.num_init = num_init
        self.risk = risk
        self.depth = depth
        self.init_level = init_level
        self.num_candidates = num_candidates
        self.epsilon = epsilon

        self.threshold_ = None
        self.anomaly_indices = []
        self.thresholds = []
        self.initial_threshold_ = None
        self.peaks_ = None
        # self.k_ = 0  # Tracks the number of processed samples

    def fit(self, X, y=None):
        """
        Initialize the SPOT detector using the first `num_init` points.

        Parameters
        ----------
        data : numpy array of shape (n_samples,)
            The input data stream.

        Returns
        -------
        self : object
            Fitted SPOT instance.
        """
        X = self._check_dimensions(X)

        if not self.num_init:
            # set whole input as training part (keep some data to be base data)
            self.num_init = len(X) - self.depth

        if len(X) < self.num_init:
            raise ValueError("Input data must have at least `num_init` samples.")

        self.base_data_ = X[:self.depth]
        init_data = X[self.depth : self.depth + self.num_init]

        for i in range(self.num_init):
            temp = init_data[i]
            init_data[i] -= self.base_data_.mean()
            np.delete(self.base_data_, 0)
            np.append(self.base_data_, temp)

        try:
            pot = POT(risk=self.risk, init_level=self.init_level,  
                    num_candidates=self.num_candidates, epsilon=self.epsilon)
            pot.fit(init_data.reshape(-1, 1))
        except ValueError as e:
            error_message = str(e)
            if "arange: cannot compute length" in error_message:
                raise ValueError(f"{error_message}. The num_init parameter is set too low.")
            else:
                raise ValueError(error_message)

        self.threshold_ = pot.threshold_
        self.initial_threshold_ = pot.threshold_
        self.peaks_ = init_data[init_data > self.initial_threshold_] - self.initial_threshold_

        # threshold of training data is inital threshold
        self.thresholds = [pot.threshold_] * (self.depth + self.num_init)

        self.decision_scores_ = np.zeros(len(X)) 

        return self
    
    def decision_function(self, X):
        # return self.predict(X) #TODO maybe we can keep this? Or is it inconsistent?
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
        X = self._check_dimensions(X)
        
        k = self.num_init
        current_threshold = self.initial_threshold_
        peaks = self.peaks_
        base_data = self.base_data_

        anomaly_indices = []

        for index, x in enumerate(X):
            temp = x
            x -= base_data.mean()
            if x > current_threshold:
                anomaly_indices.append(index)
            elif x > self.initial_threshold_:
                peaks = np.append(peaks, x - self.initial_threshold_)
                gamma, sigma = grimshaw(peaks=peaks, threshold=self.initial_threshold_)
                k = k + 1
                r = k * self.risk / peaks.size
                if gamma != 0:
                    current_threshold = self.initial_threshold_ + (sigma / gamma) * (pow(r, -gamma) - 1)
                else:
                    # TODO what happens if gamma is 0 ?
                    current_threshold = self.initial_threshold_ - sigma
                np.delete(base_data, 0)
                np.append(base_data, temp)
            else:
                k = k + 1
                np.delete(base_data, 0)
                np.append(base_data, temp)

            self.thresholds.append(current_threshold)
        
        self.anomaly_indices = anomaly_indices

        preds = np.zeros(len(X), dtype=int)
        preds[anomaly_indices] = 1
        return preds

    def get_thresholds(self):
        """
        Get dynamic thresholds of trained and fitted data.
        If not fitted, returns only thresholds of trained data."""
        return self.thresholds
    
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


# def run_Custom_AD_Semisupervised(data, num_init, 
#             risk, init_level, num_candidates, epsilon):
#     clf = SPOT(num_init=num_init, risk=risk, init_level=init_level,
#                num_candidates=num_candidates, epsilon=epsilon)
def run_Custom_AD_Semisupervised(data_train, data_test, HP):
    clf = DSPOT(**HP)
    clf.fit(data_train)
    score = clf.predict(data_test)
    # score = clf.decision_function(data_test)
    # score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

if __name__ == '__main__':

    Start_T = time.time()
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Running DSPOT')
    parser.add_argument('--filename', type=str, default='001_NAB_id_1_Facility_tr_1007_1st_2014.csv')
    parser.add_argument('--data_direc', type=str, default='Datasets/TSB-AD-U/')
    parser.add_argument('--AD_Name', type=str, default='DSPOT')

    # mutlivariate
    # parser.add_argument('--filename', type=str, default='057_SMD_id_1_Facility_tr_4529_1st_4629.csv')
    # parser.add_argument('--data_direc', type=str, default='Datasets/TSB-AD-M/')

    args = parser.parse_args()

    Custom_AD_HP = {
        'num_init': 500, # if this is set too low, grimshaw cannot be computed
        'risk': 1e-4,
        'depth': 100,
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

    output = run_Custom_AD_Semisupervised(data_train, data, Custom_AD_HP)
    # output = run_Custom_AD_Semisupervised(data_train, data, **Custom_AD_HP)

    end_time = time.time()
    run_time = end_time - start_time

    pred = output   #NOTE output has already the predictions
    # pred = output > (np.mean(output)+3*np.std(output))
    evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow, pred=pred)
    print('Evaluation Result: ', evaluation_result)