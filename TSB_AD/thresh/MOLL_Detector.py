# -*- coding: utf-8 -*-
# Author: Konstantinos Gallos <kggallos@gmail.com>
# License: Apache-2.0 License
"""
This code is adapted from [pythresh] by [KulikDM]
Original source: [https://github.com/KulikDM/pythresh]
"""

# https://github.com/geomdata/gda-public/blob/master/timeseries/curve_geometry.pyx

import pandas as pd
import numpy as np
import scipy.signal as signal
from scipy import integrate
import argparse, time

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.models.base import BaseDetector

from .thresholding_utils import check_scores, normalize

class MOLL(BaseDetector):
    r"""MOLL class for Friedrichs' mollifier thresholder.

       Use the Friedrichs' mollifier to evaluate a non-parametric means
       to threshold scores generated by the decision_scores where outliers
       are set to any value beyond one minus the  maximum of the smoothed
       dataset via convolution. See :cite:`keyzer1997moll` for details.

       Parameters
       ----------

       Attributes
       ----------

        threshold_ : float
            The threshold value that separates inliers from outliers.

        decision_scores_: ndarray of shape (n_samples,) #TODO
            Not actually used, present for API consistency by convention.
            It contains 0s and 1s because this is a thresholding method.

       Notes
       -----

       Friedrichs' mollifier is a smoothing function that is applied to create sequences
       of smooth functions. These functions can be used to approximate generalized functions
       that may be non-smooth. The decision scores are assumed to be a part of a generalized
       function with a non-smooth nature in terms of the interval space between the scores
       respectively. Friedrichs' mollifier is defined by:

       .. math::

           \varphi(x) = \begin{cases}
                        Ce^{\frac{1}{\lvert x \rvert^2-1}} & \text{if } \lvert x \rvert < 1 \\
                        0 & \text{if } \lvert x \rvert \geq 1
                        \end{cases} \mathrm{,}

       where :math:`C` is a normalization constant and :math:`x` is the z-scores of pseudo
       scores generated over the same range as the scores but with a smaller step size. The
       normalization constant is calculated by:

       .. math::

           C = \left(\int_{-1}^{1} e^{\frac{1}{(x^2-1)}}\right)^{-1} \mathrm{.}

       The mollifier is inserted into a discrete convolution operator and the smoothed
       scores are returned. The threshold is set at one minus the maximum of the smoothed
       scores.

    """
    
    def __init__(self, random_state=1234, normalize=True):
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

        dat_range = np.linspace(0, 1, len(X))

        # Set the inliers to be where the 1-max(smoothed scores)
        limit = 1-np.max(self._mollifier(dat_range, np.sort(X)))

        self.threshold_ = limit

        preds = np.zeros(n_samples, dtype=int)
        preds[X >= limit] = 1

        return preds
    
    
    def _mollifier(self, time, position, refinement=5, width=1.0):

        N = len(position)

        delta = (time[-1]-time[0])/(N-1)

        # compute boundary space padding
        left_pad = np.arange(time[0], time[0]-(width+delta), step=-delta)
        left_pad = np.flipud(left_pad)[:-1]
        left_pad_num = left_pad.shape[0]
        right_pad = np.arange(time[-1], time[-1]+(width+delta), step=delta)[1:]
        right_pad_num = right_pad.shape[0]
        time_pad = np.concatenate((left_pad, time, right_pad))

        # compute boundary score padding
        position_pad = np.pad(position, (left_pad_num, right_pad_num), 'edge')

        # Define a new smaller space scale s, ds (here we a evenly spaced)
        s, ds = np.linspace(time_pad[0], time_pad[-1],
                            (refinement)*time_pad.shape[0],
                            retstep=True)
        right_pad_num = (refinement)*right_pad_num
        left_pad_num = (refinement)*left_pad_num
        position_interp = np.interp(s, time_pad, position_pad)

        # Compute the mollifier kernel
        norm_const, err = integrate.quad(
            lambda x: np.exp(1.0/(x**2-1.0)), -1.0, 1.0)
        norm_const = 1.0/norm_const

        # Compute the mollifier rho
        p = np.abs((s - (s[0]+s[-1])/2.0)/width)
        r = np.zeros_like(s)
        q = p[p < 1.0]
        r[p < 1.0] = np.exp(1.0/(q**2-1.0))
        rho = (norm_const/width)*r

        # Perform convolution to make smooth reconstruction
        conv_func = signal.fftconvolve if s.shape[0] > 500 else np.convolve
        smooth = conv_func(ds*position_interp, rho, mode='same')

        # remove padding
        s = s[left_pad_num:-right_pad_num]
        smooth = smooth[left_pad_num:-(right_pad_num)]

        return np.asarray(smooth)


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
    clf = MOLL(**Custom_AD_HP)
    # clf.fit(data_train)
    output = clf.predict(data_test)
    pred = output   # output has already the predictions

    end_time = time.time()
    run_time = end_time - start_time

    evaluation_result = get_metrics(output, label_test, slidingWindow=slidingWindow, pred=pred)
    print('Evaluation Result: ', evaluation_result)

    ####!
    print("------- ON WHOLE DATA -------")
    clf = MOLL(**Custom_AD_HP)
    # clf.fit(data)
    output = clf.predict(data)
    pred = output
    evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow, pred=pred)
    print('Evaluation Result: ', evaluation_result)
