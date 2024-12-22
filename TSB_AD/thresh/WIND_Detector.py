# -*- coding: utf-8 -*-
# Author: Konstantinos Gallos <kggallos@gmail.com>
# License: Apache-2.0 License
"""
This code is adapted from [pythresh] by [KulikDM]
Original source: [https://github.com/KulikDM/pythresh]
"""

import pandas as pd
import numpy as np
from scipy import integrate, stats
import argparse, time

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank

from .thresholding_utils import check_scores, normalize, gen_kde

class WIND():
    r"""WIND class for topological Winding number thresholder.

       Use the topological winding number (with respect to the origin) to
       evaluate a non-parametric means to threshold scores generated by
       the decision_scores where outliers are set to any value beyond the
       mean intersection point calculated from the winding number.
       See :cite:`jacobson2013wind` for details.

       Parameters
       ----------

       random_state : int, optional (default=1234)
            Random seed for the normal distribution. Can also be set to None.

       Attributes
       ----------

        threshold_ : float
            The threshold value that separates inliers from outliers.

        decision_scores_: ndarray of shape (n_samples,) #TODO
            Not actually used, present for API consistency by convention.
            It contains 0s and 1s because this is a thresholding method.

       Notes
       -----

       The topological winding number or the degree of a continuous mapping. It is an
       integer sum of the number of completed/closed counterclockwise rotations in a plane
       around a point. And is given by,

       .. math::

           \mathrm{d}\theta = \frac{1}{r^2} \left(x\mathrm{d}y - y\mathrm{d}x \right) \mathrm{,}

       where :math:`r^2 = x^2 + y^2`

       .. math::

           wind(\gamma,0) = \frac{1}{2\pi} \oint_\gamma \mathrm{d}\theta

       The winding number intuitively captures self-intersections/contours, with a change in the
       distribution of the dataset or shift from inliers to outliers relating to these intersections.
       With this, it is assumed that if an intersection exists, then adjacent/incident regions
       must have different region labels. Since multiple intersection regions may exist. The
       threshold between inliers and outliers is taken as the mean intersection point.

       Examples
       --------
       The effects of randomness can affect the thresholder's output performance
       significantly. Therefore, to alleviate the effects of randomness on the
       thresholder a combined model can be used with different random_state values.
       E.g.



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

        # Create a normal distribution and normalize
        size = min(len(X), 1500)
        norm = stats.norm.rvs(size=size, loc=0.0, scale=1.0,
                              random_state=self.random_state)
        norm = normalize(norm)

        # Create a KDE of the labels and the normal distribution
        # Generate KDE
        val_data, dat_range = gen_kde(X, 0, 1, len(X)*3)
        val_norm, _ = gen_kde(norm, 0, 1, len(X)*3)

        # Get the rsquared value
        r2 = val_data**2 + val_norm**2

        val_data = val_data/np.max(val_data)
        val_norm = val_norm/np.max(val_norm)

        # Find the first derivatives of the decision and norm kdes
        # with respect to the decision scores
        deriv_data = np.gradient(val_data, dat_range[1]-dat_range[0])
        deriv_norm = np.gradient(val_norm, dat_range[1]-dat_range[0])

        # Compute integrand
        integrand = self._dtheta(
            val_data, val_norm, deriv_data, deriv_norm, r2)

        # Integrate to find winding numbers mean intersection point
        limit = integrate.simpson(integrand)/np.sum((val_data+val_norm)/2)

        self.threshold_ = limit

        preds = np.zeros(n_samples, dtype=int)
        preds[X >= self.threshold_] = 1

        return preds

    def _dtheta(self, x, y, dx, dy, r2):
        """Calculate dtheta for the integrand."""
        return (x*dy - y*dx)/r2

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

    clf = WIND(**Custom_AD_HP)
    output = clf.predict(data)
    pred = output
    evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow, pred=pred)
    print('Evaluation Result: ', evaluation_result)
