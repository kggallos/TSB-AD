from __future__ import division
from __future__ import print_function

import abc
import warnings
from collections import defaultdict
from inspect import signature

import numpy as np
from numpy import percentile
from scipy.special import erf
from scipy.stats import binom
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import deprecated
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

import sys
sys.path.append('..')
try:
    from utils.utility import precision_n_scores, _pprint
except:
    from ..utils.utility import precision_n_scores, _pprint


class BaseDetector(metaclass=abc.ABCMeta):
    """Abstract class for all outlier detection algorithms.


    Parameters
    ----------
    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set,
        i.e. the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is fitted.

    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.

    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    """

    @abc.abstractmethod
    def __init__(self, contamination=0.1):

        if (isinstance(contamination, (float, int))):

            if not (0. < contamination <= 0.5):
                raise ValueError("contamination must be in (0, 0.5], "
                                 "got: %f" % contamination)

        # allow arbitrary input such as PyThreshld object
        self.contamination = contamination

    # noinspection PyIncorrectDocstring
    @abc.abstractmethod
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

    @abc.abstractmethod
    def decision_function(self, X):
        """Predict raw anomaly scores of X using the fitted detector.

        The anomaly score of an input sample is computed based on the fitted
        detector. For consistency, outliers are assigned with
        higher anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        pass

    @deprecated()
    def fit_predict(self, X, y=None):
        """Fit detector first and then predict whether a particular sample
        is an outlier or not. y is ignored in unsupervised models.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        outlier_labels : numpy array of shape (n_samples,)
            For each observation, tells whether
            it should be considered as an outlier according to the
            fitted model. 0 stands for inliers and 1 for outliers.

        .. deprecated:: 0.6.9
          `fit_predict` will be removed in pyod 0.8.0.; it will be
          replaced by calling `fit` function first and then accessing
          `labels_` attribute for consistency.
        """

        self.fit(X, y)
        return self.labels_

    def predict(self, X, return_confidence=False):
        """Predict if a particular sample is an outlier or not.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        return_confidence : boolean, optional(default=False)
            If True, also return the confidence of prediction.

        Returns
        -------
        outlier_labels : numpy array of shape (n_samples,)
            For each observation, tells whether
            it should be considered as an outlier according to the
            fitted model. 0 stands for inliers and 1 for outliers.
        confidence : numpy array of shape (n_samples,).
            Only if return_confidence is set to True.
        """

        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
        pred_score = self.decision_function(X)

        if isinstance(self.contamination, (float, int)):
            prediction = (pred_score > self.threshold_).astype('int').ravel()

        # if this is a PyThresh object
        else:
            prediction = self.contamination.eval(pred_score)

        if return_confidence:
            confidence = self.predict_confidence(X)
            return prediction, confidence

        return prediction

    def predict_proba(self, X, method='linear', return_confidence=False):
        """Predict the probability of a sample being outlier. Two approaches
        are possible:

        1. simply use Min-max conversion to linearly transform the outlier
           scores into the range of [0,1]. The model must be
           fitted first.
        2. use unifying scores, see :cite:`kriegel2011interpreting`.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        method : str, optional (default='linear')
            probability conversion method. It must be one of
            'linear' or 'unify'.

        return_confidence : boolean, optional(default=False)
            If True, also return the confidence of prediction.


        Returns
        -------
        outlier_probability : numpy array of shape (n_samples, n_classes)
            For each observation, tells whether or not
            it should be considered as an outlier according to the
            fitted model. Return the outlier probability, ranging
            in [0,1]. Note it depends on the number of classes, which is by
            default 2 classes ([proba of normal, proba of outliers]).
        """

        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
        train_scores = self.decision_scores_

        test_scores = self.decision_function(X)

        probs = np.zeros([X.shape[0], int(self._classes)])
        if method == 'linear':
            scaler = MinMaxScaler().fit(train_scores.reshape(-1, 1))
            probs[:, 1] = scaler.transform(
                test_scores.reshape(-1, 1)).ravel().clip(0, 1)
            probs[:, 0] = 1 - probs[:, 1]

            if return_confidence:
                confidence = self.predict_confidence(X)
                return probs, confidence

            return probs

        elif method == 'unify':
            # turn output into probability
            pre_erf_score = (test_scores - self._mu) / (
                    self._sigma * np.sqrt(2))
            erf_score = erf(pre_erf_score)
            probs[:, 1] = erf_score.clip(0, 1).ravel()
            probs[:, 0] = 1 - probs[:, 1]

            if return_confidence:
                confidence = self.predict_confidence(X)
                return probs, confidence

            return probs
        else:
            raise ValueError(method,
                             'is not a valid probability conversion method')

    def predict_confidence(self, X):
        """Predict the model's confidence in making the same prediction
        under slightly different training sets.
        See :cite:`perini2020quantifying`.

        Parameters
        -------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        confidence : numpy array of shape (n_samples,)
            For each observation, tells how consistently the model would
            make the same prediction if the training set was perturbed.
            Return a probability, ranging in [0,1].

        """

        check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])

        n = len(self.decision_scores_)

        # todo: this has an optimization opportunity since the scores may
        # already be available
        test_scores = self.decision_function(X)

        count_instances = np.vectorize(
            lambda x: np.count_nonzero(self.decision_scores_ <= x))
        n_instances = count_instances(test_scores)

        # Derive the outlier probability using Bayesian approach
        posterior_prob = np.vectorize(lambda x: (1 + x) / (2 + n))(n_instances)

        if not isinstance(self.contamination, (float, int)):
            contam = np.sum(self.labels_) / n
        # if this is a PyThresh object
        else:
            contam = self.contamination

        # Transform the outlier probability into a confidence value
        confidence = np.vectorize(
            lambda p: 1 - binom.cdf(n - int(n * contam), n, p))(
            posterior_prob)

        if isinstance(self.contamination, (float, int)):
            prediction = (test_scores > self.threshold_).astype('int').ravel()
        # if this is a PyThresh object
        else:
            prediction = self.contamination.eval(test_scores)
        np.place(confidence, prediction == 0, 1 - confidence[prediction == 0])

        return confidence

    def _predict_rank(self, X, normalized=False):
        """Predict the outlyingness rank of a sample by a fitted model. The
        method is for outlier detector score combination.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        normalized : bool, optional (default=False)
            If set to True, all ranks are normalized to [0,1].

        Returns
        -------
        ranks : array, shape (n_samples,)
            Outlying rank of a sample according to the training data.

        """

        check_is_fitted(self, ['decision_scores_'])

        test_scores = self.decision_function(X)
        train_scores = self.decision_scores_

        sorted_train_scores = np.sort(train_scores)
        ranks = np.searchsorted(sorted_train_scores, test_scores)

        if normalized:
            # return normalized ranks
            ranks = ranks / ranks.max()
        return ranks

    @deprecated()
    def fit_predict_score(self, X, y, scoring='roc_auc_score'):
        """Fit the detector, predict on samples, and evaluate the model by
        predefined metrics, e.g., ROC.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        scoring : str, optional (default='roc_auc_score')
            Evaluation metric:

            - 'roc_auc_score': ROC score
            - 'prc_n_score': Precision @ rank n score

        Returns
        -------
        score : float

        .. deprecated:: 0.6.9
          `fit_predict_score` will be removed in pyod 0.8.0.; it will be
          replaced by calling `fit` function first and then accessing
          `labels_` attribute for consistency. Scoring could be done by
          calling an evaluation method, e.g., AUC ROC.
        """

        self.fit(X)

        if scoring == 'roc_auc_score':
            score = roc_auc_score(y, self.decision_scores_)
        elif scoring == 'prc_n_score':
            score = precision_n_scores(y, self.decision_scores_)
        else:
            raise NotImplementedError('PyOD built-in scoring only supports '
                                      'ROC and Precision @ rank n')

        print("{metric}: {score}".format(metric=scoring, score=score))

        return score

    # def score(self, X, y, scoring='roc_auc_score'):
    #     """Returns the evaluation resulted on the given test data and labels.
    #     ROC is chosen as the default evaluation metric
    #
    #     :param X: The input samples
    #     :type X: numpy array of shape (n_samples, n_features)
    #
    #     :param y: Outlier labels of the input samples
    #     :type y: array, shape (n_samples,)
    #
    #     :param scoring: Evaluation metric
    #
    #             -' roc_auc_score': ROC score
    #             - 'prc_n_score': Precision @ rank n score
    #     :type scoring: str, optional (default='roc_auc_score')
    #
    #     :return: Evaluation score
    #     :rtype: float
    #     """
    #     check_is_fitted(self, ['decision_scores_'])
    #     if scoring == 'roc_auc_score':
    #         score = roc_auc_score(y, self.decision_function(X))
    #     elif scoring == 'prc_n_score':
    #         score = precision_n_scores(y, self.decision_function(X))
    #     else:
    #         raise NotImplementedError('PyOD built-in scoring only supports '
    #                                   'ROC and Precision @ rank n')
    #
    #     print("{metric}: {score}".format(metric=scoring, score=score))
    #
    #     return score

    def _set_n_classes(self, y):
        """Set the number of classes if `y` is presented, which is not
        expected. It could be useful for multi-class outlier detection.

        Parameters
        ----------
        y : numpy array of shape (n_samples,)
            Ground truth.

        Returns
        -------
        self
        """

        self._classes = 2  # default as binary classification
        if y is not None:
            check_classification_targets(y)
            self._classes = len(np.unique(y))
            warnings.warn(
                "y should not be presented in unsupervised learning.")
        return self

    def _process_decision_scores(self):
        """Internal function to calculate key attributes:

        - threshold_: used to decide the binary label
        - labels_: binary labels of training data

        Returns
        -------
        self
        """

        if isinstance(self.contamination, (float, int)):
            self.threshold_ = percentile(self.decision_scores_,
                                         100 * (1 - self.contamination))
            self.labels_ = (self.decision_scores_ > self.threshold_).astype(
                'int').ravel()

        # if this is a PyThresh object
        else:
            self.labels_ = self.contamination.eval(self.decision_scores_)
            self.threshold_ = self.contamination.thresh_
            if not self.threshold_:
                self.threshold_ = np.sum(self.labels_) / len(self.labels_)

        # calculate for predict_proba()

        self._mu = np.mean(self.decision_scores_)
        self._sigma = np.std(self.decision_scores_)

        return self

    # noinspection PyMethodParameters
    def _get_param_names(cls):
        # noinspection PyPep8
        """Get parameter names for the estimator

        See http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
        and sklearn/base.py for more information.
        """

        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    # noinspection PyPep8
    def get_params(self, deep=True):
        """Get parameters for this estimator.

        See http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
        and sklearn/base.py for more information.

        Parameters
        ----------
        deep : bool, optional (default=True)
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """

        out = dict()
        for key in self._get_param_names():
            # We need deprecation warnings to always be on in order to
            # catch deprecated param values.
            # This is set in utils/__init__.py but it gets overwritten
            # when running under python3 somehow.
            warnings.simplefilter("always", DeprecationWarning)
            try:
                with warnings.catch_warnings(record=True) as w:
                    value = getattr(self, key, None)
                if len(w) and w[0].category == DeprecationWarning:
                    # if the parameter is deprecated, don't show it
                    continue
            finally:
                warnings.filters.pop(0)

            # XXX: should we rather test if instance of estimator?
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        # noinspection PyPep8
        """Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        See http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
        and sklearn/base.py for more information.

        Returns
        -------
        self : object
        """

        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def __repr__(self):
        # noinspection PyPep8
        """
        See http://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
        and sklearn/base.py for more information.
        """

        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name, _pprint(self.get_params(deep=False),
                                               offset=len(class_name), ),)