"""
Ensembles of subgroup-discovery-based methods. These methods are based on the
scikit-learn library.

Given the similarities between the ``RandomForests`` and ``RandomSubgroups``,
whenever possible the RandomForest variables nomenclature was used.

The module structure is the following:

- The ``SubgroupPredictorBase`` base class implements a common ``fit`` method for all
  the estimators in the module. This ``fit`` method calls the ``fit`` method of each
   sub-estimator on random samples (with replacement, a.k.a. bootstrap) of the training set.

- The ``RandomSubgroupClassifier`` and ``RandomSubgroupRegressor`` derived
  classes provide the user with concrete implementations of
  the ensembles using classical Subgroup Discovery approaches
  from the ``pysubgroup`` package.
  ``BinaryTarget`` and ``NumericTarget`` implementations are used in
  ``_searchSG`` as sub-estimator implementations.

Single and multi-output problems are both handled.
"""

# Authors: Claudio Rebelo Sa <c.f.pinho.rebelo.de.sa@liacs.leidenuniv.nl>
#
# License: BSD 3 clause


import numbers
from random import random

import numpy as np
import pysubgroup as ps
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_is_fitted

from randomsubgroups.pysubgrouputils import decodeSubgroup, encodeSubgroup


def _get_n_samples_bootstrap(n_samples, max_samples):
    """
    Get the number of samples in a bootstrap sample.
    Parameters
    ----------
    n_samples : int
        Number of samples in the dataset.
    max_samples : int or float
        The maximum number of samples to draw from the total available:
            - if float, this indicates a fraction of the total and should be
              the interval `(0, 1)`;
            - if int, this indicates the exact number of samples;
            - if None, this indicates the total number of samples.
    Returns
    -------
    n_samples_bootstrap : int
        The total number of samples to draw for the bootstrap sample.
    """
    if max_samples is None:
        return n_samples

    if isinstance(max_samples, numbers.Integral):
        if not (1 <= max_samples <= n_samples):
            msg = "`max_samples` must be in range 1 to {} but got value {}"
            raise ValueError(msg.format(n_samples, max_samples))
        return max_samples

    if isinstance(max_samples, numbers.Real):
        if not (0 < max_samples < 1):
            msg = "`max_samples` must be in range (0, 1) but got value {}"
            raise ValueError(msg.format(max_samples))
        return round(n_samples * max_samples)

    msg = "`max_samples` should be int or float, but got type '{}'"
    raise TypeError(msg.format(type(max_samples)))


class SubgroupPredictorBase(BaseEstimator):
    """
    Base class for the Random Subgroups predictors.
    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    def __init__(self,
                 n_estimators=100,
                 # *,
                 max_depth=1,
                 max_features="auto",
                 bootstrap=True,
                 # oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 # class_weight=None,
                 max_samples=None,
                 search_strategy='bestfirst'):

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.bootstrap = bootstrap
#         self.oob_score = oob_score
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
#         self.class_weight = class_weight
        self.max_samples = max_samples
        self.search_strategy=search_strategy

        self.size = 1

    def get_max_n_features(self):
        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError("Invalid value for max_features. "
                                 "Allowed string values are 'auto', "
                                 "'sqrt' or 'log2'.")
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1,
                                   int(self.max_features * self.n_features_))
            else:
                max_features = 0

        self.max_features = max_features

    def _apply(self, X):
        """
        Apply Subgroup Discovery to X and return the subgroup predictor.

        Parameters
        ----------
        X : {DataFrame} of shape (n_samples, n_features+1). It includes
        ``n_features`` columns and one column with the target.

        Returns
        -------
        subg : dictionary with the conditions and description of the subgroup

        target : integer indicating the target id to which ``subg`` is
            associated with.
        """

        if hasattr(self, "classes_"):
            if len(self.classes_)>2:
                self.target_name = np.random.choice(self.classes_)
                X.target = (X.target == self.target_name)
            else:
                self.target_name = True

        if self.bootstrap:
            n_samples_bootstrap = _get_n_samples_bootstrap(
                self.n_samples, self.max_samples
            )
            X = X.sample(n=n_samples_bootstrap, replace=True)

        if self.search_strategy == 'bestfirst':
            self.ps_algorithm = ps.BestFirstSearch()
        elif self.search_strategy == 'beam':
            self.ps_algorithm = ps.BeamSearch()
        elif self.search_strategy == 'apriori':
            self.ps_algorithm = ps.Apriori()
        elif self.search_strategy == 'randomsubsets':
            if not hasattr(self, 'subset_percentage'):
                self.subset_percentage = self.max_features/self.n_features_
                self.max_features = self.n_features_
            self.ps_algorithm = ps.RandomSubsetSearch(self.subset_percentage)
        else:
            msg = "Unknown pysubgroup Algorithm. Available options are: ['bestfirst', 'beam', 'apriori', 'randomsubsets']"
            raise ValueError(msg)

        subsetcols = np.append( np.random.choice(self.n_features_, size=self.max_features, replace=False),
                                self.n_features_ )

        subg, target = self._subgroup_discovery(X.iloc[:, subsetcols])

        return subg, target

    def fit(self, X, y):
        """
        Build an ensemble of subgroups for prediction from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,). The target values (class labels
            in classification, real numbers in regression).

        Returns
        -------
        self : object
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Create dataframe to be used in the Subgroup Discovery task
        Xy = X.copy()
        Xy['target'] = y

        # Store the classes seen during fit
        if hasattr(self, "classes_"):
            self.classes_ = unique_labels(y)

        self.n_features_ = X.shape[1]
        self.n_samples = X.shape[0]
        self.get_max_n_features()

        # backend = ['loky', 'multiprocessing', 'sequential', 'threading']
        model_desc = Parallel(n_jobs=self.n_jobs, verbose = self.verbose, backend='loky')(
                                   delayed(self._apply)(Xy) for _ in range(self.n_estimators))

        # Make self.target private
        self.target = [trgt for dsc, trgt in model_desc]
        self.estimators_ = [dsc for dsc, trgt in model_desc]

        self.estimators_ = [encodeSubgroup(sg) for sg in self.estimators_]

        return self


class RandomSubgroupClassifier(SubgroupPredictorBase):
    """
    A random subgroups classifier.

    A random subgroups classifier is a meta estimator that fits a number
    subgroups on various sub-samples of the dataset.
    The sub-sample size is controlled with the `max_samples` parameter if
    `bootstrap=True` (default), otherwise the whole dataset is used to search
    for each subgroup.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of subgroups in the ensemble.

    max_depth : int, default=1
        The maximum depth of the subgroup discovery task.

    max_features : {"auto", "sqrt", "log2"}, int or float, default="auto"
        The number of features to consider when looking for the best subgroup:

        - If int, then consider `max_features` features for each subgroup.
        - If float, then `max_features` is a fraction and
          `round(max_features * n_features)` features are considered for each
          subgroup.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

    bootstrap : bool, default=True
        Whether bootstrap samples are used when looking for subgroups. If False, the
        whole dataset is used to build each subgroup.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
        :meth:`decision_path` and :meth:`apply` are all parallelized over the
        subgroup discovery task. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors. See :term:`Glossary
        <n_jobs>` for more details.

    random_state : int or RandomState, default=None
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``) and the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``).
        See :term:`Glossary <random_state>` for details.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    class_weight : {"balanced", "balanced_subsample"}, dict or list of dicts, \
            default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.
        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``
        The "balanced_subsample" mode is the same as "balanced" except that
        weights are computed based on the bootstrap sample for every tree
        grown.
        For multi-output, the weights of each column of y will be multiplied.
        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.
        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus,
          `max_samples` should be in the interval `(0, 1)`.
        .. versionadded:: 0.22

    Attributes
    ----------
    base_estimator_ : DecisionTreeClassifier
        The child estimator template used to create the collection of fitted
        sub-estimators.

    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.
    classes_ : ndarray of shape (n_classes,) or a list of such arrays
        The classes labels (single output problem), or a list of arrays of
        class labels (multi-output problem).
    n_classes_ : int or list
        The number of classes (single output problem), or a list containing the
        number of classes for each output (multi-output problem).
    n_features_ : int
        The number of features when ``fit`` is performed.
    n_outputs_ : int
        The number of outputs when ``fit`` is performed.
    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.
        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.
    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
        This attribute exists only when ``oob_score`` is True.
    oob_decision_function_ : ndarray of shape (n_samples, n_classes)
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN. This attribute exists
        only when ``oob_score`` is True.

    See Also
    --------

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.
    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data,
    ``max_features=n_features`` and ``bootstrap=False``, if the improvement
    of the criterion is identical for several splits enumerated during the
    search of the best split. To obtain a deterministic behaviour during
    fitting, ``random_state`` has to be fixed.
    References
    ----------

    --------
    """

    def __init__(self,
                 n_estimators=100, #*,
                 #criterion="gini",
                 max_depth=1,
                 #min_samples_split=2,
                 #min_samples_leaf=1,
                 #min_weight_fraction_leaf=0.,
                 max_features="auto",
                 #max_leaf_nodes=None,
                 #min_impurity_decrease=0.,
                 #min_impurity_split=None,
                 bootstrap=True,
                 #oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 #warm_start=False,
                 #class_weight=None,
                 #ccp_alpha=0.0,
                 max_samples=None,
                 random_flip_binary_target=0.0,
                 search_strategy='bestfirst'):
        super().__init__(
            n_estimators=n_estimators,
            #estimator_params=estimator_params,
            bootstrap=bootstrap,
            #oob_score=oob_score,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            #warm_start=warm_start,
            #class_weight=class_weight,
            max_samples=max_samples,
            search_strategy=search_strategy)

        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.search_strategy = search_strategy

        self.random_flip_binary_target = random_flip_binary_target
        self.classes_ = []

        self.size = 1

    def _subgroup_discovery(self, X):

        # Because we can obtain different subgroups if we flip the target
        if self.random_flip_binary_target > 0 and (random() < self.random_flip_binary_target):
            self.target_name = False

        _SubgroupDiscoveryTarget = ps.BinaryTarget ('target', True)

        _SubgroupDiscoverySearchspace = ps.create_selectors(X, ignore=['target'])

        task = ps.SubgroupDiscoveryTask (
            X,
            _SubgroupDiscoveryTarget,
            _SubgroupDiscoverySearchspace,
            result_set_size = self.size,
            depth = self.max_depth,
            qf = ps.WRAccQF()
        )
        result = self.ps_algorithm.execute(task)

        # Use the top subgroup in the list result.to_descriptions()
        subgroup = result.to_descriptions()[0]
        decoded_subg = decodeSubgroup(subgroup)

        target = int(self.target_name)

        return decoded_subg, target

    def predict(self, X, threshold = None):

        class_type = self.classes_[0].dtype

        proba = self.predict_proba(X)

        # We can change this threshold to let the
        # model also predict 'Unknown Class' or default
        if threshold is not None:
            proba = self.predict_proba(X) > threshold

        predictions = np.argmax(proba, axis=1)

        predictions.astype(class_type)

        return predictions

    def predict_proba(self, X):

        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        #X = check_array(X)

        # return predictions
        n_samples = X.shape[0]

#         proba = np.zeros(n_samples)

#         for subgroup in self.model:
#             proba[subgroup.covers(X)] += 1
        proba = np.zeros((n_samples, self.classes_.shape[0]), dtype=np.float64)

#         for subgroup in self.model:
        for i in range(self.n_estimators):
            proba[self.estimators_[i].covers(X), self.target[i]] += 1

        proba = proba / self.n_estimators

        return proba
#     def predict(self, X, threshold = 0.5):
#
#         class_type = self.classes_[0].dtype
#
#         proba = self.predict_proba(X)
#         predictions = np.argmax(proba, axis=1)
#
#         predictions.astype(class_type)
#
#         return predictions
#
#     def predict_proba(self, X):
#
#         # Check is fit had been called
#         #check_is_fitted(self)
#
#         # Input validation
#         #X = check_array(X)
#
#         # return predictions
#         n_samples = X.shape[0]
#
#         proba = np.zeros((n_samples, self.classes_.shape[0]), dtype=np.float64)
#
# #         for subgroup in self.model:
#         for i in range(self.n_estimators):
#             proba[self.model[i].covers(X),self.target[i]] += 1
#
# #         proba = proba / len(self.model)
# #         for proba in all_proba:
# #             proba /= len(self.classes_)
#         default_class = 0
#         for pr in proba:
#             total_predictions = sum(pr)
#             if total_predictions>0:
#                 pr /= sum(pr)
#             else:
#                 pr[default_class] = 1
#
# #         return proba


class RandomSubgroupRegressor(SubgroupPredictorBase):

    def __init__(self,
                 n_estimators=100, #*,
                 criterion="average",
                 max_depth=1,
                 max_features="auto",
                 bootstrap=True,
                 #oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 #class_weight=None,
                 max_samples=None,
                 QF_weight=1,
                 QF='absolute',
                 search_strategy='bestfirst'):
        super().__init__(
            n_estimators=n_estimators,
            #estimator_params=estimator_params,
            bootstrap=bootstrap,
            #oob_score=oob_score,
            max_features = max_features,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            #warm_start=warm_start,
            #class_weight=class_weight,
            max_samples=max_samples,
            search_strategy=search_strategy)

        self.search_strategy=search_strategy
        self.QF_weight = QF_weight
        self.QF=QF
        self.criterion = criterion

        self.size = 1

    def _subgroup_discovery(self, X):

        _target_type = ps.NumericTarget('target')

        _searchspace = ps.create_selectors(X, ignore=['target'])

        if (self.QF == 'standard'):
            _qf = ps.StandardQFNumeric(a=self.QF_weight, estimator=self.criterion)
        elif (self.QF == 'absolute'):
            _qf = ps.AbsoluteQFNumeric(a=self.QF_weight, estimator=self.criterion)
        else:
            msg = "Unknown numeric quality measure! Available options are: ['standard', 'absolute']"
            raise ValueError(msg)

        task = ps.SubgroupDiscoveryTask (
            X,
            _target_type,
            _searchspace,
            result_set_size = self.size,
            depth = self.max_depth,
            qf = _qf
        )
        result = self.ps_algorithm.execute(task)

        # Use the top subgroup in the list result.to_descriptions()
        subgroup = result.to_descriptions()[0]
        decoded_subg = decodeSubgroup(subgroup)

        idx = subgroup[1].covers(X)
        target = [ np.mean(X.target[idx]), np.mean(X.target[np.invert(idx)]) ]

        return decoded_subg, target

    def predict(self, X):

        #class_type = self.classes_[0].dtype

        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        #X = check_array(X)

        # return predictions
        n_samples = X.shape[0]

        predictions = np.zeros(n_samples)
        counter = np.zeros(n_samples)

        for i in range(0,self.n_estimators):
            idx = self.model[i].covers(X)
            predictions[idx] += self.target[i][0]
            counter[idx] += 1
#             counter += 1
#             predictions[np.invert(idx)] += self.target[i][1]
            #predictions[idx == False] += self.complement_mean[i]

        print(counter)
        predictions = predictions / counter

        #predictions.astype(class_type)

        return predictions
