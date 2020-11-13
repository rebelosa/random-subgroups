"""
Ensembles of subgroup-discovery-based methods based on the
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
  ``_subgroup_discovery`` as sub-estimator implementations.

Only single output problems are handled.
"""

# Authors: Claudio Rebelo Sa <c.f.pinho.rebelo.de.sa@liacs.leidenuniv.nl>
#
# License: BSD 3 clause


import numbers
import warnings

import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.base import is_classifier, is_regressor
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted  # , check_random_state

from randomsubgroups.pysubgrouputils import *


# This function '_get_n_samples_bootstrap' is taken from sklearn.ensemble._forest
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
                 # random_state=None,
                 verbose=0,
                 # class_weight=None,
                 max_samples=None,
                 search_strategy='bestfirst',
                 result_set_size=1,
                 intervals_only=True,
                 n_bins=5):

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        #         self.oob_score = oob_score
        self.max_features = max_features
        self.n_jobs = n_jobs
        # self.random_state = random_state
        self.verbose = verbose
        #         self.class_weight = class_weight
        self.max_samples = max_samples
        self.search_strategy = search_strategy
        self.result_set_size = result_set_size
        self.intervals_only = intervals_only
        self.n_bins = n_bins

        self.is_fitted_ = False
        self.estimators_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.target = None
        self.n_features_ = None
        self.n_samples = None

        self.column_names = None

        # self.balance_bins = balance_bins

    # This function 'get_max_n_features' is taken from sklearn.tree._classes
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

    def _check_estimator_output(self, result):

        try:
            # Use the top ``n`` subgroup in the list result.to_descriptions()
            if self.result_set_size > 1:
                n = np.random.choice(len(result.to_descriptions()))
            else:
                n = 0
            return result.to_descriptions()[n]
        except UserWarning:
            warnings.warn("Could not find subgroups for one or more estimators.")
            return None

    def _apply(self, xy):
        """
        Apply Subgroup Discovery to ``xy`` and return the subgroup predictor.

        Parameters
        ----------

        xy : {DataFrame} of shape (n_samples, n_features+1)
            It includes ``n_features`` columns and one column with the target.

        Returns
        -------
        subgroup : dict
            Dictionary with the conditions and description of the subgroup

        target : int
            Indicates the target id to which the ``subgroup`` is associated with.
        """

        if self.bootstrap:
            n_samples_bootstrap = _get_n_samples_bootstrap(
                self.n_samples, self.max_samples
            )
            xy = xy.sample(n=n_samples_bootstrap, replace=True)

        if self.search_strategy == 'bestfirst':
            self.ps_algorithm = ps.BestFirstSearch()
        elif self.search_strategy == 'bestfirst2':
            self.ps_algorithm = ps.BestFirstSearch2()
        elif self.search_strategy == 'bestfirst3':
            self.ps_algorithm = ps.BestFirstSearch3(self.max_features)
        elif self.search_strategy == 'beam':
            self.ps_algorithm = ps.BeamSearch()
        elif self.search_strategy == 'apriori':
            self.ps_algorithm = ps.Apriori()
        else:
            msg = "Unknown search strategy. Available options are: " \
                  "['bestfirst', 'bestfirst2', 'bestfirst3', 'beam', 'apriori']"
            raise ValueError(msg)

        if self.search_strategy == 'bestfirst3':
            subgroup, target = self._subgroup_discovery(xy)
        else:
            subset_columns = np.append(np.random.choice(self.n_features_, size=self.max_features, replace=False),
                                       self.n_features_)
            subgroup, target = self._subgroup_discovery(xy.iloc[:, subset_columns].copy())

        return subgroup, target

    def fit(self, x, y):
        """
        Build an ensemble of subgroups for prediction from the training set (X, y).

        Parameters
        ----------
        x : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,).
            The target values (class labels in classification, real numbers in regression).

        Returns
        -------
        self : object

        """
        # random_state = check_random_state(self.random_state)

        is_classification = is_classifier(self)

        if is_classification:
            check_classification_targets(y)
            y = np.copy(y)

            # Store the classes seen during fit and encode y
            self.classes_, y_encoded = np.unique(y, return_inverse=True)

            self.n_classes_ = len(self.classes_)

            y = y_encoded

        # Check that x is a dataframe, if not then
        # creates a dataframe to be used in the Subgroup Discovery task
        if not isinstance(x, pd.DataFrame):
            self.column_names = ['Col' + str(i) for i in range(1, x.shape[1] + 1)]
            x = pd.DataFrame.from_records(x, columns=self.column_names)

        xy = x.copy()
        xy['target'] = y

        self.n_samples, self.n_features_ = x.shape
        self.get_max_n_features()

        model_desc = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend='loky')(
            delayed(self._apply)(xy.copy()) for _ in range(self.n_estimators))

        self.target = np.array([target for dsc, target in model_desc if dsc is not None])
        self.estimators_ = [dsc for dsc, target in model_desc if dsc is not None]

        self.estimators_ = [encode_subgroup(sg) for sg in self.estimators_]

        self.is_fitted_ = True

        return self

    def show_models(self):

        # Check if fit had been called
        check_is_fitted(self)

        if is_classifier(self):
            targets = self.target
        else:
            targets = self.target[:, 0]

        [print(f"Target: {target}; Model: {model}") for target, model in
         sorted(zip(targets, self.estimators_), key=lambda zp: zp[0])]

    def show_decision(self, x):

        # Check if fit had been called
        check_is_fitted(self)

        x = pd.DataFrame(x).transpose()

        if is_classifier(self):
            target = self.target
        else:
            target = self.target[:, 0]

        estimator_target = filter(lambda zp: zp[0].covers(x), zip(self.estimators_, target))

        if len(estimator_target) > 0:
            print("The predicted value is:", self.predict(x))
            print("From a total of", len(estimator_target), "estimators.\n")

            print("The subgroups used in the prediction are:")
            target_distribution = []
            for est, tgt in estimator_target:
                target_distribution.append(tgt)
                print(est, "--->", tgt)

            print("\nThe targets of the subgroups used in the prediction have the following distribution:")
            pd.Series(target_distribution).hist()
        else:
            print("No subgroups cover this example. The default prediction is used.")


class RandomSubgroupClassifier(SubgroupPredictorBase, ClassifierMixin):
    """
    A random subgroups classifier.

    A random subgroups classifier is a meta estimator that fits a number
    subgroups on various sub-samples of the dataset.
    The sub-sample size is controlled with the `max_samples` parameter if
    `bootstrap=True` (default), otherwise the whole dataset is used to search
    for each subgroup.

    Read more in https://pypi.org/project/random-subgroups/

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
    verbose : int, default=0
        Controls the verbosity when fitting and predicting.
    max_samples : int or float, default=None
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.
        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus,
          `max_samples` should be in the interval `(0, 1)`.

    Attributes
    ----------
    base_estimator_ : Subgroup
        The child estimator template used to create the collection of fitted
        sub-estimators.

    estimators_ : list of Subgroup
        The collection of fitted sub-estimators.

    classes_ : ndarray of shape (n_classes,) or a list of such arrays
        The classes labels (single output problem), or a list of arrays of
        class labels (multi-output problem).

    n_classes_ : int or list
        The number of classes (single output problem), or a list containing the
        number of classes for each output (multi-output problem).

    n_features_ : int
        The number of features when ``fit`` is performed.

    Notes
    -----
    The default values for the parameters controlling the size of the subgroups
    (in particular ``max_depth``) is defined as 1 to reduce memory consumption.
    Higher values (2-3) can lead to better results, but on some data sets but
    can drastically increase memory consumption and CPU usage. Therefore, the
    complexity of the subgroups should be controlled by setting this parameter.
    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data,
    ``max_features=n_features`` and ``bootstrap=False``.

    Examples
    --------
    >>> from randomsubgroups import RandomSubgroupClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=4,
    ...                            n_informative=2, n_redundant=0,
    ...                            random_state=0, shuffle=False)
    >>> clf = RandomSubgroupClassifier(max_depth=2)
    >>> clf.fit(X, y)
    RandomSubgroupClassifier(...)
    >>> print(clf.predict([[0, 0, 0, 0]]))
    [1]
    """

    def __init__(self,
                 n_estimators=100,
                 max_depth=1,
                 max_features="auto",
                 bootstrap=True,
                 # oob_score=False,
                 n_jobs=None,
                 # random_state=None,
                 verbose=0,
                 # class_weight=None,
                 max_samples=None,
                 quality_function_weight=0.5,
                 search_strategy='bestfirst',
                 result_set_size=1,
                 intervals_only=True,
                 n_bins=5):
        super().__init__(
            n_estimators=n_estimators,
            # estimator_params=estimator_params,
            bootstrap=bootstrap,
            # oob_score=oob_score,
            max_features=max_features,
            n_jobs=n_jobs,
            # random_state=random_state,
            verbose=verbose,
            # class_weight=class_weight,
            max_samples=max_samples,
            search_strategy=search_strategy,
            result_set_size=result_set_size,
            intervals_only=intervals_only)

        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.quality_function_weight = quality_function_weight

        self.search_strategy = search_strategy
        self.result_set_size = result_set_size
        self.intervals_only = intervals_only
        self.n_bins = n_bins

        self.classes_ = []

        self._estimator_type = "classifier"

    def _subgroup_discovery(self, xy):

        # self.target_id = random_state.choice(self.n_classes_)
        self.target_id = np.random.choice(self.n_classes_)
        xy.loc[:, 'target'] = (xy.loc[:, 'target'] == self.target_id)

        _target = ps.BinaryTarget(target_attribute='target', target_value=True)

        _search_space = ps.create_selectors(xy, ignore=['target'],
                                            intervals_only=self.intervals_only,
                                            nbins=self.n_bins)

        task = ps.SubgroupDiscoveryTask(
            data=xy,
            target=_target,
            search_space=_search_space,
            result_set_size=self.result_set_size,
            depth=self.max_depth,
            qf=ps.StandardQF(a=self.quality_function_weight),
            # qf=ps.ChiSquaredQF(),
            min_quality=0
        )
        result = self.ps_algorithm.execute(task)

        subgroup = self._check_estimator_output(result)
        if subgroup is not None:
            decoded_subgroup = decode_subgroup(subgroup)

            target = int(self.target_id)
        else:
            decoded_subgroup = None
            target = None

        return decoded_subgroup, target

    def predict(self, x):
        """
        Predict class for X.

        The predicted class of an input sample is a vote by the subgroups in
        the ensemble, weighted by their probability estimates. That is,
        the predicted class is the one with highest mean probability
        estimate across the subgroups.

        Parameters
        ----------
        x : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``.

        Returns
        -------
        y : array of shape (n_samples,)
            The predicted classes.
        """
        proba = self.predict_proba(x)

        return self.classes_.take(np.argmax(proba, axis=1), axis=0)

    def predict_proba(self, x):
        """
        Predict class probabilities for X.

        The predicted class probabilities of an input sample are computed as
        the mean predicted class of the subgroups in the ensemble.

        Parameters
        ----------
        x : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        if not isinstance(x, pd.DataFrame):
            # Create a dataframe to be used in the Subgroup Discovery task
            x = pd.DataFrame.from_records(x, columns=self.column_names)
        # X = check_array(X)
        # self._validate_X_predict(X)

        n_samples = x.shape[0]

        proba = np.zeros((n_samples, self.n_classes_), dtype=np.float64)
        for i in range(self.n_estimators):
            proba[self.estimators_[i].covers(x), self.target[i]] += 1

        proba = proba / self.n_estimators

        return proba


class RandomSubgroupRegressor(SubgroupPredictorBase, RegressorMixin):
    """
    A random subgroups regressor.

    A random subgroups regressor is a meta estimator that fits a number
    subgroups on various sub-samples of the dataset.
    The sub-sample size is controlled with the `max_samples` parameter if
    `bootstrap=True` (default), otherwise the whole dataset is used to search
    for each subgroup.

    The Parameters
    --------------
    n_estimators : int, default=100
       The number of subgroups in the ensemble.
    max_depth : int, default=1
       The maximum depth of the subgroup discovery task.
    max_features : {"auto", "sqrt", "log2"}, int or float, default="auto"
       The number of features to consider when looking for the best split:
       - If int, then consider `max_features` features at each split.
       - If float, then `max_features` is a fraction and
         `round(max_features * n_features)` features are considered at each
         split.
       - If "auto", then `max_features=n_features`.
       - If "sqrt", then `max_features=sqrt(n_features)`.
       - If "log2", then `max_features=log2(n_features)`.
       - If None, then `max_features=n_features`.
       Note: the search for a split does not stop until at least one
       valid partition of the node samples is found, even if it requires to
       effectively inspect more than ``max_features`` features.
    bootstrap : bool, default=True
       Whether bootstrap samples are used when building subgroups. If False,
       the whole dataset is used to build each tree.
    n_jobs : int, default=None
       The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
       :meth:`decision_path` and :meth:`apply` are all parallelized over the
       trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
       context. ``-1`` means using all processors. See :term:`Glossary
       <n_jobs>` for more details.
    verbose : int, default=0
       Controls the verbosity when fitting and predicting.
    max_samples : int or float, default=None
       If bootstrap is True, the number of samples to draw from X
       to train each base estimator.
       - If None (default), then draw `X.shape[0]` samples.
       - If int, then draw `max_samples` samples.
       - If float, then draw `max_samples * X.shape[0]` samples. Thus,
         `max_samples` should be in the interval `(0, 1)`.

    The Attributes
    --------------
    base_estimator_ : Subgroup
       The child estimator template used to create the collection of fitted
       sub-estimators.
    estimators_ : list of Subgroup
       The collection of fitted sub-estimators.
    n_features_ : int
       The number of features when ``fit`` is performed.

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
    search of the best split.
    The default value ``max_features="auto"`` uses ``n_features``
    rather than ``n_features / 3``. The latter was originally suggested in
    [1], whereas the former was more recently justified empirically in [2].

    Examples
    --------
    >>> from randomsubgroups import RandomSubgroupRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_features=4, n_informative=2,
    ...                        random_state=0, shuffle=False)
    >>> regr = RandomSubgroupRegressor(max_depth=2)
    >>> regr.fit(X, y)
    RandomSubgroupRegressor(...)
    >>> print(regr.predict([[0, 0, 0, 0]]))
    [-8.32987858]
    """
    def __init__(self,
                 n_estimators=100,
                 # *,
                 criterion="average",
                 max_depth=1,
                 max_features="auto",
                 bootstrap=True,
                 # oob_score=False,
                 n_jobs=None,
                 # random_state=None,
                 verbose=0,
                 # class_weight=None,
                 max_samples=None,
                 quality_function_weight=0.5,
                 quality_function='absolute',
                 search_strategy='bestfirst',
                 result_set_size=1,
                 balance_bins=None,
                 intervals_only=True,
                 n_bins=5):
        super().__init__(
            n_estimators=n_estimators,
            # estimator_params=estimator_params,
            max_depth=max_depth,
            bootstrap=bootstrap,
            # oob_score=oob_score,
            max_features=max_features,
            n_jobs=n_jobs,
            # random_state=random_state,
            verbose=verbose,
            # warm_start=warm_start,
            # class_weight=class_weight,
            max_samples=max_samples,
            search_strategy=search_strategy,
            result_set_size=result_set_size,
            intervals_only=intervals_only)

        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.quality_function_weight = quality_function_weight
        self.quality_function = quality_function
        self.criterion = criterion

        self.search_strategy = search_strategy
        self.result_set_size = result_set_size
        self.intervals_only = intervals_only
        self.n_bins = n_bins

        self._estimator_type = "regressor"

        self.balance_bins = balance_bins

    def _subgroup_discovery(self, xy):

        if self.balance_bins:
            discrete_target = pd.cut(xy.target, self.balance_bins)
            # print(discrete_target)
            # q = (1 - pd.Series(discrete_target).value_counts(normalize=True))
            # q = q / np.sum(q)
            # q2 = np.zeros(self.bins)
            # for i in range(self.bins):
            #     q2[discrete_target ==
            unique_discrete_target = discrete_target.unique()
            target_a = np.random.choice(unique_discrete_target)
            target_b = np.random.choice(unique_discrete_target)
            idx = (discrete_target == target_a) | (discrete_target == target_b)
            xy = xy[idx]
            # print(np.mean(xy.target))

        _target = ps.NumericTarget('target')

        _search_space = ps.create_selectors(xy, ignore=['target'],
                                            intervals_only=self.intervals_only,
                                            nbins=self.n_bins)

        if self.quality_function == 'standard':
            _qf = ps.StandardQFNumeric(a=self.quality_function_weight, estimator=self.criterion)
        elif self.quality_function == 'old':
            _qf = ps.OldStandardQFNumeric(a=self.quality_function_weight, estimator=self.criterion)
        elif self.quality_function == 'absolute':
            _qf = ps.AbsoluteQFNumeric(a=self.quality_function_weight, estimator=self.criterion)
        elif self.quality_function == 'kl':
            _qf = ps.KLDivergenceNumeric(a=self.quality_function_weight)
        else:
            msg = "Unknown numeric quality measure! Available options are: ['standard', 'absolute', 'kl']"
            raise ValueError(msg)

        task = ps.SubgroupDiscoveryTask(
            data=xy,
            target=_target,
            search_space=_search_space,
            result_set_size=self.result_set_size,
            depth=self.max_depth,
            qf=_qf,
            min_quality=0
        )
        result = self.ps_algorithm.execute(task)

        subgroup = self._check_estimator_output(result)
        decoded_subgroup = decode_subgroup(subgroup)

        idx = subgroup[1].covers(xy)
        target = np.array([np.mean(xy.target[idx]), np.mean(xy.target[np.invert(idx)])])

        return decoded_subgroup, target

    def predict(self, x, all_contribute=False):

        # class_type = self.classes_[0].dtype
        if not isinstance(x, pd.DataFrame):
            x = pd.DataFrame.from_records(x, columns=self.column_names)

        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        # X = check_array(X)

        # return predictions
        n_samples = x.shape[0]

        predictions = np.zeros(n_samples)
        counter = np.zeros(n_samples)

        for i in range(0, self.n_estimators):
            idx = self.estimators_[i].covers(x)
            predictions[idx] += self.target[i][0]

            if all_contribute:
                predictions[np.invert(idx)] += self.target[i][1]
                counter += 1
            else:
                counter[idx] += 1

        predictions_missing = (counter == 0)
        if any(predictions_missing):
            default_prediction = np.mean([e[1] for e in self.target])
            predictions[predictions_missing] = default_prediction
            counter[predictions_missing] += 1
            default_prediction_percent = round(np.mean(predictions_missing)*100, 0)
            if default_prediction_percent > 5:
                if self.verbose:
                    print("There are", str(default_prediction_percent), "% of default predictions. \n"
                          "Consider increasing the number 'n_estimators'.")

        predictions = predictions / counter

        # predictions.astype(class_type)

        return predictions
