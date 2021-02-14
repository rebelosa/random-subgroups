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


import warnings
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.base import is_classifier, is_regressor
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted  # , check_random_state
from tqdm import trange

from randomsubgroups.pysubgrouputils import *
from randomsubgroups.subgroup import SubgroupPredictor


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
                 n_estimators=200,
                 max_depth=2,
                 max_features="auto",
                 bootstrap=True,
                 max_samples=None,
                 n_jobs=None,
                 verbose=0,
                 search_strategy='static',
                 top_n=1,
                 result_set_size=5,
                 intervals_only=False,
                 n_bins=5,
                 binning='ef'):

        # Ensemble parameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_samples = max_samples

        # General parameters
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Subgroup Discovery parameters
        self.search_strategy = search_strategy
        self.top_n = top_n
        self.result_set_size = result_set_size
        self.intervals_only = intervals_only
        self.n_bins = n_bins
        self.binning = binning

        self.is_fitted_ = False
        self.estimators_ = None
        self.classes_ = None
        self.n_classes_ = None
        self.n_features_ = None
        self.n_samples = None
        self.default_prediction = None

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
            max_features = None
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

        if len(result.to_descriptions()) > 0:

            if len(result.to_descriptions()) > self.top_n:
                # Randomly select from the the top_n subgroups in the list result.to_descriptions()
                n = np.random.choice(self.top_n, 1, replace=False)[0]
                # return [result.to_descriptions()[i] for i in n]
            elif self.result_set_size > 1:
                n = np.random.choice(len(result.to_descriptions()))
            else:
                n = 0
            return [result.to_descriptions()[n]]
        else:
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

        if self.search_strategy in ['static', 'dynamic']:
            self.ps_algorithm = LightBestFirstSearch(max_features=self.max_features, n_bins=self.n_bins,
                                                     intervals_only=self.intervals_only,
                                                     specialization=self.search_strategy,
                                                     binning=self.binning)
            subgroup_predictor = self._subgroup_discovery(xy)
            return subgroup_predictor
        elif self.search_strategy == 'bestfirst':
            self.ps_algorithm = ps.BestFirstSearch()
        elif self.search_strategy == 'beam':
            self.ps_algorithm = ps.BeamSearch()
        elif self.search_strategy == 'apriori':
            self.ps_algorithm = ps.Apriori()
        else:
            msg = "Unknown search strategy. Available options are: " \
                  "['static', 'dynamic', 'beam', 'apriori']"
            raise ValueError(msg)

        subset_columns = np.append(np.random.choice(self.n_features_, size=self.max_features, replace=False),
                                   self.n_features_)
        subgroup_predictor = self._subgroup_discovery(xy.iloc[:, subset_columns].copy())

        return subgroup_predictor

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

        if is_classifier(self):
            check_classification_targets(y)
            # y = np.copy(y)

            # Store the classes seen during fit and encode y
            self.classes_, y_encoded = np.unique(y, return_inverse=True)

            self.n_classes_ = len(self.classes_)

            y = y_encoded
        elif is_regressor(self):
            self.default_prediction = np.mean(y)

        # Check that x is a dataframe, if not then
        # creates a dataframe to be used in the Subgroup Discovery task
        if not isinstance(x, pd.DataFrame):
            self.column_names = ['Col' + str(i) for i in range(0, x.shape[1])]
            x = pd.DataFrame.from_records(x, columns=self.column_names)
        else:
            self.column_names = x.columns

        xy = x.copy()
        xy['target'] = y

        self.n_samples, self.n_features_ = x.shape
        self.get_max_n_features()

        model_desc = Parallel(n_jobs=self.n_jobs, verbose=0, backend='loky')(
            delayed(self._apply)(xy.copy()) for _ in (trange(self.n_estimators)
                                                      if self.verbose else range(self.n_estimators)))

        self.estimators_ = [SubgroupPredictor.from_dict(sg) for subgroup in model_desc
                            if subgroup is not None for sg in subgroup]

        n = self.n_estimators - len(self.estimators_)
        if n > 0 and self.verbose:
            print("Could not find {} out of {} estimators.".format(n, self.n_estimators))

        self.is_fitted_ = True

        return self

    def show_models(self):

        # Check if fit had been called
        check_is_fitted(self)

        if is_classifier(self):
            sorted_list = [[self.classes_[estimator.target], estimator] for estimator in
                           sorted(self.estimators_, key=lambda e: e.target)]
        elif is_regressor(self):
            sorted_list = [[estimator.target, estimator] for estimator in
                           sorted(self.estimators_, key=lambda e: e.target)]
        else:
            msg = "Unknown type of model. Must be 'regressor' or 'classifier'"
            raise ValueError(msg)

        [print(f"Target: {target}; Model: {estimator}") for target, estimator in sorted_list]
        return pd.DataFrame(sorted_list, columns=["Target", "Model"])

    def show_decision(self, x):

        # Check if fit had been called
        check_is_fitted(self)

        # Input validation
        # x = pd.DataFrame(x).transpose()
        if not isinstance(x, pd.DataFrame):
            # Create a dataframe to be used in the Subgroup Discovery task
            x = pd.DataFrame.from_records((x,), columns=self.column_names)

        covered_estimators = filter(lambda e: e.covers(x), self.estimators_)
        covered_estimators = list(sorted(covered_estimators, key=lambda e: (e.target, e)))

        if len(covered_estimators) > 0:

            print("The predicted value is:", self.predict(x)[0])
            print("From a total of", len(covered_estimators), "estimators.\n")

            print("The subgroups used in the prediction are:")

            if is_classifier(self):
                target_distribution = []
                initial_target = -1
                for estimator in covered_estimators:
                    if estimator.target != initial_target:
                        initial_target = estimator.target
                        print("\n Predicting target {}".format(initial_target))
                    target_distribution.append(estimator.target)
                    print(estimator, "--->", estimator.target)
                print("\nThe targets of the subgroups used in the prediction have the following distribution:")
                pd.Series(target_distribution).value_counts().plot.pie()

            elif is_regressor(self):
                target_distribution = []
                for estimator in covered_estimators:
                    target_distribution.append(estimator.target)
                    print(estimator, "--->", estimator.target)
                print("\nThe targets of the subgroups used in the prediction have the following distribution:")
                pd.Series(target_distribution).hist(bins=self.balance_bins)
            else:
                msg = "Can only show decision for Classifier or Regression models"
                raise ValueError(msg)
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
                 max_depth=2,
                 max_features="auto",
                 bootstrap=True,
                 n_jobs=None,
                 verbose=0,
                 max_samples=None,
                 quality_function='standard',
                 quality_function_weight=0.5,
                 search_strategy='static',
                 top_n=1,
                 result_set_size=5,
                 intervals_only=False,
                 n_bins=5,
                 binning='ef'):
        super().__init__()

        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Subgroup Discovery parameters
        self.quality_function = quality_function
        self.quality_function_weight = quality_function_weight

        self.search_strategy = search_strategy
        self.top_n = top_n
        self.result_set_size = result_set_size
        self.intervals_only = intervals_only
        self.n_bins = n_bins
        self.binning = binning

        self.classes_ = []

        self._estimator_type = "classifier"

    def _subgroup_discovery(self, xy):

        # target_id = random_state.choice(self.n_classes_)
        target_id = list(np.random.choice(self.n_classes_, 1, replace=False))[0]
        # xy = xy.query("target == @target_id")

        _target = ps.BinaryTarget(target_attribute='target', target_value=target_id)

        if self.search_strategy == 'dynamic':
            _search_space = []
        elif self.search_strategy == 'static':
            _search_space = get_search_space(xy, ignore=['target'],
                                             intervals_only=self.intervals_only,
                                             n_bins=self.n_bins)
        else:
            _search_space = ps.create_selectors(xy, ignore=['target'],
                                                intervals_only=self.intervals_only,
                                                nbins=self.n_bins)

        if self.quality_function == 'standard':
            _qf = ps.StandardQF(a=self.quality_function_weight)
        elif self.quality_function == 'chisquared':
            _qf = ps.ChiSquaredQF()
        elif self.quality_function == 'ga':
            _qf = ps.GeneralizationAware_StandardQF(a=self.quality_function_weight)
        else:
            msg = "Unknown nominal quality measure! Available options are: ['standard']"
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
        if subgroup is not None:
            subgroup_predictor_dict = [SubgroupPredictor(sg, target=target_id,
                                                         alternative_target=None).to_dict() for sg in subgroup]
            # decoded_subgroup_predictor = SubgroupPredictor(subgroup, target=int(target_id)).to_dict()
        else:
            subgroup_predictor_dict = None

        return subgroup_predictor_dict

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
        for estimator in self.estimators_:
            proba[estimator.covers(x), estimator.target] += 1

        proba = proba / self.n_estimators

        return proba

    def feature_importances(self, absolute=False):

        # Check is fit had been called
        check_is_fitted(self)

        counter = {target: {att: 0 for att in self.column_names} for target in range(self.n_classes_)}
        for e in self.estimators_:
            for f in e.get_features():
                counter[e.target][f] += 1

        counter_df = pd.DataFrame().from_dict(counter)
        counter_df.columns = self.classes_

        print(counter_df.sum(axis=1)/counter_df.values.sum())

        if not absolute:
            counter_df = counter_df / counter_df.sum(axis=0)

        return counter_df


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
                 max_depth=2,
                 max_features="auto",
                 bootstrap=True,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 max_samples=None,
                 quality_function_weight=0.5,
                 quality_function='absolute',
                 criterion='average',
                 search_strategy='static',
                 top_n=1,
                 result_set_size=5,
                 balance_bins=None,
                 intervals_only=False,
                 min_support=2,
                 n_bins=5,
                 binning='ef'):
        super().__init__()

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.max_samples = max_samples

        # Subgroup Discovery parameters
        self.quality_function_weight = quality_function_weight
        self.quality_function = quality_function
        self.criterion = criterion
        self.min_support = min_support
        self.search_strategy = search_strategy
        self.top_n = top_n
        self.result_set_size = result_set_size
        self.intervals_only = intervals_only
        self.n_bins = n_bins
        self.binning = binning

        self.default_prediction = None
        self._estimator_type = "regressor"

        # Experimental
        self.balance_bins = balance_bins
        # np.random.seed(self.random_state)
        # print(np.random.get_state()[1][0])

    def _subgroup_discovery(self, xy):

        if self.balance_bins:
            # discrete_target = pd.qcut(xy.target, self.balance_bins, labels=False, duplicates='drop')
            discrete_target = pd.cut(xy.target, self.balance_bins, labels=False)
            (target_a, target_b) = np.random.choice(discrete_target.max()+1, 2, replace=False)
            xy = xy[(discrete_target == target_a) | (discrete_target == target_b)]

            # (idx1,) = np.where(discrete_target == target_a)
            # (idx2,) = np.where(discrete_target == target_b)
            # new_size = np.max([len(idx1), len(idx2)])
            # idx = np.concatenate([np.random.choice(idx1, new_size), np.random.choice(idx2, new_size)])
            # xy = xy.iloc[idx, :]

        _target = ps.NumericTarget('target')

        if self.search_strategy == 'dynamic':
            _search_space = []
        elif self.search_strategy == 'static':
            _search_space = get_search_space(xy, ignore=['target'],
                                             intervals_only=self.intervals_only,
                                             n_bins=self.n_bins)
        else:
            _search_space = ps.create_selectors(xy, ignore=['target'],
                                                intervals_only=self.intervals_only,
                                                nbins=self.n_bins)

        if self.quality_function == 'ps.standard':
            _qf = ps.StandardQFNumeric(a=self.quality_function_weight)
        elif self.quality_function == 'standard':
            _qf = StandardNumeric(a=self.quality_function_weight, min_support=self.min_support)
        elif self.quality_function == 'absolute':
            _qf = AbsoluteNumeric(a=self.quality_function_weight, min_support=self.min_support)
        elif self.quality_function == 'kl':
            _qf = KLDivergenceNumeric(a=self.quality_function_weight, min_support=self.min_support)
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

        if subgroup is not None:
            subgroup_predictor_dict = []
            for sg in subgroup:
                idx = sg[1].covers(xy)
                if self.criterion == "average":
                    target = float(np.mean(xy.target[idx]))
                    alternative_target = float(np.mean(xy.target[np.invert(idx)]))
                elif self.criterion == "median":
                    target = float(np.median(xy.target[idx]))
                    alternative_target = float(np.median(xy.target[np.invert(idx)]))
                elif self.criterion == "maxmin":
                    dataset_mean = np.mean(xy.target)
                    if np.mean(xy.target[idx]) > dataset_mean:
                        target = float(np.max(xy.target[idx]))
                        alternative_target = float(np.max(xy.target[np.invert(idx)]))
                    else:
                        target = float(np.min(xy.target[idx]))
                        alternative_target = float(np.min(xy.target[np.invert(idx)]))
                subgroup_predictor_dict.append(SubgroupPredictor(sg, target=target,
                                                                 alternative_target=alternative_target).to_dict())
        else:
            subgroup_predictor_dict = None

        return subgroup_predictor_dict

    def predict(self, x):

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

        for estimator in self.estimators_:
            idx = estimator.covers(x)
            if any(idx):
                counter[idx] += 1
                for i, p in enumerate(estimator.predict(x)):
                    if p is not None:
                        predictions[i] += p

        predictions_missing = (counter == 0)
        if any(predictions_missing):
            # default_prediction = np.mean([e.alternative_target for e in self.estimators_])
            # default_prediction = self.mean_target
            predictions[predictions_missing] = self.default_prediction
            counter[predictions_missing] += 1
            default_prediction_percent = round(np.mean(predictions_missing) * 100, 0)
            if default_prediction_percent > 5:
                if self.verbose:
                    msg = '''There were {}% default predictions.
                             Consider increasing 'n_estimators' or using 
                             'balance_bins' > 2 '''.format(default_prediction_percent)
                    # warnings.warn(msg)
                    print(msg)

        predictions = predictions / counter

        # predictions.astype(class_type)

        return predictions

    def feature_importances(self, absolute=False):

        # Check is fit had been called
        check_is_fitted(self)

        counter = {att: 0 for att in self.column_names}
        for e in self.estimators_:
            for f in e.get_features():
                counter[f] += 1

        counter_df = pd.Series(counter)

        print(counter_df/len(self.estimators_))

        if not absolute:
            counter_df = counter_df / counter_df.sum(axis=0)

        return counter_df