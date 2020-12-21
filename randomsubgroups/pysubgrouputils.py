"""
    This extends the pysubgroup package classes
"""

# Authors: Claudio Rebelo Sa <c.f.pinho.rebelo.de.sa@liacs.leidenuniv.nl>
#
# License: BSD 3 clause

import random
from collections import namedtuple
from heapq import heappop, heappush
from itertools import chain, islice

import numpy as np
import pandas as pd
import pysubgroup as ps
from pysubgroup import BinaryTarget, defaultdict
from pysubgroup.subgroup_description import EqualitySelector, IntervalSelector
from scipy.stats import entropy
import numbers


class QFNumeric(object):
    tpl = namedtuple('QF_parameters', ('size', 'mean'))

    def __init__(self, a, min_support=2):
        if not isinstance(a, numbers.Number):
            raise ValueError(f'a is not a number. Received a={a}')
        self.a = a
        self.dataset_statistics = None
        self.all_target_values = None
        self.has_constant_statistics = False
        self.min_support = min_support

    def calculate_constant_statistics(self, data, target):
        self.all_target_values = data[target.target_variable].to_numpy()
        target_mean = np.mean(self.all_target_values)
        data_size = len(data)
        self.dataset_statistics = QFNumeric.tpl(data_size, target_mean)
        self.has_constant_statistics = True

    def evaluate(self, subgroup, target, data, statistics=None):
        cover_arr = subgroup.covers(data)
        subgroup_size = np.count_nonzero(cover_arr)
        if subgroup_size >= self.min_support:
            subgroup_target = self.all_target_values[cover_arr]
            return self.get_score(self.dataset_statistics.size, self.all_target_values, subgroup_size, subgroup_target)
        else:
            return float("-inf")

    def calculate_statistics(self, subgroup, data, cached_statistics=None):
        return 0


class AbsoluteNumeric(QFNumeric):

    def get_score(self, dataset_size, dataset_target, subgroup_size, subgroup_target):
        # return (subgroup_size) ** self.a * np.abs(
        #     np.mean(subgroup_target) - self.dataset_statistics.mean)
        return (subgroup_size/self.dataset_statistics.size) ** self.a * np.abs(np.mean(subgroup_target) - self.dataset_statistics.mean)


class StandardNumeric(QFNumeric):

    def get_score(self, dataset_size, dataset_target, subgroup_size, subgroup_target):
        # return (subgroup_size ** self.a) * (np.mean(subgroup_target) - self.dataset_statistics.mean)
        return (subgroup_size/self.dataset_statistics.size) ** self.a * (np.mean(subgroup_target) - self.dataset_statistics.mean)


class KLDivergenceNumeric(QFNumeric):

    @staticmethod
    def kl_divergence(data_target, subgroup_target):
        epsilon = 0.00001

        pk = np.histogram(subgroup_target)[0] + epsilon
        pk = pk / sum(pk)
        qk = np.histogram(data_target)[0] + epsilon
        qk = qk / sum(qk)
        return entropy(pk, qk)

    def get_score(self, dataset_size, dataset_target, subgroup_size, subgroup_target):
        # return subgroup_size ** self.a * self.kl_divergence(dataset_target, subgroup_target)
        return (subgroup_size/self.dataset_statistics.size) ** self.a * self.kl_divergence(dataset_target, subgroup_target)


class SpecializationOperator:
    """
        This SpecializationOperator is similar to
        StaticSpecializationOperator except that in every
        depth search it can reconstruct the intervals with the dynamic specialization.

        It is therefore computationally more heavy than the static specialization.
    """
    def __init__(self,
                 data,
                 n_bins=5,
                 max_features=None,
                 intervals_only=True,
                 binning='ef',
                 specialization='static',
                 search_space=None):
        self.data = data
        self.n_bins = n_bins
        self.max_features = max_features
        self.intervals_only = intervals_only
        self.binning = binning
        self.specialization = specialization

        if search_space:
            self.search_space = search_space
        elif specialization == 'static':
            self.search_space = get_search_space(self.data, ignore=['target'],
                                                 intervals_only=self.intervals_only,
                                                 n_bins=self.n_bins,
                                                 binning=self.binning)

        self.columns = list(self.data.columns)

    def refinements(self, subgroup):

        if self.max_features:
            subset_attributes = random.sample(self.columns, self.max_features)
        else:
            subset_attributes = self.columns

        if self.specialization == 'static':
            if self.max_features:
                search_space_dict = {key: values for key, values in self.search_space.items()
                                     if key in subset_attributes}
            else:
                search_space_dict = self.search_space

        elif self.specialization == 'dynamic':
            data_subset = self.data[subgroup.covers(self.data)]

            search_space_dict = get_search_space(data_subset[subset_attributes], ignore=['target'],
                                                 n_bins=self.n_bins,
                                                 intervals_only=self.intervals_only,
                                                 binning=self.binning)

        if subgroup.depth > 0 and not self.max_features:
            index_of_last = list(search_space_dict.keys()).index(subgroup._selectors[-1].attribute_name)
            search_space = islice(search_space_dict.values(), index_of_last + 1, None)
        else:
            search_space = search_space_dict.values()

        new_selectors = chain.from_iterable(search_space)

        return (subgroup & sel for sel in new_selectors)


class LightBestFirstSearch:
    """
    This LightBestFirstSearch is a lighter version of the BestFirstSearch from pysubgroup package:
    https://github.com/flemmerich/pysubgroup/blob/master/pysubgroup/algorithms.py
    This one, among other things, skips the optimistic estimates to improve the efficiency.
    It uses the SpecializationOperator() instead of StaticSpecializationOperator()
    """
    def __init__(self,
                 max_features=None,
                 n_bins=5,
                 intervals_only=True,
                 specialization="static",
                 binning='ef'):
        self.max_features = max_features
        self.n_bins = n_bins
        self.intervals_only = intervals_only
        self.specialization = specialization
        self.binning = binning

    def execute(self, task):
        result = []
        queue = [(float("-inf"), ps.Conjunction([]))]

        operator = SpecializationOperator(data=task.data.drop(['target'], axis=1), n_bins=self.n_bins,
                                          max_features=self.max_features,
                                          intervals_only=self.intervals_only,
                                          binning=self.binning, specialization=self.specialization,
                                          search_space=task.search_space)
        task.qf.calculate_constant_statistics(task.data, task.target)
        while queue:
            q, old_description = heappop(queue)
            q = -q
            if not q > ps.minimum_required_quality(result, task):
                break
            for candidate_description in operator.refinements(old_description):
                score_eval = task.qf.evaluate(candidate_description, task.target, task.data, None)
                ps.add_if_required(result, candidate_description, score_eval, task)
                if len(candidate_description) < task.depth:
                    heappush(queue, (-score_eval, candidate_description))

        result.sort(key=lambda x: x[0], reverse=True)
        return ps.SubgroupDiscoveryResult(result, task)

# def create_selectors(data, n_bins=5, intervals_only=True, binning='ef', ignore=None):
#     if ignore is None:
#         ignore = []
#     search_space = []
#     [search_space.extend(make_nominal_bins(values)) for key, values in
#                         data.select_dtypes(exclude=['number']).iteritems() if key not in ignore]
#     [search_space.extend(make_numeric_bins(values, n_bins, intervals_only, binning)) for key, values in
#                         data.select_dtypes(include=['number']).iteritems() if key not in ignore]
#     return search_space


def get_search_space(data, n_bins=5, intervals_only=True, binning='ef', ignore=None):
    if ignore is None:
        ignore = []
    search_space = {}
    search_space.update({key: make_nominal_bins(values) for key, values in
                        data.select_dtypes(exclude=['number']).iteritems() if key not in ignore})
    search_space.update({key: make_numeric_bins(values, n_bins, intervals_only, binning) for key, values in
                        data.select_dtypes(include=['number']).iteritems() if key not in ignore})
    return search_space


def jitter(a_series, noise_reduction=1000000):
    return a_series + (np.random.random(len(a_series)) * a_series.std() / noise_reduction) - (
                a_series.std() / (2 * noise_reduction))


def make_numeric_bins(data_col, n_bins=5, intervals_only=True, binning='ef'):
    numeric_selectors = []

    if binning == 'ew':
        _, cut_points = pd.cut(data_col.values, n_bins+2, retbins=True, precision=3)
    elif binning == 'ef':
        _, cut_points = pd.qcut(data_col.values, n_bins+2, retbins=True, precision=3, duplicates='drop')
    else:
        msg = "Binning strategy must be equal width of equal frequency: ['ew', 'ef]"
        print(msg)

    cut_points = cut_points.tolist()[1:-1]

    if intervals_only:
        old_cutpoint = float("-inf")
        for c in cut_points:
            numeric_selectors.append(IntervalSelector(data_col.name, old_cutpoint, c))
            old_cutpoint = c
        numeric_selectors.append(IntervalSelector(data_col.name, old_cutpoint, float("inf")))
    #             numeric_selectors.append(IntervalSelector(attr_name, float("-inf"), cutpoints[0]))
    #             for i, c1 in enumerate(cutpoints):
    #                 for c2 in cutpoints[(i+1):]:
    # #                 for c2 in cutpoints[(i+1):][::-1]:
    #                     numeric_selectors.append(IntervalSelector(attr_name, c1, c2))
    #             numeric_selectors.append(IntervalSelector(attr_name, c1, float("inf")))
    else:
        for c in cut_points:
            numeric_selectors.append(IntervalSelector(data_col.name, float("-inf"), c))
            numeric_selectors.append(IntervalSelector(data_col.name, c, float("inf")))

    # numeric_selectors = random.sample(numeric_selectors, len(numeric_selectors))
    return numeric_selectors


def make_nominal_bins(data_col):
    nominal_selectors = []
    for val in pd.unique(data_col):
        nominal_selectors.append(EqualitySelector(data_col.name, val))
    return nominal_selectors