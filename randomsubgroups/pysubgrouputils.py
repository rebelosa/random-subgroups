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
import pysubgroup as ps
from pysubgroup import defaultdict, NumericTarget, BinaryTarget
from scipy.stats import entropy

# Create a new Numeric QM by cloning the StandardQFNumeric and then define the differences
setattr(ps, 'AbsoluteQFNumeric', ps.StandardQFNumeric)


def standard_qf_numeric(a, _, mean_dataset, instances_subgroup, mean_sg):
    return instances_subgroup ** a * np.abs(mean_sg - mean_dataset)


setattr(ps.AbsoluteQFNumeric, 'standard_qf_numeric', standard_qf_numeric)


class StaticSpecializationOperatorWithMaxFeatures:
    """
        The StaticSpecializationOperatorWithMaxFeatures is an adapted version of the original
        StaticSpecializationOperator which randomly selects a subset of the search space
        when max_features is not None.

        It can also allow the reuse of the same attributes when fine_search is True
    """

    def __init__(self, selectors, max_features=None, fine_search=False):

        self.max_features = max_features
        self.fine_search = fine_search

        self.search_space_attributes = defaultdict(list)
        if self.fine_search:
            for selector in selectors:
                self.search_space_attributes[selector.__repr__()].append([selector])
        else:
            for selector in selectors:
                self.search_space_attributes[selector.attribute_name].append([selector])

    def refinements(self, subgroup):

        if self.max_features:
            if self.max_features > 0:
                if self.fine_search:
                    subgroup_attributes = [s.__repr__() for s in subgroup._selectors]
                else:
                    subgroup_attributes = [s.attribute_name for s in subgroup._selectors]
                subset_search_space = [att for att in self.search_space_attributes.keys() if
                                       att not in subgroup_attributes]

                if len(subset_search_space) > 0:
                    max_subset = max(1, int(np.sqrt(len(subset_search_space))))
                    subset_attributes = random.sample(subset_search_space, max_subset)
                    search_space = [self.search_space_attributes[k] for k in subset_attributes]
                else:
                    search_space = []
            else:
                msg = "'max_features' must be an int > 0 "
                raise ValueError(msg)

        elif subgroup.depth > 0:
            if self.fine_search:
                index_of_last = list(self.search_space_attributes.keys()).index(subgroup._selectors[-1].__repr__())
            else:
                index_of_last = list(self.search_space_attributes.keys()).index(subgroup._selectors[-1].attribute_name)
            search_space = islice(self.search_space_attributes.values(), index_of_last + 1, None)
        else:
            search_space = self.search_space_attributes.values()

        new_selectors = chain.from_iterable(search_space)

        return (subgroup & sel for sel in new_selectors)


class DynamicSpecializationOperatorWithMaxFeatures:
    """
        This StaticSpecializationOperator is the same as the original
        but it searches for subsets of the attribute when given by max_features
    """
    def __init__(self,
                 data,
                 min_size=2,
                 n_bins=5,
                 max_features=None,
                 intervals_only=True):
        self.data = data
        self.min_size = min_size
        self.n_bins = n_bins
        self.max_features = max_features
        self.intervals_only = intervals_only

    def refinements(self, subgroup):

        if self.max_features:
            subset_attributes = random.sample(list(self.data.drop(['target'], axis=1).columns), self.max_features)
            # subset_attributes.append('target')
            data_subset = self.data[subgroup.covers(self.data)][subset_attributes]
        else:
            data_subset = self.data.iloc[subgroup.covers(self.data), :]

        selectors = ps.create_selectors(data_subset,
                                        nbins=self.n_bins, ignore=['target'],
                                        intervals_only=self.intervals_only)

        search_space_dict = defaultdict(list)
        for selector in selectors:
            search_space_dict[selector.attribute_name].append(selector)

        # if self.max_features:
        #     if self.max_features > 0:
        #         subgroup_attributes = [s.attribute_name for s in subgroup._selectors]
        #         subset_search_space = [att for att in search_space_dict.keys() if
        #                                att not in subgroup_attributes]
        #         # subset_attributes = random.sample(subset_search_space, self.max_features)
        #         if len(subset_search_space) > 0:
        #             subset_attributes = random.sample(subset_search_space, 1)
        #             # print(subset_attributes)
        #             search_space = [search_space_dict[k] for k in subset_attributes]
        #         else:
        #             search_space = []
        #     else:
        #         msg = "'max_features' must be an int > 0 "
        #         raise ValueError(msg)

        if subgroup.depth > 0 and not self.max_features:
            index_of_last = list(search_space_dict.keys()).index(subgroup._selectors[-1].attribute_name)
            #             search_space = [att for i, att in enumerate(self.search_space_attributes.values()) if i > index_of_last]
            search_space = islice(search_space_dict.values(), index_of_last + 1, None)
        else:
            search_space = search_space_dict.values()

        new_selectors = chain.from_iterable(search_space)

        return (subgroup & sel for sel in new_selectors)


class LightBestFirstSearch:
    """
    This LightBestFirstSearch is a lighter version of the BestFirstSearch from pysubgroup package
    and it uses both:
     - StaticSpecializationOperatorWithMaxFeatures()
     - DynamicSpecializationOperatorWithMaxFeatures()
    instead of StaticSpecializationOperator()
    """
    def __init__(self,
                 max_features=None,
                 n_bins=5,
                 intervals_only=True,
                 fine_search=False,
                 discretization="static"):
        self.max_features = max_features
        self.n_bins = n_bins
        self.intervals_only = intervals_only
        self.fine_search = fine_search
        self.discretization = discretization

    def execute(self, task):
        result = []
        queue = [(float("-inf"), ps.Conjunction([]))]
        if self.discretization == "static":
            operator = StaticSpecializationOperatorWithMaxFeatures(task.search_space, max_features=self.max_features,
                                                                   fine_search=self.fine_search)
        elif self.discretization == "dynamic":
            operator = DynamicSpecializationOperatorWithMaxFeatures(data=task.data, min_size=2,
                                                     n_bins=self.n_bins, max_features=self.max_features,
                                                     intervals_only=self.intervals_only)
        else:
            msg = "Discretization type not recognized. Only 'static' or 'dynamic' can be used"
            raise ValueError(msg)

        task.qf.calculate_constant_statistics(task.data, task.target)
        while queue:
            q, old_description = heappop(queue)
            q = -q
            if not q > ps.minimum_required_quality(result, task):
                break
            for candidate_description in operator.refinements(old_description):
                statistics = task.qf.calculate_statistics(candidate_description, task.target, task.data)
                if ps.constraints_satisfied(task.constraints_monotone, candidate_description, statistics,
                                            task.data):
                    score_eval = task.qf.evaluate(candidate_description, task.target, task.data, statistics)
                    if score_eval > q or q == float("inf"):
                        ps.add_if_required(result, candidate_description, score_eval, task, statistics=statistics)
                        if len(candidate_description) < task.depth:
                            heappush(queue, (-score_eval, candidate_description))

        result.sort(key=lambda x: x[0], reverse=True)
        return ps.SubgroupDiscoveryResult(result, task)


def kl_divergence(data_target, subgroup_target):
    epsilon = 0.00001

    pk = np.histogram(subgroup_target)[0] + epsilon
    pk = pk / sum(pk)
    qk = np.histogram(data_target)[0] + epsilon
    qk = qk / sum(qk)
    return entropy(pk, qk)


class KLDivergenceNumeric(ps.BoundedInterestingnessMeasure):
    tpl = namedtuple('KLDivergenceNumeric_parameters', ('size', 'target', 'estimate'))

    @staticmethod
    def weighted_mutual_info_score(a, _, target_dataset, instances_subgroup, target_subgroup):
        return instances_subgroup ** a * kl_divergence(target_dataset, target_subgroup)
        # return instances_subgroup ** a * (mean_sg - mean_dataset)

    def __init__(self, a, invert=False):
        self.a = a
        self.invert = invert
        self.required_stat_attrs = ('size', 'mean')
        self.dataset_statistics = None
        self.all_target_values = None
        self.has_constant_statistics = False
        # self.estimator = KLDivergenceNumeric.KLEstimator(self)

    def calculate_constant_statistics(self, data, target):
        # data = self.estimator.get_data(data, target)
        self.all_target_values = data[target.target_variable].to_numpy()
        data_size = len(data)
        target_data = self.all_target_values
        self.dataset_statistics = KLDivergenceNumeric.tpl(data_size, target_data, None)
        self.has_constant_statistics = True

    def evaluate(self, subgroup, target, data, statistics=None):
        cover_arr, sg_size = ps.get_cover_array_and_size(subgroup, len(self.all_target_values), data)
        if sg_size > 0:
            sg_target_values = self.all_target_values[cover_arr]
        else:
            sg_target_values = []
        # statistics = self.ensure_statistics(subgroup, target, data, statistics)
        dataset = self.dataset_statistics
        return KLDivergenceNumeric.weighted_mutual_info_score(self.a, dataset.size, dataset.target, sg_size,
                                                              sg_target_values)

    def calculate_statistics(self, subgroup, target, data, statistics=None):
        cover_arr, sg_size = ps.get_cover_array_and_size(subgroup, len(self.all_target_values), data)
        if sg_size > 0:
            sg_target_values = self.all_target_values[cover_arr]
            # sg_mean = np.mean(sg_target_values)
            # estimate = self.estimator.get_estimate(subgroup, sg_size, sg_mean, cover_arr, sg_target_values)
            estimate = float('inf')
        else:
            sg_target_values = []
            estimate = float('-inf')
        return KLDivergenceNumeric.tpl(sg_size, sg_target_values, estimate)

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        statistics = self.ensure_statistics(subgroup, target, data, statistics)
        return statistics.estimate
        # return float('-inf')


    # class KLEstimator:
    #     def __init__(self, qf):
    #         self.qf = qf
    #         self.indices_greater_mean = None
    #         self.target_values_greater_mean = None

        # def get_data(self, data, target):
        #     return data

        # def calculate_constant_statistics(self, data, target):  # pylint: disable=unused-argument
        #     self.indices_greater_mean = self.qf.all_target_values > self.qf.dataset_statistics.mean
        #     self.target_values_greater_mean = self.qf.all_target_values#[self.indices_greater_mean]

        # def get_estimate(self, subgroup, sg_size, sg_mean, cover_arr, _):  # pylint: disable=unused-argument
        # larger_than_mean = self.target_values_greater_mean[cover_arr][self.indices_greater_mean[cover_arr]]
        # size_greater_mean = len(larger_than_mean)
        # sum_greater_mean = np.sum(larger_than_mean)
        #
        # return sum_greater_mean - size_greater_mean * self.qf.dataset_statistics.mean
        # return float('inf')


setattr(ps, 'KLDivergenceNumeric', KLDivergenceNumeric)

