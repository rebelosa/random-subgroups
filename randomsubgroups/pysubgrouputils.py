import pysubgroup as ps

from itertools import chain
from heapq import heappop, heappush
import random
import numpy as np
from pysubgroup import defaultdict


def encode_subgroup(decoded_subgroup):

    # score = decodedSubgroup['score']

    conjunction = []
    for cond in decoded_subgroup['conditions'].values():

        attribute_name = cond['attribute_name']
        selector_type = cond['selector_type']

        if selector_type == 'IntervalSelector':
            conjunction.append(ps.subgroup_description.IntervalSelector(attribute_name,
                               lower_bound=cond['lower_bound'],
                               upper_bound=cond['upper_bound']))
        elif selector_type == 'EqualitySelector':
            conjunction.append(ps.subgroup_description.EqualitySelector(attribute_name,
                               attribute_value=cond['attribute_value']))
        else:
            msg = "Unknown pysubgroup Selector type"
            raise ValueError(msg)

    subgroup = ps.subgroup_description.Conjunction(conjunction)

    return subgroup


def decode_subgroup(subgroup):

    decoded_subgroup = {'description': str(subgroup[1]), 'conditions': {}, 'score': subgroup[0]}

    for i in range(len(subgroup[1]._selectors)):

        condition = subgroup[1]._selectors[i]

        decoded_subgroup['conditions'].update({i: {'attribute_name': condition._attribute_name}})

        if issubclass(type(condition), ps.subgroup_description.IntervalSelector):
            decoded_subgroup['conditions'][i].update({'lower_bound': condition._lower_bound,
                                                      'upper_bound': condition._upper_bound})

            selector_type = 'IntervalSelector'
        elif issubclass(type(condition), ps.subgroup_description.EqualitySelector):
            decoded_subgroup['conditions'][i].update({'attribute_value': condition._attribute_value})

            selector_type = 'EqualitySelector'

        decoded_subgroup['conditions'][i].update({'selector_type': selector_type})

    return decoded_subgroup

# Extending the class pysubgroup

# Create a new Numeric QM by cloning the StandardQFNumeric and then define the differences


setattr(ps, 'AbsoluteQFNumeric', ps.StandardQFNumeric)


def standard_qf_numeric(a, _, mean_dataset, instances_subgroup, mean_sg):
    return instances_subgroup ** a * np.abs(mean_sg - mean_dataset)


setattr(ps.AbsoluteQFNumeric, 'standard_qf_numeric', standard_qf_numeric)


class StaticSpecializationOperator2:
    """
        This StaticSpecializationOperator is the same as the original
        but it creates a search space that is more fine grained per attribute
        (for this reason is much slower)
    """
    def __init__(self, selectors):
        search_space_dict = defaultdict(list)
        for selector in selectors:
            search_space_dict[selector.__repr__()].append(selector)
        self.search_space = list(search_space_dict.values())
        self.search_space_index = {key: i for i, key in enumerate(search_space_dict.keys())}

    def refinements(self, subgroup):
        if subgroup.depth > 0:
            index_of_last = self.search_space_index[subgroup._selectors[-1].__repr__()]
            new_selectors = chain.from_iterable(self.search_space[index_of_last + 1:])
        else:
            new_selectors = chain.from_iterable(self.search_space)

        return (subgroup & sel for sel in new_selectors)


setattr(ps, 'StaticSpecializationOperator2', StaticSpecializationOperator2)


class BestFirstSearch2:
    """
    This BestfirstSearch is the same as the original
    but it used StaticSpecializationOperator2() instead of StaticSpecializationOperator()
    """
    def execute(self, task):
        result = []
        queue = [(float("-inf"), ps.Conjunction([]))]
        operator = ps.StaticSpecializationOperator2(task.search_space)
        task.qf.calculate_constant_statistics(task.data, task.target)
        while queue:
            q, old_description = heappop(queue)
            q = -q
            if not q > ps.minimum_required_quality(result, task):
                break
            for candidate_description in operator.refinements(old_description):
                sg = candidate_description
                statistics = task.qf.calculate_statistics(sg, task.target, task.data)
                ps.add_if_required(result, sg, task.qf.evaluate(sg, task.target, task.data, statistics), task, statistics=statistics)
                if len(candidate_description) < task.depth:
                    optimistic_estimate = task.qf.optimistic_estimate(sg, task.target, task.data, statistics)

                    # compute refinements and fill the queue
                    if optimistic_estimate >= ps.minimum_required_quality(result, task):
                        if ps.constraints_satisfied(task.constraints_monotone, candidate_description, statistics, task.data):
                            heappush(queue, (-optimistic_estimate, candidate_description))

        result.sort(key=lambda x: x[0], reverse=True)
        return ps.SubgroupDiscoveryResult(result, task)


setattr(ps, 'BestFirstSearch2', BestFirstSearch2)


class StaticSpecializationOperator3:
    """
        This StaticSpecializationOperator is the same as the original
        but it searches for subsets of the attribute, as given by max_features
    """
    def __init__(self, selectors, max_features=None):
        self.search_space_dict = defaultdict(list)
        self.search_space_attributes = defaultdict(list)
        for i, selector in enumerate(selectors):
            self.search_space_dict[selector.__repr__()].append(selector)
            self.search_space_attributes[selector.attribute_name].append([selector])
        self.search_space = list(self.search_space_dict.values())
        self.search_space_index = {key: i for i, key in enumerate(self.search_space_dict.keys())}
        self.search_space_attributes_index = {key: i for i, key in enumerate(self.search_space_attributes.keys())}
        self.max_features = max_features

    def refinements(self, subgroup):
        search_space = self.search_space

        if self.max_features:
            if self.max_features > 0:
                search_space_ids = random.sample(list(self.search_space_attributes_index), self.max_features)
                search_space = [self.search_space_attributes[i] for i in search_space_ids]
            else:
                msg = "'max_features' must be an int > 0 "
                raise ValueError(msg)

        elif subgroup.depth > 0:
            index_of_last = self.search_space_index[subgroup.__repr__()]
            search_space = search_space[index_of_last + 1:]

        new_selectors = chain.from_iterable(search_space)

        return (subgroup & sel for sel in new_selectors)


setattr(ps, 'StaticSpecializationOperator3', StaticSpecializationOperator3)


class BestFirstSearch3:
    """
    This BestfirstSearch is the same as the original
    but it used StaticSpecializationOperator3() instead of StaticSpecializationOperator()
    """
    def __init__(self,
                 max_features=None):
        self.max_features = max_features

    def execute(self, task):
        result = []
        queue = [(float("-inf"), ps.Conjunction([]))]
        operator = ps.StaticSpecializationOperator3(task.search_space, max_features=self.max_features)
        task.qf.calculate_constant_statistics(task.data, task.target)
        while queue:
            q, old_description = heappop(queue)
            q = -q
            if not q > ps.minimum_required_quality(result, task):
                break
            for candidate_description in operator.refinements(old_description):
                sg = candidate_description
                statistics = task.qf.calculate_statistics(sg, task.target, task.data)
                ps.add_if_required(result, sg, task.qf.evaluate(sg, task.target, task.data, statistics), task,
                                   statistics=statistics)
                if len(candidate_description) < task.depth:
                    optimistic_estimate = task.qf.optimistic_estimate(sg, task.target, task.data, statistics)

                    # compute refinements and fill the queue
                    if optimistic_estimate >= ps.minimum_required_quality(result, task):
                        if ps.constraints_satisfied(task.constraints_monotone, candidate_description, statistics,
                                                    task.data):
                            heappush(queue, (-optimistic_estimate, candidate_description))

        result.sort(key=lambda x: x[0], reverse=True)
        return ps.SubgroupDiscoveryResult(result, task)


setattr(ps, 'BestFirstSearch3', BestFirstSearch3)