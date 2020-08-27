import pysubgroup as ps

from itertools import chain
from heapq import heappop, heappush
import random


def encodeSubgroup(decodedSubgroup):

    #score = decodedSubgroup['score']

    conjunction = []
    for cond in decodedSubgroup['conditions'].values():

        attribute_name = cond['attribute_name']
        selectorType = cond['selector_type']

        if selectorType == 'IntervalSelector':
            conjunction.append(ps.subgroup.IntervalSelector(attribute_name,
                                                            lower_bound=cond['lower_bound'],
                                                            upper_bound=cond['upper_bound']))
        elif selectorType == 'EqualitySelector':
            conjunction.append(ps.subgroup.EqualitySelector(attribute_name,
                                                            attribute_value=cond['attribute_value']))
        else:
            msg = "Unknown pysubgroup Selector type"
            raise ValueError(msg)

    subgroup = ps.boolean_expressions.Conjunction(conjunction)

    return subgroup


def decodeSubgroup(subgroup):

    subg = {'description': str(subgroup[1]), 'conditions': {}, 'score': subgroup[0]}

    for i in range(len(subgroup[1]._selectors)):

        condition = subgroup[1]._selectors[i]

        subg['conditions'].update({i: {'attribute_name': condition._attribute_name}})

        if issubclass(type(condition), ps.subgroup.IntervalSelector):
            subg['conditions'][i].update({'lower_bound': condition._lower_bound,
                                          'upper_bound': condition._upper_bound})

            selector_type = 'IntervalSelector'
        elif issubclass(type(condition), ps.subgroup.EqualitySelector):
            subg['conditions'][i].update({'attribute_value': condition._attribute_value})

            selector_type = 'EqualitySelector'

        subg['conditions'][i].update({'selector_type': selector_type})

    return subg

# Extending the class pysubgroup

# Create a new Numeric QM by cloning the StandardQFNumeric and then define the differences


setattr(ps, 'AbsoluteQFNumeric', ps.StandardQFNumeric)


def standard_qf_numeric(a, _, mean_dataset, instances_subgroup, mean_sg):
    print(mean_sg)
    print(mean_dataset)
    return instances_subgroup ** a * np.abs(mean_sg - mean_dataset)


setattr(ps.AbsoluteQFNumeric, 'standard_qf_numeric', standard_qf_numeric)


def refinements_random_subset(self, subgroup, prob):
    search_space_index = list(self.search_space_index.values())
    subset_size = int(prob * len(self.search_space_index))
    index_random_subset = random.sample(search_space_index, subset_size)

    random_subset = [self.search_space[i] for i in index_random_subset]

    new_selectors = chain.from_iterable(random_subset)

    return (subgroup & sel for sel in new_selectors)


setattr(ps.StaticSpecializationOperator, 'refinements_random_subset', refinements_random_subset)


class RandomSubsetSearch:
    def __init__(self,
                 prob=0.5):
        self.prob = prob

    def execute(self, task):
        result = []
        queue = [(float("-inf"), ps.Conjunction([]))]
        operator = ps.StaticSpecializationOperator(task.search_space)
        task.qf.calculate_constant_statistics(task)
        while queue:
            q, old_description = heappop(queue)
            q = -q
            if not (q > ps.minimum_required_quality(result, task)):
                break
            for candidate_description in operator.refinements_random_subset(old_description, self.prob):
                sg = candidate_description
                statistics = task.qf.calculate_statistics(sg, task.data)
                ps.add_if_required(result, sg, task.qf.evaluate(sg, statistics), task)
                optimistic_estimate = task.qf.optimistic_estimate(sg, statistics)

                # compute refinements and fill the queue
                if len(candidate_description) < task.depth and optimistic_estimate >= ps.minimum_required_quality(
                        result, task):
                    heappush(queue, (-optimistic_estimate, candidate_description))

        result.sort(key=lambda x: x[0], reverse=True)
        return ps.SubgroupDiscoveryResult(result, task)


setattr(ps, 'RandomSubsetSearch', RandomSubsetSearch)