import numpy as np
from pysubgroup import ps
from pysubgroup.subgroup_description import Conjunction


def encode_subgroup(decoded_subgroup):

    conjunction = []
    for attribute_name, cond in decoded_subgroup['conditions'].items():

        selector_type = cond['selector_type']
        if selector_type == 'IntervalSelector':
            conjunction.append(ps.subgroup_description.IntervalSelector(attribute_name,
                                                                        lower_bound=cond['lower_bound'],
                                                                        upper_bound=cond['upper_bound']))
        elif selector_type == 'EqualitySelector':
            for att_value in cond['attribute_value']:
                conjunction.append(ps.subgroup_description.EqualitySelector(attribute_name, att_value))
        else:
            msg = "Unknown pysubgroup Selector type"
            raise ValueError(msg)

    conjunction = ps.subgroup_description.Conjunction(conjunction)

    return tuple((decoded_subgroup['score'], conjunction))


class SubgroupPredictor(Conjunction):
    """
    Class for the Subgroup predictors.

    """
    def __init__(self,
                 subgroup,
                 target,
                 alternative_target=None,
                 predict_complement=False
                 ):
        Conjunction.__init__(self, subgroup[1].selectors)

        self.target = target
        self.alternative_target = alternative_target
        self.predict_complement = predict_complement
        self.score = subgroup[0]
        if predict_complement and alternative_target is None:
            msg = "Cannot predict complement if alternative target is not given"
            raise ValueError(msg)

    @classmethod
    def from_dict(cls, decoded_subgroup):
        subgroup = encode_subgroup(decoded_subgroup)
        target = decoded_subgroup['target']

        return cls(subgroup, target=target, alternative_target=decoded_subgroup['alternative_target'])

    def predict(self, x, predict_complement=False):
        """
        Predict class for X.

        The predicted class of an input sample are computed as
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
        # check_is_fitted(self)

        predictions = np.array([None] * x.shape[0])
        covered_ids = self.covers(x)
        predictions[covered_ids] = self.target
        if predict_complement:
            predictions[list(np.invert(covered_ids))] = self.alternative_target

        return predictions

    def to_dict(self):

        decoded_subgroup = {'description': self.__str__(), 'conditions': {},
                            'score': self.score, 'target': self.target, 'alternative_target': self.alternative_target}

        for condition in self._selectors:

            already_exists = condition._attribute_name in decoded_subgroup['conditions'].keys()
            if issubclass(type(condition), ps.subgroup_description.IntervalSelector):
                if already_exists:
                    stored_condition = decoded_subgroup['conditions'][condition._attribute_name]
                    lower_bound = max([condition._lower_bound, stored_condition['lower_bound']])
                    upper_bound = min([condition._upper_bound, stored_condition['upper_bound']])
                else:
                    lower_bound = condition._lower_bound
                    upper_bound = condition._upper_bound

                decoded_subgroup['conditions'].update({condition._attribute_name: {'lower_bound': lower_bound,
                                                                                   'upper_bound': upper_bound,
                                                                                   'selector_type': 'IntervalSelector'}})

            elif issubclass(type(condition), ps.subgroup_description.EqualitySelector):
                if already_exists and decoded_subgroup['conditions'][condition._attribute_name]['selector_type'] == 'EqualitySelector':
                    decoded_subgroup['conditions'][condition._attribute_name]['attribute_value'].append(
                        condition._attribute_value)
                else:
                    decoded_subgroup['conditions'].update(
                        {condition._attribute_name: {'attribute_value': [condition._attribute_value],
                                                     'selector_type': 'EqualitySelector'}})

        return decoded_subgroup

    def get_features(self):

        attribute_list = []
        for condition in self._selectors:
            if condition._attribute_name not in attribute_list:
                attribute_list.append(condition._attribute_name)
        return attribute_list

    # def merge(self, other_subgroup):
    #
    #     if self.target == other_subgroup.target:
    #         other_attributes = [condition._attribute_name for condition in other_subgroup._selectors]
    #         self_attributes = [condition._attribute_name for condition in self._selectors]
    #
    #         if np.all([att in other_attributes for att in self_attributes]):
    #             main_subgroup = other_subgroup.to_dict()
    #             secondary_subgroup = self
    #             is_contained = True
    #         elif np.all([att in self_attributes for att in other_attributes]):
    #             main_subgroup = self.to_dict()
    #             secondary_subgroup = other_subgroup
    #             is_contained = True
    #         else:
    #             is_contained = False
    #
    #         if is_contained:
    #             conjunction = []
    #             for cond in main_subgroup['conditions'].values():
    #
    #                 attribute_name = cond['attribute_name']
    #                 selector_type = cond['selector_type']
    #
    #                 # Search for the same attribute in self.
    #                 passed = False
    #                 self_condition = None
    #                 for a in secondary_subgroup.selectors:
    #                     if a._attribute_name == attribute_name:
    #                         self_condition = a
    #                         passed = True
    #
    #                 if not passed:
    #                     continue
    #
    #                 if selector_type == 'IntervalSelector':
    #
    #                     lower_bound = np.min([self_condition._lower_bound, cond['lower_bound']])
    #                     upper_bound = np.max([self_condition._upper_bound, cond['upper_bound']])
    #                     conjunction.append(ps.subgroup_description.IntervalSelector(attribute_name,
    #                                                                                 lower_bound=lower_bound,
    #                                                                                 upper_bound=upper_bound))
    #                 elif selector_type == 'EqualitySelector':
    #                     att_value = cond['attribute_value']
    #                     if att_value==self_condition._attribute_value:
    #                         conjunction.append(ps.subgroup_description.EqualitySelector(attribute_name,
    #                                                                                 attribute_value=att_value))
    #                 else:
    #                     msg = "Unknown pysubgroup Selector type"
    #                     raise ValueError(msg)
    #
    #             conjunction = ps.subgroup_description.Conjunction(conjunction)
    #             subgroup = tuple((main_subgroup['score'], conjunction))
    #
    #             return SubgroupPredictor(subgroup, target=self.target)