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