from pysubgroup.subgroup_description import Conjunction
from pysubgroup import ps
import numpy as np


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
                                                                        attribute_value=cond[
                                                                            'attribute_value']))
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
                 predict_complement=False
                 ):
        Conjunction.__init__(self, subgroup[1].selectors)

        # self.selectors = selectors
        self.predict_complement = predict_complement
        self.score = subgroup[0]

        if isinstance(target, int):
            self.target = target
        elif target.shape[0] == 2:
            self.target_complement = target[1]
            self.target = target[0]
        else:
            raise ValueError("Unknown subgroup estimator target shape.")
        self.original_target = target

    @classmethod
    def from_dict(cls, decoded_subgroup):
        subgroup = encode_subgroup(decoded_subgroup)
        target = decoded_subgroup['target']

        return cls(subgroup, target)

    def predict(self, x):
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
        if not self.predict_complement:
            predictions[self.covers(x)] = self.target
        else:
            covered_ids = self.covers(x)
            predictions[covered_ids] = self.target
            predictions[not covered_ids] = self.target_complement

        return predictions

    def to_dict(self):

        decoded_subgroup = {'description': self.__str__(), 'conditions': {},
                            'score': self.score, 'target': self.original_target}

        for i in range(len(self._selectors)):

            condition = self._selectors[i]

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