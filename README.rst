.. -*- mode: rst -*-


random-subgroups package - Making predictions with subgroups
============================================================

.. _scikit-learn: https://scikit-learn.org

**random-subgroups** is a machine learning package compatible with scikit-learn_.

It uses ensembles of weak estimators, as in random forests, for classification and
regression tasks. The main difference from the other well-known approaches is that
it uses subgroups as estimators.

.. _pysubgroup: https://github.com/flemmerich/pysubgroup/

The subgroup discovery implementation of this package is made on top of the pysubgroup_ package. It
uses many of the features from **pysubgroup** but it also extends it with different quality
measures (more suitable for prediction) and different search strategies (e.g. RandomSubsetSearch())

Example of usage:
```python
from randomsubgroups import RandomSubgroupClassifier

# Load the example dataset
from pysubgroup.tests.DataSets import get_titanic_data
data = get_titanic_data()

target = ps.BinaryTarget ('Survived', True)
searchspace = ps.create_selectors(data, ignore=['Survived'])
task = ps.SubgroupDiscoveryTask (
    data,
    target,
    searchspace,
    result_set_size=5,
    depth=2,
    qf=ps.WRAccQF())
result = ps.BeamSearch().execute(task)
```
