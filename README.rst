.. -*- mode: rst -*-


random-subgroups package - Making predictions with subgroups
============================================================

.. _scikit-learn: https://scikit-learn.org

**random-subgroups** is a machine learning package compatible with scikit-learn_.

It uses ensembles of weak estimators, as in random forests, for classification and
regression tasks. The main difference from the random forests algorithm is that
it uses subgroups as estimators.

.. _pysubgroup: https://github.com/flemmerich/pysubgroup/

The subgroup discovery implementation of this package is made on top of the pysubgroup_ package. It
uses many of the features from **pysubgroup** but it also extends it with different quality
measures (more suitable for prediction) and different search strategies.


Example of usage:

.. code-block:: python
    
    import pandas as pd
    from randomsubgroups import RandomSubgroupClassifier

    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    data = datasets.load_breast_cancer()
    y = data.target
    X = data.data

    sg_classifier = RandomSubgroupClassifier(n_estimators=300)

    sg_classifier.fit(X, y)

