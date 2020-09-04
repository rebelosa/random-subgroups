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
measures (more suitable for prediction) and different search strategies (e.g. RandomSubsetSearch())

```python
# more python code

from randomsubgroups import RandomSubgroupClassifier

```


Example of usage:

```python

from randomsubgroups import RandomSubgroupClassifier

from sklearn import datasets
from sklearn.model_selection import train_test_split

data = datasets.load_breast_cancer()
cols = data.feature_names
target = pd.Series(data.target)
data = pd.DataFrame(data.data)
data.columns = cols

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size = 0.3)

sg_classifier = RandomSubgroupClassifier(n_estimators=300)

sg_classifier.fit(X_train, y_train)

sg_classifier.predict(X_test)


from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix

train_preds = sgclassifier.predict(X_train)
test_preds = sgclassifier.predict(X_test)

print("Accuracy train:", accuracy_score(y_true = y_train, y_pred = train_preds))
print("Accuracy test:", accuracy_score(y_true = y_test, y_pred = test_preds))

print("Precision train:", precision_score(y_true = y_train, y_pred = train_preds))
print("Precision test:", precision_score(y_true = y_test, y_pred = test_preds))
print("Recall test:", recall_score(y_true = y_test, y_pred = test_preds))

train_probs = sgclassifier.predict_proba(X_train)[:,1]
test_probs = sgclassifier.predict_proba(X_test)[:,1]

print("ROC AUC train:", roc_auc_score(y_train, train_probs))
print("ROC AUC test: ", roc_auc_score(y_test, test_probs))

cm = confusion_matrix(y_test, test_preds)
cm
```
