# --------------------------------------------------------
# Author: Tenphun0503
# A decision tree and a random forest for Pokemon problem
# --------------------------------------------------------

import random
import pandas as pd
from sklearn import tree
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt

data_file = '../public/Pokemon.csv'
data = pd.read_csv(data_file)

# Split features
X_raw = data.iloc[:, 2:11]
# Replace blank in Type2 with None
X_raw['Type 2'].fillna(value='None', inplace=True)
# One-hot encoding for 'Type1' and 'Type2'
X = pd.get_dummies(X_raw)
# Split labels
y = data.iloc[:, -1]
# Initialize a Decision Tree
dtc = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10)

# Initialize a Random Forest
rfc = ensemble.RandomForestClassifier(criterion='entropy', max_depth=10)


def test(clf, _data, _X, _y, _id):
    clf = clf.fit(_X, _y)
    sample = _X.iloc[_id, :].values.reshape(1, -1)
    sample = pd.DataFrame(sample)
    sample.columns = X.columns.tolist()
    print('test_index: ', _id + 2)
    print('index: ', _data.iloc[_id, 0])
    print('name: ', _data.iloc[_id, 1])
    print('test sample: ', sample.values.tolist())
    print('test result:', clf.predict(sample))
    print('-----------------------------------------------')


# test single sample
index = random.randint(0, 799)
test(dtc, data, X, y, index)
test(rfc, data, X, y, index)

# shows the mean of accuracy of cross validation
print('Decision Tree Classifier: %.3f' % cross_val_score(dtc, X, y, cv=10).mean())
print('Random Forest Classifier: %.3f' % cross_val_score(rfc, X, y, cv=10).mean())

# shows the decision tree
fig = plt.figure(figsize=(25, 20))
_ = tree.plot_tree(
    dtc,
    feature_names=X.columns.tolist(),
    class_names=['False', 'True'],
    filled=True,
    max_depth=3
)

fig.savefig('decision tree.png')