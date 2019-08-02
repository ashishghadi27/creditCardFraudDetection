import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

data = pd.read_csv('creditcard.csv')

print(data.columns)
data = data.sample(frac=1, random_state = 1)
print(data.shape)
print(data.describe())

data.hist(figsize = (10, 10), color= "green")
plt.show()

Fraud = data[data['Class'] == 1]
Valid = data[data['Class'] == 0]

outlier_fraction = len(Fraud)/float(len(Valid))
print(outlier_fraction)

print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
print('Valid Transactions: {}'.format(len(data[data['Class'] == 0])))

corrmat = data.corr()
fig = plt.figure(figsize = (12, 9))

sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()

columns = data.columns.tolist()

columns = [c for c in columns if c not in ["Class"]]

target = "Class"

X = data[columns]
Y = data[target]

validation_size = 0.10
seed = 9
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=validation_size, random_state = seed)

print("SHAPES")

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

print(X.shape)
print(Y.shape)

from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

state = 1

classifiers = {
    "Isolation Forest": IsolationForest(max_samples=len(X_train),
                                        contamination=outlier_fraction,
                                        random_state=state),
    "Local Outlier Factor": LocalOutlierFactor(
        n_neighbors=20,
        contamination=outlier_fraction)}

plt.figure(figsize=(9, 7))
n_outliers = len(Fraud)

for i, (class_name, classifier) in enumerate(classifiers.items()):

    if class_name == "Local Outlier Factor":
        y_pred = classifier.fit_predict(X_train)
        scores_pred = classifier.negative_outlier_factor_
    else:
        classifier.fit(X_train)
        scores_pred = classifier.decision_function(X_train)
        y_pred = classifier.predict(X_train)


    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1

    n_errors = (y_pred != Y_train).sum()

    print('{}: {}'.format(class_name, n_errors))
    print(accuracy_score(Y_train, y_pred))
    print(classification_report(Y_train, y_pred))

print("PREDICTION")

cl = IsolationForest(max_samples=len(X_test),
                                        contamination=outlier_fraction,  random_state=state)

cl.fit(X_test)

scores_prednew = cl.decision_function(X_test)
y_prednew = cl.predict(X_test)
print(scores_prednew)
print(y_prednew)
output_list = list(y_prednew)
input_list = list(scores_prednew)
for i in range(0, len(output_list)):
    if(output_list[i]==-1):
        print("Fraud Transaction Found: ",input_list[i])

