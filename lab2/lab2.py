import matplotlib.pyplot as plt
import numpy as np
import scikitplot as skplt
from pandas import DataFrame
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.decomposition import PCA

digits = datasets.load_digits()
digits_X, digits_y = datasets.load_digits(return_X_y=True)

fig = plt.figure()
fig.subplots_adjust()

for i in range(9):
    ax = fig.add_subplot(3, 3, i + 1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt.cm.gray_r, interpolation='nearest')
    ax.text(0.1, 0.7, str(digits_y[i]))
plt.show()

pca = PCA(n_components=2)
pca.fit(digits_X)

skplt.decomposition.plot_pca_2d_projection(pca, digits_X, digits_y, title='Digits 2-D Projection')

moons_X, moons_y = datasets.make_moons(n_samples=1200, noise=0.3)

df = DataFrame(dict(x=moons_X[:, 0], y=moons_X[:, 1], label=moons_y))
colors = {0: 'orange', 1: 'purple'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
plt.show()

digits_X_train, digits_X_test, digits_y_train, digits_y_test = train_test_split(
    digits_X, digits_y, test_size=0.3, shuffle=False)
print(f"********Load digits**********\nTraining data shape:\n {digits_X_train.shape}, {digits_y_train.shape}\n"
      f"Test data shape: \n {digits_X_test.shape}, {digits_y_test.shape}")

moons_X_train, moons_X_test, moons_y_train, moons_y_test = train_test_split(
    moons_X, moons_y, test_size=0.3, shuffle=False)
print(f"********Make moons**********\nTraining data shape:\n {moons_X_train.shape} {moons_y_train.shape}\n"
      f"Test data shape: \n {moons_X_test.shape} {moons_y_test.shape}")

gaus_digits = GaussianNB()
gaus_digits.fit(digits_X_train, digits_y_train)

multinom_digits = MultinomialNB()
multinom_digits.fit(digits_X_train, digits_y_train)

gaus_moons = GaussianNB()
gaus_moons.fit(moons_X_train, moons_y_train)

x_min, x_max = moons_X[:, 0].min() - 0.5, moons_X[:, 0].max() + 0.5
y_min, y_max = moons_X[:, 1].min() - 0.5, moons_X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                     np.arange(y_min, y_max, 0.2))

df = DataFrame(dict(x=moons_X[:, 0], y=moons_X[:, 1], label=moons_y))
fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    Z = gaus_moons.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.PuOr)
    ax.set_title("GaussianNB Classifier for Moons")
plt.show()


gaus_digits_predicted = gaus_digits.predict(digits_X_test)
print("********Load digits**********\nGaussianNB \n test score:\n",
      gaus_digits.score(digits_X_test, digits_y_test), "\n",
      f" train score: {gaus_digits.score(digits_X_train, digits_y_train)}")

multinom_digits_predicted = multinom_digits.predict(digits_X_test)
print("MultinomialNB \n test score:\n", multinom_digits.score(digits_X_test, digits_y_test), "\n",
      f" train score: {multinom_digits.score(digits_X_train, digits_y_train)}")

gaus_moons_predicted = gaus_moons.predict(moons_X_test)
print("********Make moons**********\nGaussianNB\n test score:\n", gaus_moons.score(moons_X_test, moons_y_test),
      f"\n train score: {gaus_moons.score(moons_X_train, moons_y_train)}")

gaus_digits_predicted_probas = gaus_digits.predict_proba(digits_X_test)
gaus_moons_predicted_probas = gaus_moons.predict_proba(moons_X_test)
multinom_digits_predicted_probas = multinom_digits.predict_proba(digits_X_test)


def model_quality(clf_name, data_name, y_test, predicted, predicted_probas):
    skplt.metrics.plot_confusion_matrix(digits_y_test, multinom_digits_predicted,
                                        title=f'Confusion Matrix for {clf_name} {data_name}')
    plt.show()
    print(f"classiffication report for {clf_name} {data_name}\n--------------------------------\n"
          f"{metrics.classification_report(y_test, predicted)}")
    skplt.metrics.plot_precision_recall(y_test, predicted_probas,
                                        title=f'Precision-Recall curves for {clf_name} {data_name}')
    plt.show()
    skplt.metrics.plot_roc(y_test, predicted_probas,
                           title=f'ROC curves for {clf_name} {data_name}')
    plt.show()


def split_evaluating(clf, X, y, test_sizes):
    for size in test_sizes:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=size, shuffle=False)
        clf.fit(X_train, y_train)
        print(f"train score for test size = {size}\n{clf.score(X_train, y_train)}"
              f"\n---------------------------------------\n")
    return 0


for clf_name, data_name, clf, X, y, \
    test, predicted, probas in [['GaussianNB', 'Digits', gaus_digits, digits_X, digits_y, digits_y_test,
                                 gaus_digits_predicted, gaus_digits_predicted_probas],
                                ['MultinomialNB', 'Digits', multinom_digits, digits_X, digits_y, digits_y_test,
                                 multinom_digits_predicted, multinom_digits_predicted_probas],
                                ['GaussianNB', 'Moons', gaus_moons, moons_X, moons_y, moons_y_test,
                                 gaus_moons_predicted, gaus_moons_predicted_probas]]:
    model_quality(clf_name, data_name, test, predicted, probas)
    print("*********************************************\n"
          " Tuning  Test Size for ", clf_name, "for", data_name, "\n")

    skplt.estimators.plot_learning_curve(clf, X, y, title=f'Learning Curve for {clf_name} {data_name}',
                                         train_sizes=np.linspace(.1, 1, 7))
    plt.show()
    split_evaluating(clf, X, y, [0.2, 0.3, 0.4, 0.5])
