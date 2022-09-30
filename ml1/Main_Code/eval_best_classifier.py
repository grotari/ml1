import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

from util import prep
from util import remove_na_columns

test_size = 0.35
random_state = 42

# Read in data
df = pd.read_csv("data/secondary_data_shuffled.csv", sep=';', header=0)


# Duplikate entfernen
df.drop_duplicates(inplace=True)
df.reset_index(drop=True)

# remove na-columns
df = remove_na_columns(df, threshold=0.5)

# Split
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,1:], df.iloc[:,0],
                                                    test_size=test_size,
                                                    shuffle=True,
                                                    random_state=random_state)

# Transformations Pipeline
preprocessing = prep()

# Trainingsdatensatz fitten und transformieren
X_train_tr = preprocessing.fit_transform(X_train)

# Label encoding für die Zielspalte
le = LabelEncoder()
y_train_tr = le.fit_transform(y_train)

# Testdaten vorbereiten
X_test_tr = preprocessing.transform(X_test)
y_test_tr = le.transform(y_test)

###################################################################
## Ablauf für das evaluieren der jeweils besten Klassifizierer aus
## den individuellen Grid-Searches

# 1. Klassifizierer auf den Trainingsdaten ausführen
# 2. Mit Klassifier auf den Trainingsdaten ein predict und ein score machen
# 3. Berechnen der False-Negative und False-Positive Rate

# scoring Funktion (Wurde zur Wahrung der Übersichtlichkeit hier belassen)
def score_classifier(classifier, name, parameter_string, X_train, y_train, X_test, y_test):
    classifier.fit(X_train, y_train)
    classifier_train_score = classifier.score(X_train, y_train)
    classifier_cross_val_score = cross_val_score(classifier, X_train, y_train, cv=5)
    classifier_test_score = classifier.score(X_test, y_test)
    print("=" * 50)
    print(f"Classifier: {name}")
    print(f"Parameter: {parameter_string}")
    print(f"Train-Score: {classifier_train_score}\nCross-Val-Score: {classifier_cross_val_score} mit Mittelwert: {np.mean(classifier_cross_val_score)}\nTest-Score: {classifier_test_score}")

    # Berechnung der Falsch Positiven und Falsch Negativen mit den tatsächlichen labels, um sicherzugehen
    y_pred = classifier.predict(X_test_tr)
    y_pred_inv = le.inverse_transform(y_pred)
    y_test_inv = le.inverse_transform(y_test_tr)
    count_fn = 0
    count_fp = 0
    for i in np.arange(len(y_pred_inv)):
        if y_pred_inv[i] == 'e' and y_test_inv[i] == 'p':
            # print(f"Index {i} has a false negative")
            count_fn = count_fn + 1
        elif y_pred_inv[i] == 'p' and y_test_inv[i] == 'e':
            count_fp = count_fp + 1
    print(f"Insgesamt falsch vorhergesagte: {count_fn + count_fp} von {len(y_test_inv)} insgesamt klassifizierten ({np.round(((count_fn + count_fp) / len(y_test_inv))* 100, decimals=2)} %)")
    print(f"Falsch negative (essbar vorhergesagt, aber giftig!): {count_fn} von {len(y_test_inv)} insgesamt klassifizierten ({np.round((count_fn / len(y_test_inv))* 100, decimals=2)} %)")
    print(f"Falsch positive (giftig vorhergesagt, aber essbar.): {count_fp} von {len(y_test_inv)} insgesamt klassifizierten ({np.round((count_fp / len(y_test_inv))* 100, decimals=2)} %)")
    return classifier_train_score, classifier_test_score, classifier_cross_val_score


# KNearestNeighbor
knn = KNeighborsClassifier(n_neighbors=2, p=1)
score_classifier(knn, "KNeighborsClassifier", "n_neighbors=2, p=1", X_train_tr, y_train_tr, X_test_tr, y_test_tr)


# Logistische Regression
lr = LogisticRegression(C=0.1, max_iter=500, penalty='l2', fit_intercept=True,
                        class_weight={1: 1.0, 0: 10.0}, solver='lbfgs', tol=1.0)
score_classifier(lr, "LogisticRegression",
                 "C=0.1, max_iter=500, penalty='l2', fit_intercept=True,\n\tclass_weight={1: 1.0, 0: 10.0}, solver='lbfgs', tol=1.0",
                 X_train_tr, y_train_tr, X_test_tr, y_test_tr)

# Support Vector Machine
svc = SVC(C=100.0, gamma='scale', kernel='rbf', tol=0.001, probability=True)
score_classifier(svc, "SVC", "C=100.0, gamma='scale', kernel='rbf', tol=0.001", X_train_tr, y_train_tr, X_test_tr, y_test_tr)

# Gaussian Naive Bayes
gnb = GaussianNB(var_smoothing=0.5)
score_classifier(gnb, "GaussianNB", "var_smooting=0.5", X_train_tr, y_train_tr, X_test_tr, y_test_tr)

# Multi-Layer Perceptron
mlp = MLPClassifier(hidden_layer_sizes=(200, 5), tol=0.00001)
score_classifier(mlp, "MLPClassifier", "hidden_layer_sizes=(200, 5), tol=0.00001", X_train_tr, y_train_tr, X_test_tr, y_test_tr)

# Ensemble Methode aus allen 5 mit hard-voting
voting_clf = VotingClassifier(
    estimators=[('svc', svc), ('logistic_regression', lr), ('MLP', mlp), ('knn', knn),
                ('gaussian_naive_bayes', gnb)], voting='hard')

score_classifier(voting_clf, "Ensemble", "voting='hard'", X_train_tr, y_train_tr, X_test_tr, y_test_tr)

# Ensemble Methode aus allen 5 mit soft-voting
voting_clf_soft = VotingClassifier(
    estimators=[('svc', svc), ('logistic_regression', lr), ('MLP', mlp), ('knn', knn),
                ('gaussian_naive_bayes', gnb)], voting='soft')

score_classifier(voting_clf_soft, "Ensemble", "voting='soft'", X_train_tr, y_train_tr, X_test_tr, y_test_tr)

# Ensemble Methode aus allen 5 mit hard-voting
voting_clf = VotingClassifier(
    estimators=[('svc', svc), ('logistic_regression', lr), ('knn', knn),
                ('gaussian_naive_bayes', gnb)], voting='hard')

score_classifier(voting_clf, "Ensemble  (NoMLP)", "voting='hard'", X_train_tr, y_train_tr, X_test_tr, y_test_tr)

# Ensemble Methode aus allen 5 mit soft-voting
voting_clf_soft = VotingClassifier(
    estimators=[('svc', svc), ('logistic_regression', lr), ('knn', knn),
                ('gaussian_naive_bayes', gnb)], voting='soft')

score_classifier(voting_clf_soft, "Ensemble (NoMLP)", "voting='soft'", X_train_tr, y_train_tr, X_test_tr, y_test_tr)

''' Ausgabe:
==================================================
Classifier: KNeighborsClassifier
Parameter: n_neighbors=2, p=1
Train-Score: 1.0
Cross-Val-Score: [1.         1.         0.99949495 0.99987374 1.        ] mit Mittelwert: 0.9998737373737374
Test-Score: 0.9998124179328456
Insgesamt falsch vorhergesagte: 4 von 21324 insgesamt klassifizierten (0.02 %)
Falsch negative (essbar vorhergesagt, aber giftig!): 0 von 21324 insgesamt klassifizierten (0.0 %)
Falsch positive (giftig vorhergesagt, aber essbar.): 4 von 21324 insgesamt klassifizierten (0.02 %)
==================================================
Classifier: LogisticRegression
Parameter: C=0.1, max_iter=500, penalty='l2', fit_intercept=True,
	class_weight={1: 1.0, 0: 10.0}, solver='lbfgs', tol=1.0
Train-Score: 0.6512790727038562
Cross-Val-Score: [0.65164141 0.65227273 0.64646465 0.64785354 0.65033464] mit Mittelwert: 0.6497133922888437
Test-Score: 0.651003564059276
Insgesamt falsch vorhergesagte: 7442 von 21324 insgesamt klassifizierten (34.9 %)
Falsch negative (essbar vorhergesagt, aber giftig!): 7402 von 21324 insgesamt klassifizierten (34.71 %)
Falsch positive (giftig vorhergesagt, aber essbar.): 40 von 21324 insgesamt klassifizierten (0.19 %)
==================================================
Classifier: SVC
Parameter: C=100.0, gamma='scale', kernel='rbf', tol=0.001
Train-Score: 0.9986110760372736
Cross-Val-Score: [0.99785354 0.99810606 0.99760101 0.99873737 0.99835838] mit Mittelwert: 0.9981312716762268
Test-Score: 0.9982179703620334
Insgesamt falsch vorhergesagte: 38 von 21324 insgesamt klassifizierten (0.18 %)
Falsch negative (essbar vorhergesagt, aber giftig!): 0 von 21324 insgesamt klassifizierten (0.0 %)
Falsch positive (giftig vorhergesagt, aber essbar.): 38 von 21324 insgesamt klassifizierten (0.18 %)
==================================================
Classifier: GaussianNB
Parameter: var_smooting=0.5
Train-Score: 0.729412358897952
Cross-Val-Score: [0.73535354 0.73282828 0.71704545 0.72474747 0.72142947] mit Mittelwert: 0.7262808441786218
Test-Score: 0.7265991371224911
Insgesamt falsch vorhergesagte: 5830 von 21324 insgesamt klassifizierten (27.34 %)
Falsch negative (essbar vorhergesagt, aber giftig!): 4292 von 21324 insgesamt klassifizierten (20.13 %)
Falsch positive (giftig vorhergesagt, aber essbar.): 1538 von 21324 insgesamt klassifizierten (7.22 %)
==================================================
Classifier: MLPClassifier
Parameter: hidden_layer_sizes=(200, 5), tol=0.00001
Train-Score: 1.0
Cross-Val-Score: [0.99936869 0.99974747 1.         0.99936869 0.99974744] mit Mittelwert: 0.999646458268759
Test-Score: 0.9999531044832114
Insgesamt falsch vorhergesagte: 1 von 21324 insgesamt klassifizierten (0.01 %)
Falsch negative (essbar vorhergesagt, aber giftig!): 0 von 21324 insgesamt klassifizierten (0.0 %)
Falsch positive (giftig vorhergesagt, aber essbar.): 1 von 21324 insgesamt klassifizierten (0.01 %)
==================================================
Classifier: Ensemble
Parameter: voting='hard'
Train-Score: 1.0
Cross-Val-Score: [0.99974747 0.99911616 0.99949495 1.         0.99987372] mit Mittelwert: 0.9996464614576117
Test-Score: 0.9999531044832114
Insgesamt falsch vorhergesagte: 1 von 21324 insgesamt klassifizierten (0.01 %)
Falsch negative (essbar vorhergesagt, aber giftig!): 0 von 21324 insgesamt klassifizierten (0.0 %)
Falsch positive (giftig vorhergesagt, aber essbar.): 1 von 21324 insgesamt klassifizierten (0.01 %)
==================================================
Classifier: Ensemble
Parameter: voting='soft'
Train-Score: 1.0
Cross-Val-Score: [1.         1.         0.99949495 1.         1.        ] mit Mittelwert: 0.99989898989899
Test-Score: 0.9999531044832114
Insgesamt falsch vorhergesagte: 1 von 21324 insgesamt klassifizierten (0.01 %)
Falsch negative (essbar vorhergesagt, aber giftig!): 0 von 21324 insgesamt klassifizierten (0.0 %)
Falsch positive (giftig vorhergesagt, aber essbar.): 1 von 21324 insgesamt klassifizierten (0.01 %)
==================================================
Classifier: Ensemble  (NoMLP)
Parameter: voting='hard'
Train-Score: 0.805449632566479
Cross-Val-Score: [0.81313131 0.80808081 0.79482323 0.80214646 0.79656522] mit Mittelwert: 0.8029494082126989
Test-Score: 0.8062277246295254
Insgesamt falsch vorhergesagte: 4132 von 21324 insgesamt klassifizierten (19.38 %)
Falsch negative (essbar vorhergesagt, aber giftig!): 4132 von 21324 insgesamt klassifizierten (19.38 %)
Falsch positive (giftig vorhergesagt, aber essbar.): 0 von 21324 insgesamt klassifizierten (0.0 %)
==================================================
Classifier: Ensemble (NoMLP)
Parameter: voting='soft'
Train-Score: 0.9999747468370414
Cross-Val-Score: [1.         1.         0.99911616 0.99974747 1.        ] mit Mittelwert: 0.9997727272727273
Test-Score: 0.9998124179328456
Insgesamt falsch vorhergesagte: 4 von 21324 insgesamt klassifizierten (0.02 %)
Falsch negative (essbar vorhergesagt, aber giftig!): 3 von 21324 insgesamt klassifizierten (0.01 %)
Falsch positive (giftig vorhergesagt, aber essbar.): 1 von 21324 insgesamt klassifizierten (0.0 %)

'''