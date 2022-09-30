import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

from util import remove_na_columns
from util import prep
from util import do_grid_search


test_size = 0.35
random_state = 42

# Read in data
df = pd.read_csv("data/secondary_data_shuffled.csv", sep=';', header=0)

# Duplikate entfernen
df.drop_duplicates(inplace=True)
df.reset_index(drop=True)

# remove na-columns
df = remove_na_columns(df, threshold=0.5)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,1:], df.iloc[:,0],
                                                    test_size=test_size,
                                                    shuffle=True,
                                                    random_state=random_state)

# Get Transformation pipeline
preprocessing = prep()

# Transformierter Trainingsdatensatz
X_train_tr = preprocessing.fit_transform(X_train)

# Label encoding für die Zielspalte
le = LabelEncoder()
y_train_tr = le.fit_transform(y_train)

# Model Training

# Beispiel:
classifier = GaussianNB()
parameter = {'var_smoothing':[0.0001, 0.01]}

folds = 5
dateiname = "output_test.csv"
do_grid_search(X_train_tr, y_train_tr, classifier, parameter, folds, dateiname)

