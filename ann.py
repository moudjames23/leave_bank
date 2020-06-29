import pandas as pd

dataset = pd.read_csv("data_banque.csv")

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, -1].values

# Encodages des données catégoriques
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelEncoderGeography = LabelEncoder()
X[:, 1] = labelEncoderGeography.fit_transform(X[:, 1])

labelEncoderGender = LabelEncoder()
X[:, 2] = labelEncoderGender.fit_transform(X[:, 2])

oneHotEncoder = OneHotEncoder(categorical_features=[1])

X = oneHotEncoder.fit_transform(X).toarray()

# Séparation des données
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Mise en échelle des données
from sklearn.preprocessing import StandardScaler
sc = StandardScaler();
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Configuration des réseaux de neuronnes
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

model = Sequential()

model.add(Dense(units=128,
                kernel_initializer="uniform",
                activation="relu",
                input_dim=12
                ))
model.add(Dropout(0.2))

model.add(Dense(units=64,
                kernel_initializer="uniform",
                activation="relu"))
model.add(Dropout(0.2))


model.add(Dense(units=32,
                kernel_initializer="uniform",
                activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(units=16,
                kernel_initializer="uniform",
                activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(units=1,
                kernel_initializer="uniform",
                activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy",
              metrics=["accuracy"])

model.fit(X_train, y_train, batch_size = 10, epochs=100)

score_train = model.evaluate(X_train, y_train) # 90.6%
score_test = model.evaluate(X_test, y_test) # 85.7%

