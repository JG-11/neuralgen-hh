"""
Authors:
    - Dachely Otero @ Tecnologico de Monterrey, 2022.
    - Genaro Almaraz @ Tecnologico de Monterrey, 2022.
    - Benjamin M. Sainz-Tinajero @ Tecnologico de Monterrey, 2022.
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import pandas as pd

import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

import joblib

def train_decision_tree(X, y):
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    df = pd.DataFrame(X)
    df["y"] = y
    df.to_csv("input_decision_tree.csv", index=False)

    tree = DecisionTreeClassifier(criterion='entropy', random_state=42)
    tree.fit(X, y)

    y_predict = tree.predict(X)

    acc_training = metrics.accuracy_score(y, y_predict)

    print("Training accuracy:", acc_training)

    joblib.dump(tree, 'decision_tree_model_joblib.pickle')

    return tree


def train_neural_network(X, y, verbose=0, input_dim=5):
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    df = pd.DataFrame(X)
    df["y"] = y
    df.to_csv("input_neural_network.csv", index=False)

    model = Sequential()
    model.add(Dense(128, kernel_initializer='normal', input_dim=input_dim, activation='relu'))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(512, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='softmax'))

    opt = Adam(learning_rate=0.01)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.fit(X, y, epochs=300, batch_size=64, verbose=verbose)
    
    _, training_acc = model.evaluate(X, y, verbose=verbose)
    
    print("Training accuracy:", training_acc)

    model.save('neuralgen.h5')

    return model