"""
Authors:
    - Dachely Otero @ Tecnologico de Monterrey, 2022.
    - Genaro Almaraz @ Tecnologico de Monterrey, 2022.
    - Benjamin M. Sainz-Tinajero @ Tecnologico de Monterrey, 2022.
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np

def train_neural_network(X, y, verbose=0, input_dim=5):
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    model = Sequential()
    model.add(Dense(128, kernel_initializer='normal', input_dim=input_dim, activation='relu'))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(512, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='softmax'))

    opt = Adam(learning_rate=0.01)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.fit(X, y, epochs=100, batch_size=64, verbose=verbose)
    
    _, training_acc = model.evaluate(X, y, verbose=verbose)
    
    print("Training accuracy:", training_acc)

    model.save('neuralgen.h5')

    return model