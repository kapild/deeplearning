import numpy as np

from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.utils.visualize_util import plot

training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")
target_data = np.array([[0],[1],[1],[0]], "float32")

model = Sequential()
model.add(Dense(32, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])

model.fit(training_data, target_data, nb_epoch=1000, verbose=2)

print np.round(model.predict(training_data))
plot(model, to_file='model.png', show_shapes=True)

