import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

l0 = Dense(units=1, input_shape=[1])  # One dense layer with, in this case, one input (xs)
model = Sequential([l0])  # Define layer, one line = one layer
model.compile(optimizer='sgd', loss='mean_squared_error')
# Specify the optimizer, in this case it will use
# Stochastic gradient descent(sgd) function

# Declaring the arrays
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Fit the Xs to the Ys, and try it 500 times
model.fit(xs, ys, epochs=500)

# Asking to predict Y when X = 10
print(model.predict([10.0]))
# Print out the values(weights), in this case Y =(first_value)X - (second_value)
# Since the model has a single neuron, it learn weight and bias, so that Y = WX + B
print("Here is what I learned: {}".format(l0.get_weights()))
