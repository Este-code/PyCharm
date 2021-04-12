import tensorflow as tf

# class that will stop the training when has hit 95% of accuracy
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.95:
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True


callbacks = myCallback()
mnist = tf.keras.datasets.fashion_mnist

# return training and test sets
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# normalizing images
training_images = training_images / 255.0
test_images = test_images / 255.0

# Flatten will specify that there will be 28 x 28 images
# In the first layer of neurons we're asking for 128 neurons
# and the rectified linear unit(relu) as activation function
# In the second layer of neurons we're asking for 10 neurons,
# where where it will match the input pixels to one of the 10 output values
# that is why we're using softmax as activation function
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# As loss function we're using sparse_categorical_crossentropy
# As optimizer we're using adam(evolution of sgd)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# fitting training images to training labels over 50 epochs
model.fit(training_images, training_labels, epochs=50, callbacks=[callbacks])
