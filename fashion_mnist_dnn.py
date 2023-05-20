import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

tf.random.set_seed(42)

fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
# 55_000 samples for training
#  5_000 samples for validation
# 10_000 samples for test set
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist
X_train, y_train = X_train_full[:-5000], y_train_full[:-5000]
X_valid, y_valid = X_train_full[-5000:], y_train_full[-5000:]

# scale the pixel intensity to the 0-1 range
X_train, X_valid, X_test = X_train / 255. , X_valid / 255., X_test / 255.

class_names = np.array(["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"])

# Create the DNN
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=[28, 28]))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(300, activation='relu'))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.summary()

# Compile the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))

# Plot the learning curve
pd.DataFrame(history.history).plot(
    figsize=(8, 5), xlim=[0, 29], ylim=[0, 1], grid=True, 
    xlabel='EPoch', ylabel='Loss', 
    style=['r--', 'r--.', 'b-', 'b-*']
)
plt.savefig('fashion_mnist_learning_curve.png')
plt.show()

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("loss on test set: %.2f" % loss)
print("accuracy on test set: %.2f" % accuracy)

# Use the model to make predictions
X_new = X_test[:3]
print("*********")
print(X_new.shape)
print("*********")
y_proba = model.predict(X_new)
y_pred = y_proba.argmax(axis=-1)

y_test_classnames = class_names[y_test[:3]]
y_pred_classnames = class_names[y_pred]
for y, y_hat in zip(y_test_classnames, y_pred_classnames):
     print("True class = %s, Prediction = %s" % (y, y_hat))

# save model for later use
model.save('my_fashion_mnist_model', save_format='tf')
