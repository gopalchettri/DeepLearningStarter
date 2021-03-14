from utilities.nn.cnn import LeNet
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import datetime
from tensorflow.keras.datasets import mnist
from keras.utils import np_utils

# Grab the MNIST dataset
print('[INFO]: Accessing MNIST....')
# dataset= mnist.load_data()
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 'channels_first' ordering
if K.image_data_format() == "channels_first":
    # Reshape the design matrix such that the matrix is: num_samples x depth x rows x columns
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
# 'channels_last' ordering
else:
    # Reshape the design matrix such that the matrix is: num_samples x rows x columns x depth
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# Scale the input data to the range [0, 1]
X_train = X_train.astype('float') / 255.0
X_test = X_test.astype('float') / 255.0

# Convert 1-dimensional class arrays to 10-dimensional class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# and perform a train/test split
# (train_x, test_x, train_y, test_y) = train_test_split(data / 255.0, dataset.target.astype("int"), test_size=0.25, random_state=42)

# Convert the labels from integers to vectors
lb = LabelBinarizer()
train_y = lb.fit_transform(Y_train)
test_y = lb.transform(Y_test)

# Initialize the optimizer and model
print("[INFO]: Compiling model....")
optimizer = SGD(lr=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Train the network
print("[INFO]: Training....")
start = datetime.datetime.now()
H = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=128, epochs=20, verbose=1)
stop = datetime.datetime.now()
print("Time taken to execute:" + str(stop - start))

# Evaluate the network
print("[INFO]: Evaluating....")
predictions = model.predict(X_test, batch_size=128)
print(classification_report(Y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(x) for x in lb.classes_]))

# Plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 20), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
