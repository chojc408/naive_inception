import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils  import to_categorical

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Reshape
from tensorflow.keras.layers import concatenate, add, Activation
from tensorflow.keras.models import Model

def get_mnist_dataset(file_name="mnist.npz"):
    mnist = np.load(file_name, allow_pickle=True)
    X_train = mnist['x_train']
    X_test  = mnist['x_test']
    y_train = mnist['y_train']
    y_test  = mnist['y_test']
    # image reshape to (28, 28, 1)
    image_size = X_train.shape[1]
    X_train = np.reshape(X_train, [-1, image_size, image_size, 1])
    X_test  = np.reshape(X_test,  [-1, image_size, image_size, 1])
    # pixel rescaling
    X_train = X_train.astype('float')/255
    X_test  = X_test.astype('float')/255
    return X_train, X_test, y_train, y_test

def integer_to_one_hot(integer, num_classes):
    one_hot_vector = to_categorical(integer, num_classes=num_classes)
    return one_hot_vector

# === Network parameters ======
batch_size    = 64                    # 64 or 128 for CPU
epochs        = 10
element_shape = (28, 28, 1)           # the shape of data point
num_classes   = 10                    # the number of labels
# -----------------------------
n_channels    = element_shape[-1]       # 1 for BW, 3 for RGB
element_dim   = np.prod(element_shape)  # 28 * 28 * 1 = 784
conv_layers   = 2
# -----------------------------
conv_filters  = 32 
kernel_size   = 3
strides       = 1
pool_size     = 2
# -----------------------------

# === Load Dataset & Preprocess
X_train, X_test, y_train_int, y_test_int = get_mnist_dataset(file_name="mnist.npz")
y_train = integer_to_one_hot(y_train_int, num_classes=num_classes)
y_test  = integer_to_one_hot(y_test_int,  num_classes=num_classes)
# --- small subset for testing
train_limit = 2000
test_limit  = 1000
X_train = X_train[:train_limit]
y_train = y_train[:train_limit]
X_test  = X_test[:test_limit]
y_test  = y_test[:test_limit]


x_in = Input(shape=element_shape)

# === Naive Inception Module
def resnet_module(module_in, filters, kernel_size, strides=2):
    conv_out = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding='same',
                      activation='relu')(module_in)
    conv_out = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=1, padding='same',
                      activation=None)(conv_out)
    merge_in = Conv2D(filters=filters, kernel_size=1,
                      strides=strides, padding='same',
                      activation=None)(module_in)
    module_out = add([conv_out, merge_in])
    module_out = Activation('relu')(module_out)
    return module_out

x  = x_in
y  = resnet_module(x, filters=32, kernel_size=3, strides=1)
y  = resnet_module(y, filters=32, kernel_size=3, strides=2)
y  = resnet_module(y, filters=64, kernel_size=3, strides=1)
y  = resnet_module(y, filters=64, kernel_size=3, strides=2)
y  = Flatten()(y)
y  = Dense(10, activation='softmax')(y)

ResNet = Model(x_in, y)
ResNet.summary()

# === Train

ResNet.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
ResNet.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
           validation_split=0.2, verbose=2)

# === Performance Report
loss, acc = ResNet.evaluate(X_test, y_test, verbose=0)
print("-------------------------------------------")
print("Test Loss:", np.round(loss, 4),
      "Test Accuracy:", np.round(acc, 4))
print("-------------------------------------------")
