import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils  import to_categorical

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
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

# === Model Build
def get_model():
    x_in = Input(shape=element_shape) # shape=(28,28,1)
    x = x_in
    for layer in range(conv_layers):
        x = Conv2D(filters=conv_filters, kernel_size=kernel_size,
                   strides=strides, padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size)(x)
    x = Flatten()(x)
    y_out = Dense(num_classes, activation='softmax')(x)  # multi-class classification
    model = Model(x_in, y_out)
    model.summary()
    return model

# === Compile and Train

CNN = get_model()
CNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
CNN.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
        validation_split=0.2, verbose=2)

# === Performance Report
loss, acc = CNN.evaluate(X_test, y_test, verbose=0)
print("Loss:", np.round(loss, 4), "Accuracy:", np.round(acc, 4))
y_pred = CNN.predict(X_test)
# retruns probailities of classes
# for small test dataset, use y_pred = model(x) for faster calculation
y_pred_int = np.argmax(y_pred, axis=1)
# retruns predicted class (integer)









conv_output_shape = K.int_shape(x_enc)
# (None,7, 7, 64): Decoder needs this info
# --- latent space
x_enc = Flatten()(x_enc)
flat_output_shape = K.int_shape(x_enc)
# (None, 3136) = (None, 7*7*64): Decoder needs this info
z = Dense(latent_dim, activation=None)(x_enc)
# --- build model
Encoder = Model(x_in, z, name='encoder')
Encoder.summary()

# === Decoder part ============
z_in  = Input(shape=latent_dim)
x_dec = z_in
x_dec = Dense(flat_output_shape[1], activation=None)(x_dec) # (7*7*64=3136)
x_dec = Reshape((conv_output_shape[1:]))(x_dec)
x_dec = Conv2DTranspose(filters=conv_filters[1], kernel_size=kernel_size,
                        strides=strides,padding='same',
                        activation='relu')(x_dec)
x_dec = Conv2DTranspose(filters=conv_filters[0], kernel_size=kernel_size,
                        strides=strides, padding='same',
                        activation='relu')(x_dec)
# --- to original shape-------
x_out = Conv2DTranspose(filters=n_channels, kernel_size=kernel_size,
                        strides=1, padding='same',
                        activation='sigmoid')(x_dec)
# --- build model
Decoder = Model(z_in, x_out, name='decoder')
Decoder.summary()

# === Autoencoder ============
z     = Encoder(x_in)
x_out = Decoder(z)
AE    = Model(x_in, x_out, name='autoencoder')
AE.summary()
AE.compile(loss='binary_crossentropy', optimizer='adam')
#AE.compile(loss='mse', optimizer='adam')
 
X_train, X_test, y_train, y_test = get_mnist_dataset(file_name="mnist.npz") 
X_train = X_train[:5]

AE.fit(X_train, X_train, epochs=epochs, batch_size=batch_size,
       shuffle=True, validation_split=0.0)

# === Reconstruction Example
for i in range(5):
    x_ori = X_train[i:i+1]
    x_rec = AE.predict(x_ori)
    view(x_ori, x_rec)
