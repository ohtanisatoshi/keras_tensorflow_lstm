import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers.recurrent import SimpleRNN
from keras.layers.recurrent import LSTM, GRU


def mask(t=200):
    mask = np.zeros(t)
    indices = np.random.permutation(np.arange(t))[:2]
    mask[indices] = 1

    return mask

def toy_problem(n=10, t=200):
    signals = np.random.uniform(low=0.0, high=1.0, size=(n, t))
    masks = np.zeros((n,t))
    for i in range(n):
        masks[i] = mask(t)

    data = np.zeros((n, t, 2))
    data[:, :, 0] = signals[:]
    data[:, :, 1] = masks[:]
    target = (signals * masks).sum(axis=1).reshape(n, 1)

    return (data, target)

N = 10000
T = 200
maxlen = T

X, Y = toy_problem(N, T)

N_train = int(N * 0.9)
N_validation = N - N_train

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=N_validation)


def weight_variable(shape):
    return np.random.normal(scale=0.1, size=shape)

n_in = len(X[0][0])
n_hidden = 20
n_out = len(Y[0])

model = Sequential()
model.add(GRU(n_hidden, kernel_initializer=weight_variable, input_shape=(maxlen, n_in)))
model.add(Dense(n_out, kernel_initializer=weight_variable))
model.add(Activation('linear'))

# 評価は２乗誤差で行う
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(loss='mean_squared_error', optimizer=optimizer)

epochs = 500
batch_size = 500

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
          validation_data=(X_validation, Y_validation))

val_loss = hist.history['val_loss']
plt.plot(val_loss)
plt.show()
