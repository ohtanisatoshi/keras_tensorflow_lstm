# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers.recurrent import SimpleRNN
from keras.layers.recurrent import LSTM


def sin(x, t=100):
    return np.sin(2.0 * np.pi * x / t)


def toy_problem(t=100, ampl=0.05):
    x = np.arange(0, 2 * t + 1)
    noise = ampl * np.random.uniform(low=-1.0, high=1.0, size=len(x))
    return sin(x) + noise

T = 100
f = toy_problem(T)

length_of_sequences = 2 * T
maxlen = 25

data = []
target = []
for i in range(0, length_of_sequences - maxlen + 1):
    data.append(f[i: i + maxlen])
    target.append(f[i + maxlen])

X = np.array(data).reshape(len(data), maxlen, 1)
Y = np.array(target).reshape(len(data), 1)

N_train = int(len(data) * 0.9)
N_validation = len(data) - N_train

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=N_validation)


def weight_variable(shape):
    return np.random.normal(scale=0.1, size=shape)

n_in = len(X[0][0])
n_hidden = 20
n_out = len(Y[0])

model = Sequential()
# maxlen=25
# 25の過去の連続データ
# n_in=1
# データの次元数、ここでは1次元のデータを扱う
#model.add(SimpleRNN(n_hidden, kernel_initializer=weight_variable, input_shape=(maxlen, n_in)))
model.add(LSTM(n_hidden, kernel_initializer=weight_variable, input_shape=(maxlen, n_in)))
# 出力層、計算結果をそのまま出力する
model.add(Dense(n_out, kernel_initializer=weight_variable))
model.add(Activation('linear'))

# 評価は２乗誤差で行う
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(loss='mean_squared_error', optimizer=optimizer)

epochs = 500
batch_size = 10

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
          validation_data=(X_validation, Y_validation),
          callbacks=[early_stopping])

val_loss = hist.history['val_loss']
plt.plot(val_loss)
plt.show()

truncate = maxlen
Z = X[:1]
original = [f[i] for i in range(maxlen)]
predicted = [None for i in range(maxlen)]

for i in range(length_of_sequences - maxlen + 1):
    z_ = Z[-1:]
    y_ = model.predict(z_)
    sequence_ = np.concatenate((z_.reshape(maxlen, n_in)[1:], y_), axis=0)
    sequence_ = sequence_.reshape(1, maxlen, n_in)
    Z = np.append(Z, sequence_, axis=0)
    predicted.append(y_.reshape(-1))


plt.rc('font', family='serif')
plt.figure()
plt.plot(toy_problem(T, ampl=0), linestyle='dotted', color='#aaaaaa')  # ノイズなしのsin波
plt.plot(original, linestyle='dashed', color='black')
plt.plot(predicted, color='black')
plt.show()
