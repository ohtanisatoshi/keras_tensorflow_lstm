# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional

mnist = datasets.fetch_mldata('MNIST original', data_home='.')

n = len(mnist.data)
N = 20000
indices = np.random.permutation(range(n))[:N]
X = mnist.data[indices]
y = mnist.target[indices]
Y = np.eye(10)[y.astype(int)]

# データの正規化
x_max = X.max().astype(np.float32)
print('max: {}'.format(x_max))
# 正規化するために最大値255で割る（0〜1の範囲となる)
X = X / x_max
# 平均を0とする
X = X - X.mean(axis=1).reshape(len(X), 1)

maxlen = 28
T = 28

X = X.reshape(N, maxlen, T)
Y = Y.reshape(N, 10)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, train_size=0.6)


# モデル定義
def weight_variable(shape):
    return np.random.normal(scale=0.1, size=shape)

n_time = len(X[0]) # 28
n_in = len(X[0][0]) # 28
n_hidden = 128
n_out = len(Y[0]) # 10

model = Sequential()
model.add(Bidirectional(GRU(n_hidden), input_shape=(n_time, n_in)))
model.add(Dense(n_out, kernel_initializer=weight_variable))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
              metrics=['accuracy'])

# 学習
epochs = 100
batch_size = 500

hist = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
                 validation_data=(X_validation, Y_validation))

# 評価
loss_and_accuracy = model.evaluate(X_test, Y_test)
print(loss_and_accuracy)


val_acc = hist.history['val_acc']
val_los = hist.history['val_loss']

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
ax1.set_xlabel('epochs')
ax1.set_ylabel('loss')
ax1.yaxis.label.set_color('red')
ax1.tick_params(axis='y', colors='red')
ax1.plot(range(len(val_los)), val_los, label='acc', color='red')
ax2.set_ylabel('acc')
ax2.yaxis.label.set_color('blue')
ax2.tick_params(axis='y', colors='blue')
ax2.plot(range(len(val_acc)), val_acc, label='los', color='blue')
plt.show()


