
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import TruncatedNormal
from keras.callbacks import EarlyStopping


# In[2]:


mnist = datasets.fetch_mldata('MNIST original', data_home='.')


# In[3]:


n = len(mnist.data)
N = 20000
indices = np.random.permutation(range(n))[:N]
X = mnist.data[indices]
y = mnist.target[indices]
Y = np.eye(10)[y.astype(int)]

x_max = X.max().astype(np.float32)
print('max: {}'.format(x_max))
# 正規化するために最大値255で割る（0〜1の範囲となる)
X = X / x_max
# 平均を0とする
X = X - X.mean(axis=1).reshape(len(X), 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, train_size=0.6)
print(len(X_train), len(Y_train))
print(len(X_validation), len(Y_validation))
print(len(X_test), len(Y_test))


# In[4]:


def weight_variable(shape):
    return np.random.normal(scale=0.1, size=shape)


# In[5]:


n_in = len(X[0]) # 784
n_hidden = [200, 200, 200]
n_out = len(Y[0]) # 10
dropout_rate = 0.5
model = Sequential()

for i, h in enumerate(n_hidden):
    if i == 0:
        # model.add(Dense(h, input_dim=n_in, kernel_initializer=weight_variable))
        model.add(Dense(h, input_dim=n_in, kernel_initializer='he_normal'))
    else:
        model.add(Dense(h, kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))

model.add(Dense(n_out, kernel_initializer=weight_variable))
model.add(Activation('softmax'))

# model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999), metrics=['accuracy'])


# In[6]:


epochs = 200
batch_size = 100

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
hist = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, 
                 validation_data=(X_validation, Y_validation),
                callbacks=[early_stopping])


# In[7]:


loss_and_accuracy = model.evaluate(X_test, Y_test)
print(loss_and_accuracy)


# In[12]:


val_acc = hist.history['val_acc']
val_los = hist.history['val_loss']

# plt.rc_context({'ytick.color':'red'})
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
# plt.plot(range(epochs), val_acc, label='acc', color='black')
# ax.xaxis.label.set_color('green')
# ax.tick_params(axis='both', color='red')
plt.show()


# In[ ]:




