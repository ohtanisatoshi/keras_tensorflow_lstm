# coding: utf-8
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional
from keras.layers.core import Dropout, RepeatVector
from keras.layers.wrappers import TimeDistributed

def n(digit=3):
    n_to_return = 0
    for i in range(digit):
        x = np.random.randint(low=0, high=9)
        n_to_return += x * (10**i)

    return n_to_return


def padding(s, max_len):
    return s + ' ' * (max_len-len(s))


N = 5000
digit = 3
input_digit = digit * 2 + 1
output_digit = digit + 1

added = set()
questions = []
answers = []
while True:
    a = n(digit)
    b = n(digit)

    pair = tuple(sorted((a, b)))
    if pair in added:
        continue

    q = padding('{:d}+{:d}'.format(a, b), input_digit)
    a = padding('{:d}'.format(a+b), output_digit)
    questions.append(q)
    answers.append(a)

    added.add(pair)

    sys.stdout.write('\rCreating test data: {:4d} / {:4d}'.format(len(questions), N))
    sys.stdout.flush()
    if len(questions) >= N:
        break

chars = '0123456789+ '
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

X = np.zeros((N, input_digit, len(chars)), dtype=np.integer)
Y = np.zeros((N, output_digit, len(chars)), dtype=np.integer)

for i, q in enumerate(questions):
    for j, c in enumerate(questions[i]):
        X[i, j, char_indices[c]] = 1
    for j, c in enumerate(answers[i]):
        Y[i, j, char_indices[c]] = 1

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, train_size=0.8)

# モデル定義
def weight_variable(shape):
    return np.random.normal(scale=0.1, size=shape)

n_in_time = input_digit
n_in = len(chars) # 12
n_hidden = 128
n_out_time = output_digit
n_out = len(chars) # 12

model = Sequential()
# encoder
model.add(GRU(n_hidden, input_shape=(n_in_time, n_in)))
model.add(Dropout(0.5))
# decoder
model.add(RepeatVector(output_digit))
model.add(GRU(n_hidden, return_sequences=True))
model.add(Dropout(0.5))

model.add(TimeDistributed(Dense(n_out)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
              metrics=['accuracy'])

# 学習
epochs = 1000
batch_size = 200

hist = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
                 validation_data=(X_validation, Y_validation))

# 評価
loss_and_accuracy = model.evaluate(X_test, Y_test)
print(loss_and_accuracy)

# テスト結果表示
q_cnt = 50
ok_cnt = 0
for i in range(50):
    x_ = X_test[i:i+1]
    y_ = Y_test[i:i+1]
    p_ = model.predict(x_)
    question = ''
    for one_hot in x_[0]:
        question += [indices_char[index] for index, flag in enumerate(one_hot) if flag==1][0]

    answer = ''
    for one_hot in y_[0]:
        answer += [indices_char[index] for index, flag in enumerate(one_hot) if flag==1][0]

    predict = ''
    for one_hot in p_[0]:
        index = np.argmax(one_hot)
        predict += indices_char[index]

    if int(answer) == int(predict):
        result = '○'
        ok_cnt += 1
    else:
        result = '×'

    print('{:7s} = {:4s} : {:4s}  {}'.format(question, answer, predict, result))
print('result: {} / {} {:.2f}%'.format(ok_cnt, q_cnt, ((ok_cnt/q_cnt)*100)))

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


