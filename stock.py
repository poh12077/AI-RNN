import numpy as np
import matplotlib.pyplot as plt
import csv

file = open('samsung_1_year.csv','r')
csvfile = csv.reader(file)

list=[]
stockPrice=[]
for line in csvfile:
    list.append(line[1])
for i in range(0, len(list) ):
    stockPrice.append( float(list[ len(list) - i - 1 ]) )

#### data preprocessing
trainDataSize= len(stockPrice)-20
inputLength=5
a=[]
x=[]
x_train=[]
x_test=[]
y_real=[]
y_train=[]
for i in range(0, len(stockPrice) - inputLength ):
    for j in range(i, i+inputLength):
        a.append( stockPrice[j] )
    x_test.append(a)
    y_real.append( stockPrice[i+inputLength] )
    x.append(i)
    if i<trainDataSize:
        x_train.append(a)
        y_train.append( stockPrice[i+inputLength] )
    a=[]
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_real = np.array(y_real)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#2. model build-up
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM
model = Sequential()
# model.add(SimpleRNN(7, input_shape = (inputLength, 1), activation ='relu'))
model.add(LSTM(7, input_shape = (inputLength, 1), activation='relu'))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(1))
model.summary()

#3. training
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
history = model.fit(x_train, y_train, epochs=30, batch_size=1)

#4. prediction
y_predict = model.predict(x_test)

plt.plot(history.history["loss"])
plt.show()

# plt.plot(x, y_real, y_predict)
plt.plot(y_real, color = 'red')
plt.plot(y_predict, color='blue')
plt.show()