import numpy as np
import matplotlib.pyplot as plt
import csv
import copy

#### hyper parameter
epoch=100
inputLength=5
numOfPrediction=5

#### read data
file = open('samsung_6_month.csv','r')
csvfile = csv.reader(file)

list=[]
stockPrice=[]
for line in csvfile:
    list.append(line[1])
for i in range(0, len(list) ):
    stockPrice.append( float(list[ len(list) - i - 1 ]) )

#### data preprocessing
trainDataSize= len(stockPrice)-numOfPrediction-inputLength
a=[]
x=[]
x_train=[]
x_test=[]
y_real=[]
y_train=[]
for i in range(0, len(stockPrice)-inputLength ):
    for j in range(i, i+inputLength):
        a.append( stockPrice[j] )
    y_real.append( stockPrice[i+inputLength] )
    x.append(i)
    if i<trainDataSize:
        x_train.append(a)
        y_train.append( stockPrice[i+inputLength] )
    if i == trainDataSize :
        x_test.append(a) 
    a=[]
y_predict=copy.deepcopy(y_train)
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#2. model build-up
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM
model = Sequential()
model.add(LSTM(7, input_shape = (inputLength, 1), activation='relu'))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(1))
model.summary()

#3. training
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
history = model.fit(x_train, y_train, epochs=epoch, batch_size=1)

#### prediction
for i in range(numOfPrediction):
    prediction = model.predict(x_test)[0][0] 
    y_predict.append(prediction)
    x_test = x_test.flatten()
    for j in range( len(x_test) -1 ):
        x_test[j]= x_test[j+1]
    x_test[ len(x_test)-1 ] = prediction
    x_test = x_test.reshape(1, x_test.shape[0], 1)
    
plt.plot(history.history["loss"])
plt.show()

plt.plot(y_real, color = 'red')
plt.plot(y_predict, color='blue')
plt.show()
