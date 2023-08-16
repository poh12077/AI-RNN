import numpy as np
import matplotlib.pyplot as plt
import copy

#### what we want to predict
def function(x):
    # return np.power(1.15,x)
    # return x*np.sin(x)
    # return x**3
    return np.log(x)

#### data preprocessing
trainData=50
inputLength=5
predictionSize=30
a=[]
x=[]
x_train=[]
x_test=[]
y_real=[]
y_train=[]
for i in range(1,trainData + predictionSize):
    for j in range(i,i+inputLength):
        a.append( function(j) )
    y_real.append( function(i+inputLength) )
    x.append(i)
    if i<trainData:
        x_train.append(a)
        y_train.append( function(i+inputLength) )
    if i == trainData :
        x_test.append(a)    
    a=[]
y_predict=copy.deepcopy(y_train)
x_train = np.array(x_train, dtype='f')
x_test = np.array(x_test, dtype='f')
y_train = np.array(y_train, dtype='f')
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#2. model build-up
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM
model = Sequential()
# model.add(SimpleRNN(7, input_shape = (inputLength, 1), activation ='relu'))
model.add(LSTM(7, input_shape = (inputLength, 1), activation ='relu'))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(1))
model.summary()

#3. training
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
history = model.fit(x_train, y_train, epochs=70, batch_size=1)

#4. prediction
for i in range(predictionSize):
    prediction = model.predict(x_test)[0][0] 
    y_predict.append(prediction)
    x_test = x_test.flatten()
    for j in range( len(x_test) -1 ):
        x_test[j]= x_test[j+1]
    x_test[ len(x_test)-1 ] = prediction
    x_test = x_test.reshape(1, x_test.shape[0], 1)

    
plt.plot(history.history["loss"])
plt.show()

plt.plot(x, y_real, color = 'red')
plt.plot(x, y_predict, color = 'blue')
plt.show()
