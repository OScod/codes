import pandas as pd
import numpy as np
df = pd.read_csv(r"C:\Users\xyz\OneDrive\Desktop\Python Datasets\pima-indiansdiabetes.csv", delimiter=',')
df.head()
x= df.iloc[:,:8]
y= df.iloc[:,8]
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
#hidden layers
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
#output layer
model.add(Dense(1, activation='sigmoid'))
#compile model
model.compile(loss='binary_crossentropy', optimizer='adam',
 metrics=['accuracy'])
#train model
model.fit(x, y, epochs = 100, batch_size=10)
#<keras.callbacks.History at 0x1f871d57580>
#evaluate
model.evaluate(x,y)
model.summary()
Model: "sequential"
