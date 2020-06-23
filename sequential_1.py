#Model written in Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.models import Sequential

model = Sequential()
model.add(Convolution3D(32, (3,3,3), input_shape=(1,61,73,61),activation='relu', padding='same', data_format='channels_first'))
model.add(MaxPooling3D((3,3,3),data_format='channels_first'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='sigmoid'))

# compile
model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
# summarize
print(model.summary())


#Model written with Input and Output
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.models import Model

def conv_pool(x,kernel_size):
    x = Convolution3D(32, (kernel_size,kernel_size,kernel_size), activation='relu', padding='same', data_format='channels_first')(x)
    x = MaxPooling3D((3,3,3),data_format='channels_first')(x)
    return x

input_1 = Input(shape=(1,61,73,61))    
x_1_1=conv_pool(input_1,3)
flat1=Flatten()(x_1_1)
dense1=Dense(128, activation='relu')(flat1)
drop3=Dropout(0.5)(dense1)
outputs = Dense(nb_classes, activation='sigmoid')(drop3)

model = Model(inputs=input_1, outputs=outputs,name='conv_pool_1_channel')

# compile
model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
# summarize
print(model.summary())
