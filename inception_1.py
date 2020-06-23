from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.layers.merge import concatenate
from keras.models import Model

def InceptionModule(x):
    tower_1_0 = Convolution3D(32, (1, 1, 1), activation='relu', padding='same', data_format='channels_first')(x)
    tower_1_1 = Convolution3D(32, (1, 1, 1), activation='relu', padding='same', data_format='channels_first')(x)
    tower_1_1 = Convolution3D(32, (3, 1, 1), activation='relu', padding='same', data_format='channels_first')(tower_1_1)
    tower_1_1 = Convolution3D(32, (1, 3, 1), activation='relu', padding='same', data_format='channels_first')(tower_1_1)
    tower_1_1 = Convolution3D(32, (1, 1, 3), activation='relu', padding='same', data_format='channels_first')(tower_1_1)
    tower_1_2 = Convolution3D(32, (1, 1, 1), activation='relu', padding='same', data_format='channels_first')(x)
    tower_1_2 = Convolution3D(32, (5, 1, 1), activation='relu', padding='same', data_format='channels_first')(tower_1_2)
    tower_1_2 = Convolution3D(32, (1, 5, 1), activation='relu', padding='same', data_format='channels_first')(tower_1_2)
    tower_1_2 = Convolution3D(32, (1, 1, 5), activation='relu', padding='same', data_format='channels_first')(tower_1_2)
    tower_1_3 = MaxPooling3D((3,3,3), strides=(1,1,1), padding='same', data_format='channels_first')(x)
    tower_1_3 = Convolution3D(32, (1, 1, 1), activation='relu', padding='same', data_format='channels_first')(tower_1_3)
    x = concatenate([tower_1_0, tower_1_1, tower_1_2, tower_1_3], axis = 1)
    return x

def conv_pool(x,kernel_size):
    x = Convolution3D(32, (kernel_size,kernel_size,kernel_size), activation='relu', padding='same', data_format='channels_first')(x)
    x = MaxPooling3D((3,3,3),data_format='channels_first')(x)
    return x

input_1 = Input(shape=(1,61,73,61))    
x_1_1 = conv_pool(input_1,3)
x_1_2 = InceptionModule(x_1_1)    

# interpretation
conv_out = Convolution3D(32, (1, 1, 1), activation='relu', padding='same', data_format='channels_first')(x_1_2)
drop3 = Dropout(0.5)(conv_out)
flat = Flatten()(drop3)
outputs = Dense(nb_classes, activation='sigmoid')(flat)
model = Model(inputs=input_1, outputs=outputs, name='inception_1_channel')

# compile
model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
# summarize
print(model.summary())
