# from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard

# import tensorflow as tf
# import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn.model_selection import train_test_split

from dataProcessing import read_data
from dataProcessing import preProcessing

#########################################################33
img_rows,img_cols,img_depth = 96,64,15

print("Reading Data")
print("########################################")

x_train, y_train = read_data()

print("Data Reading Done")
print("###################################3####")
numOfSamples = len(x_train)

print(x_train.shape)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3],1)

print(x_train.shape)

x_train = x_train.astype('float32')
x_train /= 255


dataSplitRatio = 0.2
validationtRatio = 0.2


print("Splitting Data")
x_train, x_test, y_train, y_test = train_test_split(x_train,
										y_train, test_size=dataSplitRatio)

x_train, x_valid, y_train, y_valid = train_test_split(x_train,
										y_train, test_size=validationtRatio)


batch_size = 5
nb_classes = 12
nb_epoch = 10

y_train = to_categorical(y_train,nb_classes)
y_test = to_categorical(y_test,nb_classes)
y_valid = to_categorical(y_valid,nb_classes)

RMSprob = RMSprop(learning_rate=0.001, rho=0.9, epsilon=None, decay=0.0)

def modelCNN():
	nb_filters = [32,64]
	patch_size = 15
	model = Sequential()

	# print('input shape', img_rows, 'rows', img_cols, 'cols', patch_size, 'patchsize')

	model.add(Conv3D(nb_filters[0],(3,3,3),input_shape=(img_rows, img_cols,patch_size,1),activation='relu'))

	# model.add(BatchNormalization())

	model.add(MaxPooling3D(pool_size=(2, 2, 2)))

	model.add(BatchNormalization())

	model.add(Conv3D(nb_filters[1],(3,3,3),activation='relu'))

	# model.add(BatchNormalization(momentum=0.99))

	model.add(MaxPooling3D(pool_size=(3, 3, 3)))

	model.add(BatchNormalization())

	model.add(Dropout(0.5))

	model.add(Flatten())

	model.add(Dense(256))

	model.add(Dropout(0.5))

	model.add(Dense(nb_classes))

	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['mse', 'accuracy'])

	return model



model = modelCNN()
model.summary()
# 
hist = model.fit(x_train,
			    y_train,
			    validation_data=(x_valid,y_valid),
			    batch_size=batch_size,
			    epochs = nb_epoch,
			    shuffle=True,
			    callbacks=[TensorBoard(log_dir='./log')] )#tensorboard --logdir=./log

now = str(datetime.datetime.now()).split('.')[0]
model.save('./models/'+now+"-model.h5")

score = model.evaluate(x_test,y_test,batch_size=batch_size)
# Print the results

print('**********************************************')
print('TEST SCORE : ',score[0])
print('ACCURACY : ',score[1])

plt.figure(1)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title("Loss")
plt.xlabel("Epoch")

plt.figure(2)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title("Accuracy")
plt.xlabel("Epoch")

plt.show()
