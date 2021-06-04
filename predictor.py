import os,sys,pdb,natsort
import cv2
import numpy as np
from keras.models import Sequential, load_model

import warnings
warnings.filterwarnings('ignore')

classFolderMap = {
    0 : 'Step_1',
    1 : 'Step_2_Left',
    2 : 'Step_2_Right',
    3 : 'Step_3',
    4 : 'Step_4_Left',
    5 : 'Step_4_Right',
    6 : 'Step_5_Left',
    7 :'Step_5_Right',
    8 : 'Step_6_Left',
    9 :'Step_6_Right',
    10 : 'Step_7_Left',
    11 : 'Step_7_Right',
}

model_path = './models/2021-06-03 14:39:34-model.h5'

model = load_model(model_path)
print("**************************************************")
print("model loaded")

files = os.listdir('./test/')
nb_files = len(files)
print('number of files:%s'%(nb_files))


files = natsort.natsorted(files)
# cv2.namedWindow("Display", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Display", 680,420)

x_grab = []
for f in files: 
	frames = []
	path = './test/'+f
	cap = cv2.VideoCapture(path)
	fps = cap.get(5)
	img_rows,img_cols,img_depth = 96,64,15

	for k in range(15):
	    ret, frame = cap.read()
	    frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
	    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	    frames.append(gray)
	    if cv2.waitKey(1) & 0xFF == ord('q'):
	        break
	cap.release()
	
	inputs=np.array(frames)
	ipt=np.rollaxis(np.rollaxis(inputs,2,0),2,0)
	x_grab.append(ipt)

x_test = np.array(x_grab)
test_set = np.zeros((nb_files, img_rows, img_cols, img_depth, 1))

for i in range(nb_files):
	test_set[i,:,:,:,0]=x_test[i,:,:,:]

test_set = test_set.astype('float32')
test_set -= np.mean(test_set)
test_set /= np.max(test_set)

results = []#predict results
y = model.predict(test_set)
for j in range(nb_files):
	index = np.argmax(y[j])
	action = classFolderMap[index]
	results.append(action)
print("File names | Predict labels")
for pair in zip(files,results):
	print(pair)