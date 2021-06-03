import os,sys,pdb,natsort
import cv2
import numpy as np
from keras.models import Sequential, load_model



model_path = './models/2021-06-03 14:39:34-model.h5'

model = load_model(model_path)
print("**************************************************")
print("model loaded")

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


videoPath = "./test/handWahimngMyKitchen.mp4"



# For input dimensions
img_rows,img_cols,img_depth=32,32,15

# #########################
cap = cv2.VideoCapture(videoPath)

frames = []

start = True
cv2.namedWindow("Display", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Display", 680,420)

while True:
	test_clips = []
	
	if start:
		for k in range(15):
			ret, frame = cap.read()
			if not ret:
				print("Not Enough Frames")
				sys.exit()
			
			frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
			frameCopy = frame.copy()
			# cv2.imshow("Display", frame)
			frame = cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
			
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			frames.append(gray)

		start = False
	else:
		ret, frame = cap.read()

		if not ret:
			break
		
		frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
		frameCopy = frame.copy()
		# cv2.imshow("Display", frame)
		
		frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)	
		frames.pop(0)
		frames.append(gray)

	inputs=np.array(frames)
	ipt=np.rollaxis(np.rollaxis(inputs,2,0),2,0)
	
	test_clips.append(ipt)
	
	x_test = np.array(test_clips)

	test_set = np.zeros((1, img_rows, img_cols, img_depth, 1))
	
	test_set[0,:,:,:,0]=x_test[0,:,:,:]

	test_set = test_set.astype('float32')
	test_set -= np.mean(test_set)
	test_set /= np.max(test_set)


	y = model.predict(test_set)
	index = np.argmax(y[0])

	action = classFolderMap[index]
	image = cv2.putText(frameCopy, action + "with p = "+ str(y[0][index]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 0, 0), 2, cv2.LINE_AA)
	cv2.imshow("Display", image)
	
	if cv2.waitKey(30) & 0xFF == 27:
		sys.exit()

cap.release()
cv2.destroyAllWindows()