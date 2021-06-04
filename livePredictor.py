import os,sys,pdb,natsort
import cv2
import numpy as np
from keras.models import load_model



# model_path = './models/2021-06-03 14:39:34-model.h5'
# model_path = './models/2021-06-04 13:48:37-model.h5'
model_path = './models/2021-06-04 00:09:01-model.h5'

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


videoPath = "./test/"
videosName = "handwash-004-a-01-g01_sHx4TMgg_TBKI.mp4"

# For input dimensions
img_rows,img_cols,img_depth = 96,64,15
img_rows,img_cols,img_depth = 75,60,15

print(img_rows,img_cols, img_depth)
# #########################
# cap = cv2.VideoCapture(videoPath+videosName)
cap = cv2.VideoCapture(2)

# writer = cv2.VideoWriter("output.mp4", 
                         # cv2.VideoWriter_fourcc(*"MP4V"), 30,(640,480))


frames = []

start = True
cv2.namedWindow("Display", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Display", 680,420)

while True:
	test_clips = []
	
	if start:
		for k in range(img_depth):
			ret, frame = cap.read()
			if not ret:
				print("Not Enough Frames")
				sys.exit()
			
			frameCopy = frame.copy()
			# cv2.imshow("Display", frame)
			# print(img_rows,img_cols)
			frame = cv2.resize(frame, (img_rows,img_cols), interpolation=cv2.INTER_AREA)

			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			frames.append(gray)

		start = False
	else:
		ret, frame = cap.read()

		if not ret:
			break

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
	x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],x_test.shape[3],1)

	

	x_test = x_test.astype('float32')
	x_test /= 255


	y = model.predict(x_test)
	index = np.argmax(y[0])

	action = classFolderMap[index]
	

	image = cv2.putText(frameCopy, action + "with p = "+ str(y[0][index]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 0, 0), 2, cv2.LINE_AA)
	cv2.imshow("Display", image)
	# writer.write(cv2.resize(image, (640,480)))
	
	if cv2.waitKey(30) & 0xFF == 27:
		break

# writer.release()
cap.release()
cv2.destroyAllWindows()