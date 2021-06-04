import os
import numpy as np
import cv2


def getImagesFromVideos():
	dsPath = "./hand_washing_ds/HandWashDataset/HandWashDataset/"
	osPath = "./collections"
	os.mkdir(osPath)


	allFiles = os.listdir(dsPath)

	folders = []
	for f in allFiles:
		if "Step" in f:
			folders.append(f)


	for f in folders:
		count = 1
		# Folder "step"
		os.mkdir(osPath + "/" + f)
		vids = os.listdir(dsPath + f)

		for v in vids:
			video_stream = cv2.VideoCapture(dsPath+f+"/"+v)
			frameIds = video_stream.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=5)
			
			for fid in frameIds:
				video_stream.set(cv2.CAP_PROP_POS_FRAMES, fid)
				ret, frame = video_stream.read()
				cv2.imwrite(osPath+"/"+f+"/"+str(count)+".jpg",frame)
				count += 1

			video_stream.release()


		print(f+ " completed")

	print("All steps completed")


def read_data():
	dsPath = "./hand_washing_ds/HandWashDataset/"
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
	x_set = []

	y_labels = []

	img_rows,img_cols,img_depth = 75,60,15

	for key, val in classFolderMap.items():
	    videosInFolder = os.listdir(dsPath+val)
	    for vid in videosInFolder:
	        cap = cv2.VideoCapture(dsPath + val + "/" + vid)
	        enoughFrames = True

	        for bundles in range(4):

	            frames = []
	            for k in range(img_depth):
	                ret, frame = cap.read()
	                if not ret:
	#                     print("Not enough frames in ", vid)
	                    enoughFrames = False
	                    break
	                frame = cv2.resize(frame,
	                     (img_rows, img_cols), interpolation= cv2.INTER_AREA)
	                grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	                frames.append(grayFrame)

	            if not enoughFrames:
	                break
	            inputs = np.array(frames)
	            ipt = np.rollaxis(np.rollaxis(inputs,2,0),2,0)
	            x_set.append(ipt)
	            y_labels.append(key)

	        cap.release()
	    print(len(y_labels))
	    print(val + ", done")
	return np.array(x_set), np.array(y_labels)

def preProcessing(img):
	img = cv2.equalizeHist(img)
	img = img / 255
	return img
