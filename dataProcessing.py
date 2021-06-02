import os
import numpy as np
import cv2

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
