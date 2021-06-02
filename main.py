from utils import *
import time


handWashingVideo = "./hand_washing_ds/HandWashDataset/HandWashDataset/Step_1/HandWash_001_A_01_G01.mp4"
# handWashingVideo = "./hand_washing_ds/HandWashDataset/HandWashDataset/Step_5_Right/HandWash_001_A_08_G_01.mp4"
# handWashingVideo = "./vidKit.mp4"



# Display Windows
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.namedWindow("hand_seg", cv2.WINDOW_NORMAL)
cv2.namedWindow("optical_flow_frame", cv2.WINDOW_NORMAL)

cv2.resizeWindow('Frame', 640,480)
cv2.resizeWindow('hand_seg', 640,480)
cv2.resizeWindow("optical_flow_frame", 640,480)


# CAPTURE STARTED #########################################################
cap = cv2.VideoCapture(handWashingVideo)

# Kernal for opeing and clsing
kernel = np.ones((3,3),np.uint8)


# First frame for calc optical flow
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)


# Window Display loop ###################################################################

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame, segHand = get_hand_segmented(frame,kernel = kernel,threshing = True)
    
    cv2.imshow("Frame", frame)

    nxt = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs,nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])



    cv2.imshow('optical_flow_frame',mag)
    cv2.imshow("hand_seg", segHand)
    
    k = cv2.waitKey(30) & 0xff
    
    if k ==27:
        break
    
    prvs = nxt


# Window Display loop / ##################################################################

# CAPTURE CLOSED #########################################################
cap.release()
cv2.destroyAllWindows()