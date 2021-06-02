from utils import *
import time


handWashingVideo = "./hand_washing_ds/HandWashDataset/HandWashDataset/Step_1/HandWash_001_A_01_G01.mp4"
# handWashingVideo = "./hand_washing_ds/HandWashDataset/HandWashDataset/Step_5_Right/HandWash_001_A_08_G_01.mp4"
# handWashingVideo = "./vidKit.mp4"

# Display Windows

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
# cv2.namedWindow("Diff Frame", cv2.WINDOW_NORMAL)
cv2.namedWindow("Frame_2", cv2.WINDOW_NORMAL)

cv2.resizeWindow('Frame', 640,480)
cv2.resizeWindow('Frame_2', 640,480)
# cv2.resizeWindow('Diff Frame', 640,480)

def calcRoi(singleFrame):
	sh = singleFrame.shape
	cx, cy = sh[1]//2, sh[0]//2 
	# t,b,l,r
	return [cy//2, 3*cy//2, cx//2, 3*cx//2] 


cap = cv2.VideoCapture(handWashingVideo)


# ret, frame = cap.read()
# cap.release()
# frame = cv2.imread("./sampleImage.jpg")


# frame = cv2.imread("./sampleImage.jpg")
# frame = greyWorld(frame)

mFrame = get_median_Frame(handWashingVideo)

kernel = np.ones((3,3),np.uint8)

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = greyWorld(frame)
    
    cv2.imshow("Frame", frame)
    
    f, i = getPixMap(frame)
    
    # f = np.uint8(255*f)
    
    ret, thresh = cv2.threshold(f,t,255, cv2.THRESH_BINARY)

    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)


    nxt = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs,nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    
    cv2.imshow('frame2',bgr)

    cv2.imshow("Frame_2", opening)
    
    k = cv2.waitKey(120) & 0xff
    
    if k ==27:
        break
    prvs = nxt

# cap.release()
cv2.destroyAllWindows()