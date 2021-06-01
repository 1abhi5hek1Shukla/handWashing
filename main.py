from utils import *
import time


handWashingVideo = "./hand_washing_ds/HandWashDataset/HandWashDataset/Step_1/HandWash_001_A_01_G01.mp4"

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

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # diffFrame = get_median_abs_diff(frame, mFrame)
    
    frame = greyWorld(frame)
    
    cv2.imshow("Frame", frame)
    # cv2.imshow("Diff Frame", diffFrame)
    # t,b,l,r = calcRoi(frame)
    
    f, i = getPixMap(frame)
    
    f = np.uint8(255*f)
    
    ret, thresh = cv2.threshold(f,t,255, cv2.THRESH_BINARY)

    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # segment = np.zeros(thresh.shape, dtype=thresh.dtype)
    # segment[t:b,l:r] =thresh[t:b,l:r]

    cv2.imshow("Frame_2", opening)
    
    k = cv2.waitKey(120) & 0xff
    
    if k ==27:
        break

# cap.release()
cv2.destroyAllWindows()