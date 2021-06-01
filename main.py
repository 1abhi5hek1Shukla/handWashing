from utils import *
import time




# Display Windows

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.namedWindow("Frame_2", cv2.WINDOW_NORMAL)
cv2.resizeWindow('Frame', 640,480)
cv2.resizeWindow('Frame_2', 640,480)

def calcRoi(singleFrame):
	sh = singleFrame.shape
	cx, cy = sh[1]//2, sh[0]//2 
	# t,b,l,r
	return [cy//2, 3*cy//2, cx//2, 3*cx//2] 


cap = cv2.VideoCapture("./hand_washing_ds/HandWashDataset/HandWashDataset/Step_1/HandWash_012_A_01_G03.mp4")


# ret, frame = cap.read()
# cap.release()
# frame = cv2.imread("./sampleImage.jpg")


# frame = cv2.imread("./sampleImage.jpg")
# frame = greyWorld(frame)


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    
    cv2.imshow("Frame", frame)
    t,b,l,r = calcRoi(frame)
    f, i = getPixMap(frame)
    f = np.uint8(255*f)
    
    ret, thresh = cv2.threshold(f,t,255, cv2.THRESH_BINARY)
    
    segment = np.zeros(thresh.shape, dtype=thresh.dtype)
    segment[t:b,l:r] =thresh[t:b,l:r]
    
    cv2.imshow("Frame_2", segment)
    
    k = cv2.waitKey(120) & 0xff
    
    if k ==27:
        break

# cap.release()
cv2.destroyAllWindows()