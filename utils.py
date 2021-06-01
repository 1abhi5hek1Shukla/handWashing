import cv2
import numpy as np
import matplotlib.pyplot as plt

def displayPlt(image, cmap=None):
    if cmap:
        plt.imshow(image, cmap)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        
        
def runVideo(p):
    cap = cv2.VideoCapture(p)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Frame", frame)
        k = cv2.waitKey(30) & 0xff
        if k ==27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    
def greyWorld(image):
	b,g,r = cv2.split(image)
	
	r_avg = np.average(r)
	b_avg = np.average(b)
	g_avg = np.average(g)

	avgGray = (r_avg+ b_avg + g_avg) /3

	kR = avgGray/r_avg
	kB = avgGray/b_avg
	kG = avgGray/g_avg

	scaledR = kR * r
	scaledB = kB * b
	scaledG = kG * g

	result = np.uint8(cv2.merge([scaledB, scaledG, scaledR]))

	return result



rMean, bMean, gMean =  0.33, 0.41, 0.39
varinaR, varinaG, varinaB = 0.15,0.14,0.13
t = 171

# 0.33 0.41 0.39 0.15 0.14 0.13
# t = 171
def getPixMap(img):
    
    varR = varinaR
    varG = varinaG
    varB = varinaB
    
    global bMean, rMean, gMean
#     bMean, gMean, rMean 
    b,g,r = cv2.split(img)
    
    summed = img.sum(axis=2)
    # print(rMean,bMean,gMean, varinaR, varinaG, varinaB,)
    bM = np.divide(b, summed,dtype='float32')
    gM = np.divide(g, summed,dtype='float32')
    rM = np.divide(r, summed,dtype='float32')
    
    skinProb = 1/(2*np.pi * varR**0.5 * varG**0.5) * np.exp(
        -(
            (rM - rMean)**2 / (2*varR) + 
            (gM - gMean)**2 / (2*varG) +
            (bM - bMean)**2 / (2*varB)
        )
    )
        
    return skinProb, cv2.merge([bM,gM,rM])


def on_change_red(value):
    global rMean
    rMean = value/100
def on_change_green(value):
    global gMean
    gMean = value/100
def on_change_blue(value):
    global bMean
    bMean = value/100
    
    
def on_change_r_var(value):
    global varinaR
    varinaR = value/100
def on_change_g_var(value):
    global varinaG
    varinaG = value/100
def on_change_b_var(value):
    global varinaB
    varinaB = value/100
    
    
def on_change_thresh(value):
    global t
    t = value






# RElated to Vidoe Runn
# cv2.createTrackbar('rGreen', "Frame", int(100*gMean), 100, on_change_rGreen)
# def on_change_rGreen(value):
#     global gMean
#     gMean = value/100

# cap = cv2.VideoCapture(path)

# detector = hD()
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
    
    
#     cv2.imshow("Frame", frame)
    
# #     f, i = getPixMap(frame)
# #     f = np.uint8(255*f)
# #     ret, thresh = cv2.threshold(f,1,255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
# #     cv2.imshow("Frame 2", thresh)
    
#     frame = greyWorld(frame)
#     hands = detector.findHands(frame)
#     cv2.imshow("Hands", hands)
#     k = cv2.waitKey(30) & 0xff
#     if k ==27:
#         break

# cap.release()
# cv2.destroyAllWindows()