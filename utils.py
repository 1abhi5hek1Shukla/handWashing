import cv2

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

import matplotlib.pyplot as plt



# Functions ####################################################
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
def getPixMap(img, nrmFact = 255):
    
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
        
    return np.uint8(nrmFact*skinProb), cv2.merge([bM,gM,rM])


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



def get_median_Frame(path):
    video_stream = cv2.VideoCapture(path)


    # # Randomly select 30 frames
    frameIds = video_stream.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=30)

    # # Store selected frames in an array
    frames = []
    for fid in frameIds:
        video_stream.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = video_stream.read()
        frames.append(frame)
        
    video_stream.release()

    medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

    avgFrame = np.average(frames, axis=0).astype(dtype=np.uint8)

    grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)


    return grayMedianFrame


def get_median_abs_diff(rawFrame, medFrame, blurring = False, threshing = True):
    gframe = cv2.cvtColor(rawFrame, cv2.COLOR_BGR2GRAY)

    dframe = cv2.absdiff(gframe, medFrame)
    
    if blurring:
        dframe = cv2.GaussianBlur(dframe, (21, 21), 0)  

    if threshing:
        ret, dframe = cv2.threshold(dframe,254,255,
                               cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return dframe

def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            # approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            # if area > max_area and len(approx) == 4:
            if area > max_area:
                # biggest = approx
                max_area = area
    return biggest,max_area


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