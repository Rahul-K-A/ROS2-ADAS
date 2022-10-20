import cv2
import numpy as np
import math
import time
from .utilities import  GetEuclideanDistance
from ...config import config


def GetEuclideanDistance(a,b):
    return math.sqrt( ( (a[1]-b[1])**2 ) + ( (a[0]-b[0])**2 ) )

def RejectSmallerContours(img,MinArea):

    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)[1]
    # Filter using contour area and remove small noise
    cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnts_TooSmall = []
    for index, cnt in enumerate(cnts):
        area = cv2.contourArea(cnt)
        if area < MinArea:
            cnts_TooSmall.append(cnt)
    
    thresh = cv2.drawContours(thresh, cnts_TooSmall, -1, 0, -1) # [ contour = less then minarea contour, contourIDx, Colour , Thickness ]
            
    return thresh

def FindTopAndBottomExtremes(img):
    #Keep only the non zero values of the image
    positions = np.nonzero(img) # position[0] 0 = rows 1 = cols
    
    #If the image is empty, return 0,0
    if (len(positions)!=0):
        #Find top , bottom , left , right most white/non-black points in the image
        top = positions[0].min()
        bottom = positions[0].max()
        left = positions[1].min()
        right = positions[1].max()
        thresh1=cv2.line(cv2.cvtColor(img,cv2.COLOR_GRAY2BGR),(left,top),(right,bottom),(255,0,0),1)
        cv2.imshow("Extremas",thresh1)
        
        return top,bottom
    else:
        return 0,0

def FindLowestRow(img):
    positions = np.nonzero(img) # position[0] 0 = rows 1 = cols
    
    if (len(positions)!=0):
        bottom = positions[0].max()
        return bottom
    else:
        return img.shape[0]

def ReturnLargestContour(gray):
    LargestContour_Found = False
    thresh=np.zeros(gray.shape,dtype=gray.dtype)
    _,bin_img = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
    #Find the two Contours for which you want to find the min distance between them.
    cnts = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    Max_Cntr_area = 0
    Max_Cntr_idx= -1
    for index, cnt in enumerate(cnts):
        area = cv2.contourArea(cnt)
        if area > Max_Cntr_area:
            Max_Cntr_area = area
            Max_Cntr_idx = index
            LargestContour_Found = True
    if (Max_Cntr_idx!=-1):
        thresh = cv2.drawContours(thresh, cnts, Max_Cntr_idx, (255,255,255), -1) # [ contour = less then minarea contour, contourIDx, Colour , Thickness ]
    return thresh, LargestContour_Found

def ReturnLargestContour_OuterLane(gray,minArea):
    IsLargestContour_Found = False
    thresh=np.zeros(gray.shape,dtype=gray.dtype)
    
    #Threshold the image to make sure it's binary
    _,bin_img = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
    
    #Dilate the image and erode it to get the skeleton and remove noise (known as closing operation in image processing)
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(5,5))
    bin_img_dilated = cv2.morphologyEx(bin_img, cv2.MORPH_DILATE, kernel)    #Find the two Contours for which you want to find the min distance between them.
    bin_img_ret = cv2.morphologyEx(bin_img_dilated, cv2.MORPH_ERODE, kernel)    #Find the two Contours for which you want to find the min distance between them.
    bin_img = bin_img_ret
   # cv2.imshow("Closed image",bin_img)
    
    #Iterate through all image contours and find the largest contours
    cnts = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    MaxContourArea  = 0
    MaxContour_Index= -1
    for index, cnt in enumerate(cnts):
        area = cv2.contourArea(cnt)
        if area > MaxContourArea :
            MaxContourArea = area
            MaxContour_Index = index
            IsLargestContour_Found = True
    
    #If the largest contour is still lesser than the minimum required contour val to satisfy the outerlane requirement, then return just the og image and false
    if MaxContourArea < minArea:
        IsLargestContour_Found = False
        return thresh,IsLargestContour_Found
        
        
    #Else draw only the largest contour as a mask and also return true
    if ((MaxContour_Index !=-1) and (IsLargestContour_Found)):    
        Mask_LargestContour = cv2.drawContours(thresh, cnts, MaxContour_Index, (255,255,255), -1) # [ contour = less then minarea contour, contourIDx, Colour , Thickness ]
        return Mask_LargestContour, IsLargestContour_Found

def ROI_extracter(image,strtPnt,endPnt):
    #  Selecting Only ROI from Image
    ROI_mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.rectangle(ROI_mask,strtPnt,endPnt,255,thickness=-1)
    #image_ROI = cv2.bitwise_and(image,image,mask=ROI_mask)
    image_ROI = cv2.bitwise_and(image,ROI_mask)
    return image_ROI

def ExtractPoint(img,specified_row):
    Point= (0,specified_row)
    specified_row_data = img[ specified_row-1,:]
    #print("specified_row_data",specified_row_data)
    positions = np.nonzero(specified_row_data) # position[0] 0 = rows 1 = cols
    #print("positions",positions)    
    #print("len(positions[0])",len(positions[0]))    
    if (len(positions[0])!=0):
        #print(positions[0])
        min_col = positions[0].min()
        Point=(min_col,specified_row)
    return Point

def ReturnLowestEdgePoints(contourImage):
    Outer_Points_list=[]
    thresh = np.zeros_like(contourImage)
    thresh1=thresh.copy()
    thresh2=thresh.copy()
    
    Lane_OneSide=np.zeros_like(contourImage)
    Lane_TwoSide=np.zeros_like(contourImage)

    #Threshold the image to remove noise
    _,bin_img = cv2.threshold(contourImage,0,255,cv2.THRESH_BINARY)
    
    #Find the two Contours for which you want to find the min distance between them.
    cnts = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    thresh = cv2.drawContours(thresh, cnts, 0, (255,255,255), 1) # [ contour = less then minarea contour, contourIDx, Colour , Thickness ]


    cnts2 = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    
    # cv2.imshow("Thresh11",thresh)
    # cv2.imshow("Thresh22",thresh2)
    
    LowRow_a=-1
    LowRow_b=-1
    
    Euc_row=0# Row for the points to be compared

    First_line = np.copy(Lane_OneSide)
    conts_tmp = []

    #Remove noise if present by taking only the largest contour the frame
    if(len(cnts2)>1):
        for index_tmp, cnt_tmp in enumerate(cnts2):
            if((cnt_tmp.shape[0])>50):
                conts_tmp.append(cnt_tmp)
        cnts2 = conts_tmp
        # thresh2 = cv2.drawContours(thresh, cnts2, 0, (255,255,255), 1) # [ contour = less then minarea contour, contourIDx, Colour , Thickness ]
        # cv2.imshow("Thresh22",thresh2)
           
           
     
    for index, cnt in enumerate(cnts2):
        Lane_OneSide = np.zeros(contourImage.shape,dtype=contourImage.dtype)
        Lane_OneSide = cv2.drawContours(Lane_OneSide, cnts2, index, (255,255,255), 1) 
        Lane_TwoSide = cv2.drawContours(Lane_TwoSide, cnts2, index, (255,255,255), 1) # [ contour = less then minarea contour, contourIDx, Colour , Thickness ]

        if(len(cnts2)==2):
            print("!!")
            if (index==0): 
                First_line = np.copy(Lane_OneSide)
                LowRow_a = FindLowestRow(Lane_OneSide)
            elif(index==1):
                LowRow_b = FindLowestRow(Lane_OneSide)
                if(LowRow_a<LowRow_b):# First index is shorter 
                    Euc_row=LowRow_a
                else:
                    Euc_row=LowRow_b
                Point_a = ExtractPoint(First_line,Euc_row)
                Point_b = ExtractPoint(Lane_OneSide,Euc_row)
                Outer_Points_list.append(Point_a)
                Outer_Points_list.append(Point_b)
    
    

    
    return Lane_TwoSide, Outer_Points_list

def ApproxDistanceBetweenCenters(cnt,cnt_cmp):
    # compute the center of the contour
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    # compute the center of the contour
    M_cmp = cv2.moments(cnt_cmp)
    cX_cmp = int(M_cmp["m10"] / M_cmp["m00"])
    cY_cmp = int(M_cmp["m01"] / M_cmp["m00"])
    minDist=GetEuclideanDistance((cX,cY),(cX_cmp,cY_cmp))
    Centroid_a=(cX,cY)
    Centroid_b=(cX_cmp,cY_cmp)
    return minDist,Centroid_a,Centroid_b

