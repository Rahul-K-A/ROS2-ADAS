import cv2
import numpy as np

# H,S,L are scaled in [0,255]
hls = 0
src = 0

#White Regions Range 
hue_l = 0
lit_l = 225
sat_l = 0

hue_l_y = 30
hue_h_y = 33
lit_l_y = 160
sat_l_y = 0

def maskextract():
    global frame_hls,src
    mask   = segmentImage_Color(frame_hls,(hue_l  ,lit_l   ,sat_l  ),(255,255,255))
    mask_y = segmentImage_Color(frame_hls,(hue_l_y,lit_l_y ,sat_l_y),(hue_h_y,255,255))

    mask_ = mask != 0
    dst = src * (mask_[:,:,None].astype(src.dtype))

    mask_y_ = mask_y != 0
    dst_Y = src * (mask_y_[:,:,None].astype(src.dtype))

    cv2.imshow('white_regions',dst)
    cv2.imshow('yellow_regions',dst_Y)
    dst=dst+dst_Y
    cv2.imshow('combined',dst)
    
    
def on_hue_low_change(val):
    global hue_l
    hue_l = val
    maskextract()
def on_lit_low_change(val):
    global lit_l
    lit_l = val
    maskextract()
def on_sat_low_change(val):
    global sat_l
    sat_l = val
    maskextract()
    
def on_hue_low_y_change(val):
    global hue_l_y
    hue_l_y = val
    maskextract()
def on_hue_high_y_change(val):
    global hue_h_y
    hue_h_y = val
    maskextract()
def on_lit_low_y_change(val):
    global lit_l_y
    lit_l_y = val
    maskextract()
def on_sat_low_y_change(val):
    global sat_l_y
    sat_l_y = val
    maskextract()

cv2.namedWindow("white_regions")
cv2.namedWindow("yellow_regions")

cv2.createTrackbar("Hue_L","white_regions",hue_l,255,on_hue_low_change)
cv2.createTrackbar("Lit_L","white_regions",lit_l,255,on_lit_low_change)
cv2.createTrackbar("Sat_L","white_regions",sat_l,255,on_sat_low_change)

cv2.createTrackbar("Hue_L_Y","yellow_regions",hue_l_y,255,on_hue_low_y_change)
cv2.createTrackbar("Hue_H_Y","yellow_regions",hue_h_y,255,on_hue_high_y_change)
cv2.createTrackbar("Lit_L_Y","yellow_regions",lit_l_y,255,on_lit_low_y_change)
cv2.createTrackbar("Sat_L_Y","yellow_regions",sat_l_y,255,on_sat_low_y_change)


def segmentImage_Color(hls_img,hls_lower,hls_higher):
     print("Rype lo",type(hls_lower))
     print("Rype hi",type(hls_higher))
    
     #Get mask by color segmentation
     mask=cv2.inRange(hls_img,hls_lower,hls_higher)
     #Set elliptical kernel 
     kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
     #Dilate image to reduce black spots in segmented image
     maskDilated=cv2.morphologyEx(mask,cv2.MORPH_DILATE,kernel)
     return maskDilated
     
     


def segmentLanes(frame,min_area):
    global frame_hls,src
    src=frame.copy()
    #Frame is video frame from cameras
    frame_hls=cv2.cvtColor(frame,cv2.COLOR_BGR2HLS)
    mask   = segmentImage_Color(frame_hls,np.array([hue_l  ,lit_l   ,sat_l]  ), np.array([255,255,255]))
    mask_y = segmentImage_Color(frame_hls,np.array([hue_l_y,lit_l_y ,sat_l_y]),np.array([hue_h_y,255,255]))
    cv2.waitKey(0)
    

if __name__== "__main__":
    img=cv2.imread("/home/rahul/Pictures/Test_Image.png")
    segmentLanes(img,0)
    pass