import cv2
import numpy as np

from .Morph_op import RejectSmallerContours, ReturnLowestEdgePoints,ReturnLargestContour_OuterLane

debuggingEnabled=False
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
    
    maskWhite   = SegmentByColor(hls,(hue_l  ,lit_l   ,sat_l  ),(255       ,255,255))
    maskYellow =  SegmentByColor(hls,(hue_l_y,lit_l_y ,sat_l_y),(hue_h_y,255,255))

    #Apply masks
    whiteRegion = cv2.bitwise_and(src,src,mask=maskWhite)
    yellowRegion = cv2.bitwise_and(src,src,mask=maskYellow)
    
    if debuggingEnabled:
        cv2.imshow('white_regions',whiteRegion)
        cv2.imshow('yellow_regions',yellowRegion)


def GetLargerObjectsOnly(frame,mask,min_area):
    
    # Keeping only objects larger then min_area
    frameROI = cv2.bitwise_and(frame,frame,mask=mask)
    frameROI_gray = cv2.cvtColor(frameROI,cv2.COLOR_BGR2GRAY)
    mask_of_larger_objects = RejectSmallerContours(frameROI_gray,min_area)
    frameROI_gray = cv2.bitwise_and(frameROI_gray,mask_of_larger_objects)
    
    # Extracting Edges of those larger objects
    frameROI_smoothed = cv2.GaussianBlur(frameROI_gray,(11,11),1)
    edges_of_larger_objects = cv2.Canny(frameROI_smoothed,50,150, None, 3)
    return mask_of_larger_objects,edges_of_larger_objects

def SegmentMidLane(frame,white_regions,min_area):
    mid_lane_mask ,mid_lane_edge = GetLargerObjectsOnly(frame,white_regions,min_area)
    return mid_lane_mask,mid_lane_edge

def SegmentOuterLane(frame,yellow_regions,min_area):
    outer_points_list = []
    
    #Get just the larger contours
    mask,edges = GetLargerObjectsOnly(frame,yellow_regions,min_area)
    
    #Return the largest contour within the large contours
    mask_largest, largest_found = ReturnLargestContour_OuterLane(mask,min_area)

    if largest_found:
        #cv2.imshow("Largest contour",mask_largest)
        # Keep only edges of largest region
        largestEdge= cv2.bitwise_and(edges,mask_largest)
        # Return edge points for identifying closest edge later
        lanes_sides_sep,outer_points_list= ReturnLowestEdgePoints(mask_largest)
        cv2.imshow("Lanes seperated",lanes_sides_sep)
        
        
        edges = largestEdge
    else:
        lanes_sides_sep = np.zeros((frame.shape[0],frame.shape[1]),np.uint8)
    
    return edges,lanes_sides_sep,outer_points_list


def SegmentByColor(hls,lower_range,upper_range):
    mask_in_range = cv2.inRange(hls,lower_range,upper_range)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask_dilated = cv2.morphologyEx(mask_in_range,cv2.MORPH_DILATE,kernel)
    return mask_dilated


def SegmentLanes(frame,min_area):
    global hls,src
    src = frame.copy()

    hls = cv2.cvtColor(frame,cv2.COLOR_BGR2HLS)

    # Segmenting White regions
    whiteRegion = SegmentByColor(hls,np.array([hue_l,lit_l,sat_l]),np.array([255,255,255]))
    yellowRegion = SegmentByColor(hls,np.array([hue_l_y,lit_l_y,sat_l_y]),np.array([hue_h_y,255,255]))
    
    
    if debuggingEnabled:
        cv2.imshow("white_regions",whiteRegion)
        cv2.imshow("yellow_regions",yellowRegion)
        cv2.waitKey(1)

    # Semgneting midlane from white regions
    MidLaneMask,MidLaneEdge = SegmentMidLane(frame,whiteRegion,min_area)

    # Semgneting outerlane from yellow regions
    OuterLaneEdge,outerlane_side_sep,OuterLanePoints= SegmentOuterLane(frame,yellowRegion,min_area+500)        

    return MidLaneMask,MidLaneEdge,OuterLaneEdge,outerlane_side_sep,OuterLanePoints
    

