import cv2
import numpy as np

from .utilities import Sort_Contour_Coordinates,findlaneCurvature

debuggingEnabled=False

#Need to offset because we are only estimating the outerlane, not the actual outerlane


def LanePoints(midlane,outerlane,offset):
    """Return the points corresponding to the required trajectory"""
    
    Midlane_contours = cv2.findContours(midlane,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    outer_cnts = cv2.findContours(outerlane,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    if Midlane_contours and outer_cnts:
        
        #Sort the contour points from top right to bottom left
        Midlane_contours_sorted = Sort_Contour_Coordinates(Midlane_contours,"rows")
        outer_cnts_row_sorted = Sort_Contour_Coordinates(outer_cnts,"rows")

        num_MidlanePoints = Midlane_contours_sorted.shape[0]
        num_OuterlanePoints = outer_cnts_row_sorted.shape[0]

        Midlane_Bottom_pt = Midlane_contours_sorted[num_MidlanePoints-1,:]
        OuterLane_Bottom_pt = outer_cnts_row_sorted[num_OuterlanePoints -1,:]
        Midlane_Top_pt = Midlane_contours_sorted[0,:]
        OuterLane_Top_pt = outer_cnts_row_sorted[0,:]
        
        #get the midpoint of the bottom most and top most points of the 
        Trajectory_Bottom_pt = ( int((Midlane_Bottom_pt[0] + OuterLane_Bottom_pt[0])/2)+ offset ,int((Midlane_Bottom_pt[1] + OuterLane_Bottom_pt[1])/2))
        Trajectory_Top_pt = ( int((Midlane_Top_pt[0] + OuterLane_Top_pt[0])/2)+ offset ,int((Midlane_Top_pt[1] + OuterLane_Top_pt[1])/2))
        
        if debuggingEnabled:
            Midlane_Copy=cv2.cvtColor(midlane.copy(),cv2.COLOR_GRAY2BGR)
            
            Trajectory_Bottom_pt_Debug = ( int((Midlane_Bottom_pt[0] + OuterLane_Bottom_pt[0])/2),int((Midlane_Bottom_pt[1] + OuterLane_Bottom_pt[1])/2))
            Trajectory_Top_pt_Debug = ( int((Midlane_Top_pt[0] + OuterLane_Top_pt[0])/2) ,int((Midlane_Top_pt[1] + OuterLane_Top_pt[1])/2))
            
            cv2.circle(Midlane_Copy,Trajectory_Bottom_pt,5,(255,255,255),-1)
            cv2.circle(Midlane_Copy,Trajectory_Top_pt,5,(0,255,0),-1)
            
            cv2.circle(Midlane_Copy,Trajectory_Top_pt_Debug,5,(255,0,0),-1)
            cv2.circle(Midlane_Copy,Trajectory_Bottom_pt_Debug,5,(0,0,255),-1)
            cv2.imshow("Midlane trajectory points debug",Midlane_Copy)
        return  Trajectory_Bottom_pt,Trajectory_Top_pt

    else:
        return (0,0),(0,0)

def EstimateNonMidMask(MidEdgeROI):
    Mid_Hull_Mask = np.zeros_like(MidEdgeROI)
    contours = cv2.findContours(MidEdgeROI,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]
    if contours:
        hull_list = []
        contours = np.concatenate(contours)
        hull = cv2.convexHull(contours)
        hull_list.append(hull)
        Mid_Hull_Mask = cv2.drawContours(Mid_Hull_Mask, hull_list, 0, 255,-1)
        
    if debuggingEnabled:
        cv2.imshow("Mid hull mask",Mid_Hull_Mask)
    Non_Mid_Mask=cv2.bitwise_not(Mid_Hull_Mask)
    return Non_Mid_Mask


def FetchInfoAndDisplay(Mid_lane_edge,Mid_lane,Outer_Lane,frame,Offset_correction):
    # 1. Using Both outer and middle information to create probable path
    Traj_lowP,Traj_upP = LanePoints(Mid_lane,Outer_Lane,Offset_correction)
    
    # 2. Compute Distance and Curvature from Trajectory Points 
    DistanceCarNoseDetectedLane = -1000
    if(Traj_lowP!=(0,0)):
        DistanceCarNoseDetectedLane = Traj_lowP[0] - int(Mid_lane.shape[1]/2)
        
    curvature = findlaneCurvature(Traj_lowP[0],Traj_lowP[1],Traj_upP[0],Traj_upP[1])

    # 3. Keep only those edge that are part of MIDLANE
    #Mid_lane_edge = cv2.bitwise_and(Mid_lane_edge,Mid_lane)

    # 4. Combine Mid and OuterLane to get Lanes Combined and extract its contours
    Lanes_combined = cv2.bitwise_or(Outer_Lane,Mid_lane)
    cv2.imshow("Lanes_combined",Lanes_combined)
    ProjectedLane = np.zeros_like(Lanes_combined)
    cnts = cv2.findContours(Lanes_combined,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]

    # 5. Fill ProjectedLane with fillConvexPoly
    if cnts:
        cnts = np.concatenate(cnts)
        cnts = np.array(cnts)
        cv2.fillConvexPoly(ProjectedLane, cnts, 255)
    if debuggingEnabled:
        cv2.imshow("Projected lane b4 mask",ProjectedLane)

    # 6. Remove Midlane_Region from ProjectedLane by extracting the midless mask
    # Mid_less_Mask = EstimateNonMidMask(Mid_lane_edge)
    # ProjectedLane = cv2.bitwise_and(Mid_less_Mask,ProjectedLane)

    # 7. Draw projected lane
    Lane_drawn_frame = frame
    Lane_drawn_frame[ProjectedLane==255] = Lane_drawn_frame[ProjectedLane==255] + (0,100,0)
    # Lane_drawn_frame[Outer_Lane==255] = Lane_drawn_frame[Outer_Lane==255] + (0,0,100)# Outer Lane Coloured Red
    # Lane_drawn_frame[Mid_lane==255] = Lane_drawn_frame[Mid_lane==255] + (100,0,0)# Mid Lane Coloured Blue
    Out_image = Lane_drawn_frame

    # 8. Draw Cars direction and Lanes direction and distance between car and lane path
    cv2.line(Out_image,(int(Out_image.shape[1]/2),Out_image.shape[0]),(int(Out_image.shape[1]/2),Out_image.shape[0]-int (Out_image.shape[0]/5)),(0,0,255),2)
    cv2.line(Out_image,Traj_lowP,Traj_upP,(255,0,0),2)
    if(Traj_lowP!=(0,0)):
        cv2.line(Out_image,Traj_lowP,(int(Out_image.shape[1]/2),Traj_lowP[1]),(255,255,0),2)# distance of car center with lane path

    # 9. Draw extracted distance and curvature 
    curvature_str="Curvature = " + f"{curvature:.2f}"
    PerpDist_ImgCen_CarNose_str="Distance = " + str(DistanceCarNoseDetectedLane )
    textSize_ratio = 0.5
    cv2.putText(Out_image,curvature_str,(10,30),cv2.FONT_HERSHEY_DUPLEX,textSize_ratio,(0,255,255),1)
    cv2.putText(Out_image,PerpDist_ImgCen_CarNose_str,(10,50),cv2.FONT_HERSHEY_DUPLEX,textSize_ratio,(0,255,255),1)
    return DistanceCarNoseDetectedLane ,curvature



    




