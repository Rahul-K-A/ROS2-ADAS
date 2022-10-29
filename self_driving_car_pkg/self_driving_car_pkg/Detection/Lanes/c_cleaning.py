import cv2
import numpy as np

from .utilities import GetEuclideanDistance,Sort_Contour_Coordinates


debuggingEnabled=False

#CLEANING IS SORT OF UNNECESSARY, BUT IT ENSURES SMOOTHNESS IN DETECTION


#Check if trajectory makes the car swerve throught the midlane
def DoesPathCrossMidlane(Midlane,MidlaneContour,OuterLaneContours):
    IsTrajectoryLeftOfMidlane= 0
    Ref_To_CarPath_Image = np.zeros_like(Midlane)
    Midlane_copy = Midlane.copy()
    
    
    

    #If no detected midlanes, error
    if not MidlaneContour:
        print("[Warning!!!] NO Midlane detected")
    
    #Sort all contour cordinates
    MidLaneContours_Sorted= Sort_Contour_Coordinates(MidlaneContour,"rows")
    OuterLaneContours_Sorted = Sort_Contour_Coordinates(OuterLaneContours,"rows")
    
    #Get shape of sorted coordinates
    Mid_Rows_Shape = MidLaneContours_Sorted.shape[0]
    Outer_Rows_Shape = OuterLaneContours_Sorted.shape[0]

    #Get the Bottom most points of the lanes as well as the car trajectory in (x,y) format
    MidLane_BottomMost_Pt = MidLaneContours_Sorted[Mid_Rows_Shape-1,:]
    OuterLane_BottomMost_Pt = OuterLaneContours_Sorted[Outer_Rows_Shape-1,:]
    CarTrajectory_BottomPoint =  int( (MidLane_BottomMost_Pt[0] + OuterLane_BottomMost_Pt[0]  ) / 2 ) , int( (MidLane_BottomMost_Pt[1]  + OuterLane_BottomMost_Pt[1] ) / 2 ) 
    
    # Draw line from bottom of car to trajectory bottom point
    cv2.line(Ref_To_CarPath_Image,CarTrajectory_BottomPoint,(int(Ref_To_CarPath_Image.shape[1]/2),Ref_To_CarPath_Image.shape[0]),(255,255,0),2)
    
    # Draw midlane connection line from bottom most midlane point to bottom of image
    cv2.line(Midlane_copy,MidLane_BottomMost_Pt,( MidLane_BottomMost_Pt[0],Midlane_copy.shape[0]-1),(255,255,0),2)
    
   

    
    

    
    
    #if x1 is to the right of x2, then (x1-x2) >0
    #else x1 is the left of x2 if (x1-x2) <0 int(Ref_To_CarPath_Image.shape[1]/2 - CarTrajectory_BottomPoint[0])
    #Same way, check if the trajectory path passes through the midlane from the left or to the right by subtracting the x coords
    IsTrajectoryLeftOfMidlane = ( int(Ref_To_CarPath_Image.shape[1]/2 - CarTrajectory_BottomPoint[0]) > 0 )
    
    if debuggingEnabled:
        Midlane_copy2 = Midlane.copy()
        cv2.line(Midlane_copy2,CarTrajectory_BottomPoint,(int(Ref_To_CarPath_Image.shape[1]/2),Ref_To_CarPath_Image.shape[0]),(255,255,0),5)# line from carstart to car path
        cv2.line(Midlane_copy2,tuple( MidLane_BottomMost_Pt),( MidLane_BottomMost_Pt[0],Midlane_copy.shape[0]-1),(255,255,0),10)# connecting midlane to 
        
        cv2.circle(Midlane_copy2,tuple(CarTrajectory_BottomPoint),5,(255,255,255),-1)
        cv2.circle(Midlane_copy2,tuple(MidLane_BottomMost_Pt),5,(255,0,255),-1)
        cv2.circle(Midlane_copy2,tuple(OuterLane_BottomMost_Pt),5,(255,255,0),-1)
        
        cv2.circle(Midlane_copy2,(Ref_To_CarPath_Image.shape[1]//2, Ref_To_CarPath_Image.shape[0]//2 ),10,(255,255,0),-1)
        
        print(Ref_To_CarPath_Image.shape[1]/2,CarTrajectory_BottomPoint[0],(int(Ref_To_CarPath_Image.shape[1]/2) - CarTrajectory_BottomPoint[0]))
        cv2.imshow("Midlane crossing debug",Midlane_copy2 )
        print(IsTrajectoryLeftOfMidlane ,int(Ref_To_CarPath_Image.shape[1]/2 - CarTrajectory_BottomPoint[0]))
    
    
    #If theres is overlap betweem the Car trajectory line and midlane contours, then return True and also what direction it is overlapping from
    if( np.any( (cv2.bitwise_and(Ref_To_CarPath_Image,Midlane_copy) > 0) ) ):
        #print("Yes")
        return True,IsTrajectoryLeftOfMidlane 
    else:
    #Else return false and also what direction it is overlapping from
        
        #print("No")
        return False,IsTrajectoryLeftOfMidlane 


def GetYellowInnerEdge(OuterLanes,MidLane,OuterLane_Points):
    MidlaneZerosCopy=np.zeros_like(MidLane)
    Offset_correction = 0
    Outer_Lanes_ret = np.zeros_like(OuterLanes)

    # 1. Extracting Mid and OuterLaneImage Contours
    MidlaneContours = cv2.findContours(MidLane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    OuterLaneContours = cv2.findContours(OuterLanes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # 2. Keep flag to check whether the outer lane contours was present in the last frame
    if not OuterLaneContours:
        NoOuterLane_before=True
    else:
        NoOuterLane_before=False

    # 3. Create a ref point in the mid lane contour (Initial value for now)
    Ref = (0,0) 
    #Use reference to Find Nearest YellowLaneContour
    if MidlaneContours:
        Ref = tuple(MidlaneContours[0][0][0])
    
    # 4. >>>>>>>>>>>>>> Condition 1 : if Both Midlane and Outlane is detected <<<<<<<<<<<<<
    if MidlaneContours:
        
        #If both outer lanes were detected , pick the nearest one
        if  (len(OuterLane_Points)==2):
            Point_a = OuterLane_Points[0]
            Point_b = OuterLane_Points[1]
            
            #Pick the right most outer lane
            # Closest_Index = 0
            # if (Ref[0]-Point_a[0])>0:
            #     Closest_Index=0
            # else:
            #     Closest_Index=1
                
            
            
            if(GetEuclideanDistance(Point_a,Ref) <= GetEuclideanDistance(Point_b,Ref)):
                 Closest_Index=0
            elif(len(OuterLaneContours)>1):
                 Closest_Index=1
            Outer_Lanes_ret = cv2.drawContours(Outer_Lanes_ret, OuterLaneContours, Closest_Index, 255, 1)
            Outer_cnts_ret = [OuterLaneContours[Closest_Index]]

        # (b) If Correct outlane was detected =====================================
            IsPathCrossing , IsCrossingLeft = DoesPathCrossMidlane(MidLane,MidlaneContours,Outer_cnts_ret)
            if (IsPathCrossing):
                OuterLanes = np.zeros_like(OuterLanes)
            else:
                return Outer_Lanes_ret ,Outer_cnts_ret, MidlaneContours,0

        # 4.B                    ******[len(OuterLane_Points)!=2)] ********
        elif( np.any(OuterLanes>0) ):
            IsPathCrossing , IsCrossingLeft = DoesPathCrossMidlane(MidLane,MidlaneContours,OuterLaneContours)
            if(IsPathCrossing):
                OuterLanes = np.zeros_like(OuterLanes)#Empty outerLane
            else:
                return OuterLanes ,OuterLaneContours, MidlaneContours,0
        
        # 4. >>>>>>>>>>>>>> Condition 2 : if MidLane is present but no Outlane detected >>>>>>>>>>>>>> Or Outlane got zerod because of crossings Midlane
        # Action: Create Outlane on Side that represent the larger Lane as seen by camera
        if(not np.any(OuterLanes>0)):
            # Fetching the column of the lowest point of the midlane 
            MidLaneContours_Sorted= Sort_Contour_Coordinates(MidlaneContours,"rows")
            Mid_Rows = MidLaneContours_Sorted.shape[0]
            Mid_lowP = MidLaneContours_Sorted[Mid_Rows-1,:]
            Mid_highP = MidLaneContours_Sorted[0,:]
            Mid_low_Col = Mid_lowP[0]

            # Addresing which side to draw the outerlane considering it was present before or not		
            DrawRight = False
            if NoOuterLane_before:
                if(Mid_low_Col < int(MidLane.shape[1]/2)):
                    DrawRight = True
            else:
                if IsCrossingLeft:
                    DrawRight = True

            # Setting outerlane upperand lower points column to the right if draw right and vice versa
            if DrawRight:
                low_Col=(int(MidLane.shape[1])-1)
                high_Col=(int(MidLane.shape[1])-1)
                Offset_correction = 25
            else:
                low_Col=0
                high_Col=0
                Offset_correction = -25
            
            Mid_lowP[1] = MidLane.shape[0]# setting mid_trajectory_lowestPoint_Row to MaxRows of Image
            LanePoint_lower =  (low_Col , int( Mid_lowP[1] ) )
            LanePoint_top   =  (high_Col, int( Mid_highP[1]) )
            OuterLanes = cv2.line(OuterLanes,LanePoint_lower,LanePoint_top,255,1)
            OuterLaneContours = cv2.findContours(OuterLanes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
            return OuterLanes, OuterLaneContours, MidlaneContours, Offset_correction

    # 5. Condition 3 [No MidLane]
    else:
        #print("Condition 3!")
        return OuterLanes, OuterLaneContours, MidlaneContours, Offset_correction




        