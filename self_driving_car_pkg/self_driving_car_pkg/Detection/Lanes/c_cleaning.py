import cv2
import numpy as np

from .utilities import GetEuclideanDistance,Sort_Contour_Coordinates


#CLEANING IS SORT OF UNNECESSARY, BUT IT ENSURES SMOOTHNESS IN DETECTION

def IsPathCrossingMid(Midlane,MidlaneContour,OuterLaneContours):
    is_Ref_to_path_Left = 0
    Ref_To_CarPath_Image = np.zeros_like(Midlane)
    
    Midlane_copy = Midlane.copy()

    if not MidlaneContour:
        print("[Warning!!!] NO Midlane detected")
    
    MidLaneContours_Sorted= Sort_Contour_Coordinates(MidlaneContour,"rows")
    OuterLaneContours_Sorted = Sort_Contour_Coordinates(OuterLaneContours,"rows")
    Mid_Rows = MidLaneContours_Sorted.shape[0]
    Outer_Rows = OuterLaneContours_Sorted.shape[0]

    Mid_bottom_Pt = MidLaneContours_Sorted[Mid_Rows-1,:]
    Outer_bottom_Pt = OuterLaneContours_Sorted[Outer_Rows-1,:]

    CarTraj_bottom_Pt = ( int( (Mid_bottom_Pt[0] + Outer_bottom_Pt[0]  ) / 2 ) , int( (Mid_bottom_Pt[1]  + Outer_bottom_Pt[1] ) / 2 ) )
    

    cv2.line(Ref_To_CarPath_Image,CarTraj_bottom_Pt,(int(Ref_To_CarPath_Image.shape[1]/2),Ref_To_CarPath_Image.shape[0]),(255,255,0),2)# line from carstart to car path
    cv2.line(Midlane_copy,tuple(Mid_bottom_Pt),(Mid_bottom_Pt[0],Midlane_copy.shape[0]-1),(255,255,0),2)# connecting midlane to bottom
    #cv2.imshow("fawk",Ref_To_CarPath_Image)

    is_Ref_to_path_Left = ( (int(Ref_To_CarPath_Image.shape[1]/2) - CarTraj_bottom_Pt[0]) > 0 )

    if( np.any( (cv2.bitwise_and(Ref_To_CarPath_Image,Midlane_copy) > 0) ) ):
        # Midlane and CarPath Intersets (MidCrossing)
        return True,is_Ref_to_path_Left
    else:
        return False,is_Ref_to_path_Left


def GetYellowInnerEdge(OuterLanes,MidLane,OuterLane_Points):
    MidlaneZerosCopy=np.zeros_like(MidLane)
    Offset_correction = 0
    Outer_Lanes_ret = np.zeros_like(OuterLanes)

    # 1. Extracting Mid and OuterLaneImage Contours
    MidlaneContours = cv2.findContours(MidLane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    cv2.drawContours(MidlaneZerosCopy,MidlaneContours,-1,(255,0,0),3)
    #cv2.imshow("MidlaneZerosCopy cnt",MidlaneZerosCopy)
    OuterLaneContours = cv2.findContours(OuterLanes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    # 2. Checking if OuterLaneImage was Present initially or not
    if not OuterLaneContours:
        NoOuterLane_before=True
    else:
        NoOuterLane_before=False

    # 3. Setting the first contour of Midlane as Refrence
    Ref = (0,0) #If MidContours are present use the first ContourPoint as Ref To Find Nearest YellowLaneContour
    if(MidlaneContours):
        Ref = tuple(MidlaneContours[0][0][0])
        #cv2.circle(MidlaneZerosCopy,Ref,10,(255,255,255),-1)
        #cv2.imshow("MidlaneZerosCopy cnt",MidlaneZerosCopy)

    
    # 4. >>>>>>>>>>>>>> Condition 1 : if Both Midlane and Outlane is detected <<<<<<<<<<<<<
    if MidlaneContours:
        #print("Condition 1!")
        
        # 4.A                    ******[len(OuterLane_Points)==2)] *******
        # (a) Fetching side of outelane nearest to midlane
        if  (len(OuterLane_Points)==2):
            Point_a = OuterLane_Points[0]
            Point_b = OuterLane_Points[1]
            
            Closest_Index = 0
            if(GetEuclideanDistance(Point_a,Ref) <= GetEuclideanDistance(Point_b,Ref)):
                Closest_Index=0
            elif(len(OuterLaneContours)>1):
                Closest_Index=1
            Outer_Lanes_ret = cv2.drawContours(Outer_Lanes_ret, OuterLaneContours, Closest_Index, 255, 1)
            Outer_cnts_ret = [OuterLaneContours[Closest_Index]]

        # (b) If Correct outlane was detected =====================================
            IsPathCrossing , IsCrossingLeft = IsPathCrossingMid(MidLane,MidlaneContours,Outer_cnts_ret)
            if(IsPathCrossing):
                OuterLanes = np.zeros_like(OuterLanes)
            else:
                return Outer_Lanes_ret ,Outer_cnts_ret, MidlaneContours,0


        # 4.B                    ******[len(OuterLane_Points)!=2)] ********
        elif( np.any(OuterLanes>0) ):
            IsPathCrossing , IsCrossingLeft = IsPathCrossingMid(MidLane,MidlaneContours,OuterLaneContours)
            if(IsPathCrossing):
                OuterLanes = np.zeros_like(OuterLanes)#Empty outerLane 
            else:
                return OuterLanes ,OuterLaneContours, MidlaneContours,0

        # 4. >>>>>>>>>>>>>> Condition 2 : if MidLane is present but no Outlane detected >>>>>>>>>>>>>> Or Outlane got zerod because of crossings Midlane
        # Action: Create Outlane on Side that represent the larger Lane as seen by camera
        
        if(not np.any(OuterLanes>0)):
           # print("Condition 2!")
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
                Offset_correction = 20
            else:
                low_Col=0
                high_Col=0
                Offset_correction = -20
            
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

# def ExtendShortLane(MidLaneImage, MidlaneContours, OuterLaneContours,OuterLaneImage):

#     # 1. Sorting the Mid and Outer Contours (check Sort_contour_coordinates for more info)
#     if(MidlaneContours and OuterLaneContours):
#         MidLaneContours_Sorted= Sort_Contour_Coordinates(MidlaneContours,"rows")
#         OuterLaneContours_Sorted = Sort_Contour_Coordinates(OuterLaneContours,"rows")
#         Image_bottom = MidLaneImage.shape[0]
#         TotalNumberOfMidLaneContours = MidLaneContours_Sorted.shape[0]
#         TotalNumberOfOuterLaneContours= OuterLaneContours_Sorted.shape[0]

#         # 2. Connect Midlane to imagebottom by drawing a Vertical line if not alraedy connected
#         BottomPointofMidLane = MidLaneContours_Sorted[TotalNumberOfMidLaneContours-1]
        
#         if (BottomPointofMidLane[1] < Image_bottom):
#             MidLaneImage = cv2.line(MidLaneImage,tuple(BottomPointofMidLane),(BottomPointofMidLane[0],Image_bottom),255,2)

#         # 3. Connect Outerlane to imagebottom by performing 2 steps (if neccasary)
#             # [Step 1]: Extend Outerlane in the direction of its slope

#         ## A) Taking last 20 points to estimate slope
#         BottomPointOfOuterLane = OuterLaneContours_Sorted[TotalNumberOfOuterLaneContours -1]
        
#         if (BottomPointOfOuterLane[1] < Image_bottom):
#             if(TotalNumberOfOuterLaneContours >20):
#                 shift=20
#             else:
#                 shift=2
#             #Incase screw up happens, change to RefLast10Points = OuterLaneContours_Sorted[ TotalNumberOfOuterLaneContours-shift : TotalNumberOfOuterLaneContours-1:2 ]
#             RefLast10Points = OuterLaneContours_Sorted[ TotalNumberOfOuterLaneContours-shift : TotalNumberOfOuterLaneContours-1 ]

#             ## B) Estimating Slope
#             if(len(RefLast10Points)>1):# Atleast 2 points needed to estimate a line
#                 Ref_x = RefLast10Points[:,0]#cols
#                 Ref_y = RefLast10Points[:,1]#rows
#                 Ref_parameters = np.polyfit(Ref_x, Ref_y, 1)
#                 Ref_slope = Ref_parameters[0]
#                 Ref_yiCntercept = Ref_parameters[1]
                
                
#                 # Ref_x1 = RefLast10Points[:,0]#cols
#                 # Ref_y1 = RefLast10Points[:,1]#rows
#                 # Ref_parameters1 = np.polyfit(Ref_x1, Ref_y1, 1)
#                 # Ref_yiCntercept1 = Ref_parameters1[1]
#                 # print("Y intercept without taking all point",Ref_yiCntercept1)
                
                
                
                
#                 cv2.imshow("OuterLane before extension",OuterLaneImage)
#                 ## C) Extending outerlane in the direction of its slope
#                 if(Ref_slope < 0):
#                     Ref_LineTouchPoint_col = 0
#                     Ref_LineTouchPoint_row = Ref_yiCntercept
#                 else:
#                     Ref_LineTouchPoint_col = OuterLaneImage.shape[1]-1 # Cols have lenth of ColLength But traversal is from 0 to ColLength-1
#                     Ref_LineTouchPoint_row = Ref_slope * Ref_LineTouchPoint_col + Ref_yiCntercept
#                 Ref_TouchPoint = (Ref_LineTouchPoint_col,int(Ref_LineTouchPoint_row))#(col ,row)
#                 Ref_BottomPoint_tup = tuple(BottomPointOfOuterLane)
#                 OuterLaneImage = cv2.line(OuterLaneImage,Ref_TouchPoint,Ref_BottomPoint_tup,255,2)
#                 cv2.imshow("OuterLane after extension",OuterLaneImage)
                

#                 # 3 [Step 2]: If required, connect outerlane to bottom by drawing a vertical line
#                 if(Ref_LineTouchPoint_row < Image_bottom):
#                     Ref_TouchPoint_Ref = (Ref_LineTouchPoint_col,Image_bottom)
#                     OuterLaneImage = cv2.line(OuterLaneImage,Ref_TouchPoint,Ref_TouchPoint_Ref,255,3)
                

#     return MidLaneImage,OuterLaneImage


        