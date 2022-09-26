import cv2
from .Morph_op import ApproxDistanceBetweenCenters,ReturnLargestContour


def estimate_midlane(midlane_patches,MaxAllowedDistance):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    midlane_patches = cv2.morphologyEx(midlane_patches,cv2.MORPH_DILATE,kernel)

    # 1. keep a Midlane_draw for displaying shortest connectivity later on
    midlane_connectivity_bgr = cv2.cvtColor(midlane_patches,cv2.COLOR_GRAY2BGR)

    # 2. Extract the Contours that define each object
    contours = cv2.findContours(midlane_patches,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[1]

    # 3. Keep Only those contours that are not lines 
    MinimumArea = 1
    LegitimateContours = []
    for _,contour in enumerate(contours):
        contour_area = cv2.contourArea(contour)
        if (contour_area>MinimumArea):
            LegitimateContours.append(contour)
    contours = LegitimateContours

    # 4. Connect each contous with its closest 
    #                       & 
    #    disconnecting any that may be farther then x distance

    ContourIdx_ofBestMatch = []# [BstMatchwithCnt0,BstMatchwithCnt1,....]
    for index, Contour in enumerate(contours):
        #Set initial values (prev_min)
        prevmin_dist = 100000 ; Index_BestContourFound = 0 ; BestMatch_Centroid_a=0  ; BestMatch_Centroid_b=0 
        
        #Compare the current contour with all other contours to find the closest one   
        for IndexToCompare in range(len(contours)-index):
            IndexToCompare = IndexToCompare + index
            ContourToCompare = contours[IndexToCompare]

            if (index!= IndexToCompare):
                min_dist, cent_a , cent_b = ApproxDistanceBetweenCenters(Contour,ContourToCompare)
                if (min_dist<prevmin_dist):
                    if (len(ContourIdx_ofBestMatch)==0):
                        prevmin_dist = min_dist
                        Index_BestContourFound = IndexToCompare
                        BestMatch_Centroid_a = cent_a
                        BestMatch_Centroid_b = cent_b
                    else:
                        already_present= False
                        for i in range(len(ContourIdx_ofBestMatch)):
                            if ( (IndexToCompare == i) and (index == ContourIdx_ofBestMatch[i]) ):
                                already_present = True
                        if not already_present:
                            prevmin_dist = min_dist
                            Index_BestContourFound = IndexToCompare
                            BestMatch_Centroid_a = cent_a
                            BestMatch_Centroid_b = cent_b

        if ((prevmin_dist!= 100_000) and (prevmin_dist>MaxAllowedDistance)):
            #print("prev_mindist > Max Allowed Dist !!!")
            break
        if (type(BestMatch_Centroid_a)!= int):
            ContourIdx_ofBestMatch.append(Index_BestContourFound) 
            cv2.line(midlane_connectivity_bgr,BestMatch_Centroid_a,BestMatch_Centroid_b,(0,255,0),2)

    midlane_connectivity = cv2.cvtColor(midlane_connectivity_bgr,cv2.COLOR_BGR2GRAY)                                   


    # 5. Get estimated midlane by returning the largest contour

    estimated_midlane, largest_found = ReturnLargestContour(midlane_connectivity)

    # 6. Return Estimated Midlane if found otherwise send original

    if largest_found:
        return estimated_midlane
    else:
        return midlane_patches