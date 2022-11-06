import cv2
import numpy
import math

class Tracker:
    def __init__(self) -> None:
        self.modes=["Tracking","Detection"]
        self.changeMode("Detection")
        self.prevMode="Detection"
        self.firstCenter=(0,0)
        self.prevCenter=(0,0)
        self.centers=[]
        self.ThresholdDist=5
        
    def ProcessCenter(self,center,img):
        if self.currentMode=="Detection":
            if center==(-100,-100):
                return
            else:
                self.start_tracker(center)
        else:
            if center==(-100,-100):
                self.reset_tracker()
                return
            self.Track(center,img)
    
    def start_tracker(self,center):
        self.prevCenter=center
        self.firstCenter=center
        self.changeMode("Tracking")
        
    def reset_tracker(self):
        self.changeMode("Detection")
        self.prevMode="Detection"
        self.firstCenter=(0,0)
        self.prevCenter=(0,0)
        self.centers=[]
        
    def Track(self,center,frame):
        if self.currentMode=="Tracking":
            dist=math.dist(center,self.prevCenter)
            # print(dist)
            if dist<self.ThresholdDist:
                cv2.line(frame,self.firstCenter,center,(0,0,255),5)
                self.prevCenter=center
            
        
    def changeMode(self,mode):
        print(f"Mode has been changed to {mode}")
        self.currentMode=mode
    