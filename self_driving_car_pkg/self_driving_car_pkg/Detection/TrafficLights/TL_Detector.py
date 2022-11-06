from curses.ascii import NL
import torch
import numpy as np
import cv2
from time import time
from .Tf_tracker import Tracker
from math import dist

class TLDetection:

    def __init__(self,model_path):

        self.model = self.load_model(model_path)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.TLTracker=Tracker()
        self.closeProximity= False
        self.ThresholdArea= 5000.00
        print("Using Device: ", self.device)

    def get_video_capture(self):
        return cv2.VideoCapture(self.capture_index)

    def load_model(self, model_name):
        if model_name:
            #model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
            model = torch.hub.load('/home/rahul/yolov5_deploy/yolov5-master','custom', path=model_name,force_reload=True,source='local')
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model


    def GetBBArea(self,x1,y1,x2,y2):
        area=dist((x1,0),(x2,0)) * dist((0,y1),(0,y2))
        #print(area)
        return area
        
    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        return self.classes[int(x)]

    def plot_boxes(self, results,frame):
        labels, cord = results
        n = len(labels)  
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        best_match=0
        best_match_pbr=-1
        tl_state=""
        for i in range(n):
            row=cord[i]
            if row[4]>best_match_pbr and row[4]>0.3:
                best_match=i     
        center=(-100,-100)
        if n>0:
            row=cord[best_match]
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
            
            print(f"Second pass {self.closeProximity}")
            self.closeProximity= self.GetBBArea(x1,y1,x2,y2) > self.ThresholdArea
            print(f"After second pass {self.closeProximity}")
            
            bgr = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
            center=( (x1+x2)//2,(y1+y2)//2 )
            cv2.circle(frame,center,5,(255,0,0))
            tl_state=self.class_to_label(labels[best_match])
        return frame,center,tl_state
        

    def detect(self,frame_):
        frame=frame_[:,:,:]
        
        frame=cv2.cvtColor(frame_,cv2.COLOR_BGR2RGB)
        self.closeProximity=False
        print(f"First pass {self.closeProximity}")
        
        results = self.score_frame(frame)

        
        frame,center,current_state = self.plot_boxes(results, frame)
        #print(self.closeProximity,current_state)
        self.TLTracker.ProcessCenter(center,frame)
        if self.TLTracker.currentMode=="Tracking":
            self.TLTracker.Track(center,frame)
        frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        cv2.imshow('YOLOv5 Detection', frame)
        #print()
        if current_state=="" or current_state=="yellow":
            current_state="green"
        
        
        closeProximity=self.closeProximity
        print(f"Final pass {self.closeProximity}")
        return current_state,closeProximity
 
   
        
        
