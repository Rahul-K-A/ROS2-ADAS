import cv2
from .Detection.Lanes.lane_detection import detect_lanes
from numpy import interp,uint16
from .Detection.Signs.sign_detection import detect_signs
from .Detection.TrafficLights.TL_Detector import TLDetection
from .config.config import yolo_tl_model_path

debugEnabled=False

class Control():
    def __init__(self):
        # Lane assist Variable
        self.angle = 0.0
        self.speed = 80
        # Cruise_Control Variable
        self.prev_Mode_Sign = "Detection"
        self.IncreaseTireSpeedInTurns = False
        self.TLDetector=TLDetection(yolo_tl_model_path)
        self.TL_prev_State=""
        self.was_TL_last_frame=False
        self.TL_Ignore_LaneFollow_iter=0
    
    def follow_lane(self,max_sane_dist,dist,curv,mode,tracked_class):
        #2. Cruise control speed adjusted to match road speed limit
        self.speed=60
        
        if debugEnabled:
            if((tracked_class!=0) and (self.prev_Mode_Sign == "Tracking") and (mode == "Detection")):
                if  (tracked_class =="speed_sign_30"):
                    self.speed = 30
                elif(tracked_class =="speed_sign_60"):
                    self.speed = 60
                elif(tracked_class =="speed_sign_90"):
                    self.speed = 90
                elif(tracked_class =="stop"):
                    self.speed = 0
                    print("Stopping Car !!!")
        
            
        self.prev_Mode_Sign = mode # Set prevMode to current Mode

        max_turn_angle = 90; max_turn_angle_neg =-90; req_turn_angle = 0

        if ((dist>max_sane_dist)or (dist < (-1*max_sane_dist))):
            if(dist>max_sane_dist):
                req_turn_angle = max_turn_angle + curv
            else:
                req_turn_angle = max_turn_angle_neg + curv
        else:
            car_offset = interp(dist,[-max_sane_dist,max_sane_dist],[-max_turn_angle,max_turn_angle])
            req_turn_angle = car_offset + curv
        
        #handle overflow
        if ((req_turn_angle>max_turn_angle)or (req_turn_angle<max_turn_angle_neg)):
            if (req_turn_angle>max_turn_angle):
                req_turn_angle = max_turn_angle
            else:
                req_turn_angle = max_turn_angle_neg
        # Handle max car turn ability
        self.angle = interp(req_turn_angle,[max_turn_angle_neg,max_turn_angle],[-45,45])
        if (self.IncreaseTireSpeedInTurns and (tracked_class !="left_turn")):
            if(self.angle>30):
                car_speed_turn = interp(self.angle,[30,45],[80,100])
                self.speed = car_speed_turn
            elif(self.angle<(-30)):
                car_speed_turn = interp(self.angle,[-45,-30],[100,80])
                self.speed = car_speed_turn
                
    def drive(self,Current_State,Current_TL_state):
        # if Current_TL_state!="":
        #     if self.was_TL_last_frame:
        #         self.was_TL_last_frame=True
        #         self.TL_Ignore_LaneFollow_iter=0
        #         if Current_TL_state=="red":
        #             speed_motor=0.5
        #         else:
        #             speed_motor=1.5
        #         angle_motor=0.0
        #     else:
        #         if self.TL_Ignore_LaneFollow_iter<=60:
        #             if Current_TL_state=="red":
        #                 speed_motor=0.5
        #             else:
        #                 speed_motor=1.5
        #         angle_motor=0.0
        #     return angle_motor,speed_motor
        # 
        # if Current_TL_state!="":
        #     if self.TL_prev_State=="red" and Current_TL_state="green":
                
        #     if self.TL_prev_Mode:
                
        [dist,curv,img,mode,tracked_class] = Current_State

        if ((dist!=1000)and (curv!= 1000)):
            self.follow_lane(img.shape[1]/4,dist,curv,mode,tracked_class)
        else:
            self.speed = 0.0 # Stop the car

        # Interpolating the angle and speed from real world to motor worlld
        angle_motor = interp(self.angle,[-45,45],[0.5,-0.5])
        if (self.speed!=0):
            speed_motor = interp(self.speed,[30,90] ,[1,2])
        else:
            speed_motor = 0.0

        return angle_motor,speed_motor



class Car():
    def __init__(self):
        self.Control = Control()

    def display_state(self,frame_disp,angle_of_car,current_speed,tracked_class):

        # Translate [ ROS Car Control Range ===> Real World angle and speed  ]
        angle_of_car  = interp(angle_of_car,[-0.5,0.5],[45,-45])
        if (current_speed !=0.0):
            current_speed = interp(current_speed  ,[1  ,   2],[30 ,90])

        ###################################################  Displaying CONTROL STATE ####################################

        if (angle_of_car <-10):
            direction_string="[ Left ]"
            color_direction=(120,0,255)
        elif (angle_of_car >10):
            direction_string="[ Right ]"
            color_direction=(120,0,255)
        else:
            direction_string="[ Straight ]"
            color_direction=(0,255,0)

        if(current_speed>0):
            direction_string = "Moving --> "+ direction_string
        else:
            color_direction=(0,0,255)


        cv2.putText(frame_disp,str(direction_string),(20,40),cv2.FONT_HERSHEY_DUPLEX,0.4,color_direction,1)

        angle_speed_str = "[ Angle ,Speed ] = [ " + str(int(angle_of_car)) + "deg ," + str(int(current_speed)) + "mph ]"
        cv2.putText(frame_disp,str(angle_speed_str),(20,20),cv2.FONT_HERSHEY_DUPLEX,0.4,(0,0,255),1)
        
        font_Scale = 0.37
       # cv2.putText(frame_disp,"Sign Detected ==> "+str(tracked_class),(20,80),cv2.FONT_HERSHEY_COMPLEX,font_Scale,(0,255,255),1)    

    def drive_car(self,frame):

        img = frame[0:640,238:1042]
        # resizing to minimize computation time while still achieving comparable results
        img = cv2.resize(img,(320,240))

        img_orig = img.copy()
        TL_state=self.Control.TLDetector.detect(img_orig)
        #print(TL_state)
        
        distance, Curvature = detect_lanes(img)
        

        #mode, tracked_class = detect_signs(img_orig,img)
        mode="Tracking"
        tracked_class=0
        
        Current_State = [distance,Curvature,img,mode,tracked_class]
        
        angle_m,speed_m = self.Control.drive(Current_State,TL_state)

        self.display_state(img,angle_m,speed_m,tracked_class)

        return angle_m,speed_m,img