
#!/usr/bin/python3
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from time import sleep

from RPi_GPIO_i2c_LCD import lcd
import RPi.GPIO as GPIO
import RPi.GPIO as GPIO

sensor_camera = 3
sensor_1 = 16
sensor_2 = 18
sensor_3 = 22
servo_1 = 24
servo_2 = 26
servo_3 = 32


# Setup GPIO for Raspberry pi
GPIO.setmode(GPIO.BOARD)

GPIO.setup(sensor_camera,GPIO.IN)
GPIO.setup(sensor_1,GPIO.IN)
GPIO.setup(sensor_2,GPIO.IN)
GPIO.setup(sensor_3,GPIO.IN)

GPIO.setup(servo_1,GPIO.OUT)
GPIO.setup(servo_2,GPIO.OUT)
GPIO.setup(servo_3,GPIO.OUT)


# Setup servo
p1=GPIO.PWM(servo_1,50)# 50hz frequency
p2=GPIO.PWM(servo_2,50)
p3=GPIO.PWM(servo_3,50)

p1.start(2.5)# starting duty cycle ( it set the servo to 0 degree )
p2.start(2.5)
p3.start(2.5)
# time.sleep(2)

# Setup pi camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (1280, 720)}))
picam2.start()

i2c_address = 0x27
lcdDisplay = lcd.HD44780(i2c_address)

def mango_cut(img):
    """
    Capture mango in image
        Arg: 
            img: Mango image that taken on conveyor
        Return:
            dst: Captured mango in image
    """
    img_cut=img[30:500,50:1200]
    # cv2.imwrite("cut.jpg",img)
    gray_image = cv2.cvtColor(img_cut, cv2.COLOR_BGR2GRAY)

    # blurred = cv2.medianBlur(gray_image, 15)

    ret,thresh = cv2.threshold(gray_image,127,255,0)
    blurred_bi=cv2.medianBlur(thresh, 15)

    # detect edge around object
    wide = cv2.Canny(blurred_bi, 50, 150)

    # make edge wider 
    th = cv2.threshold(wide, 150, 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    morph = cv2.morphologyEx(th, cv2.MORPH_DILATE, kernel)
    cv2.imwrite("morph.jpg",morph)


    cnts1 = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for i in cnts1[0]:
    #     print(len(i))
    # get polygon around object
    cnts=[]
    temp_cnts=max(cnts1[0],key=len)
    for i in range(len(temp_cnts)):
        cnts.append(temp_cnts[i][0])

    pts=np.array(cnts)
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    croped = img_cut[y:y+h, x:x+w].copy()

    ## (2) make mask
    pts = pts - pts.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)

    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

    ## (3) do bit-op
    dst = cv2.bitwise_and(croped, croped, mask=mask)

    # cv2.imwrite('a0.jpg',dst)
    # print(dst.shape)
    ## (4) add the white background
    bg = np.ones_like(croped, np.uint8)*255
    cv2.bitwise_not(bg,bg, mask=mask)
    dst2 = bg+ dst

    if cv2.countNonZero(mask)>100000:
        print("Duoc cat")
        return dst
    else:
        print("Khong duoc cat")
        return img

def detect_ripeness_by_hsv(image):
    """
    Get yellow mask and green mask on captured mango image
        Arg: 
            image: Captured mango image
        Return:
            yellow_mask: Yellow mask of image 
            green_mask: Green mask of image
    """
    hsvFrame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    yellow_lower = np.array([20, 50, 0], np.uint8) 
    yellow_upper = np.array([27, 255, 255], np.uint8) 
    yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper) 
  
    # Set range for green color and  
    # define mask 
    green_lower = np.array([27.1, 50, 0], np.uint8) 
    green_upper = np.array([60, 255, 255], np.uint8) 
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

    return yellow_mask, green_mask

def catchu_vang_ripeness(image):
    """
    Get yellow masks on captured mango image
        Arg: 
            image: Captured mango image
        Return:
            [light_mask, light1_mask, light2_mask, dark_mask]: 4 level brightness of yellow mask of image 
    """
    hsvFrame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    light_lower = np.array([20, 0, 0], np.uint8) 
    light_upper = np.array([30, 125, 255], np.uint8) 
    light_mask = cv2.inRange(hsvFrame, light_lower, light_upper)

    light1_lower = np.array([20, 126, 0], np.uint8) 
    light1_upper = np.array([30, 150, 255], np.uint8) 
    light1_mask = cv2.inRange(hsvFrame, light1_lower, light1_upper)

    light2_lower = np.array([20, 151, 0], np.uint8) 
    light2_upper = np.array([30, 175, 255], np.uint8) 
    light2_mask = cv2.inRange(hsvFrame, light2_lower, light2_upper)
    # Set range for green color and  
    # define mask 
    dark_lower = np.array([20, 176, 0], np.uint8) 
    dark_upper = np.array([30, 255, 255], np.uint8) 
    dark_mask = cv2.inRange(hsvFrame, dark_lower, dark_upper)

    return light_mask, light1_mask, light2_mask, dark_mask

def classify(mask):

    """
    Detect ripeness of cat chu, cat hoa loc mango based on area of each mask detected
        Arg: 
            mask: List of mask
        Return: Ripeness level
    """
    
    yel_area=cv2.countNonZero(mask[0])
    gr_area=cv2.countNonZero(mask[1])
    print(f"mask yel: {yel_area}, mask gr: {gr_area}")

    if yel_area!=0 and gr_area==0:
        return "Chin"
    elif yel_area==0 and gr_area!=0:
        return "Song"
    elif yel_area/gr_area>=5:
        return "Chin"
    elif gr_area/yel_area>=5:
        return "Song"
    elif yel_area/gr_area<5 and yel_area/gr_area>1:
        return "Chin 2"
    elif gr_area/yel_area<5 and gr_area/yel_area>=1:
        return "Chin 1"

def classify_catchuvang_ripeness(mask):

    """
    Detect ripeness of cat chu vang  mango based on area of each mask detected
        Arg: 
            mask: List of mask
        Return: Ripeness level
    """

    ur=cv2.countNonZero(mask[0])
    sr1=cv2.countNonZero(mask[1])
    sr2=cv2.countNonZero(mask[2])
    r=cv2.countNonZero(mask[3])
    array=[ur,sr1,sr2,r]
    if ur==max(array):
        return "Song"
    elif sr1==max(array):
        return "Chin 1"
    elif sr2==max(array):
        return "Chin 2"
    elif r==max(array):
        return "Chin"

def LCD_display(mango_type):

    lcdDisplay.set("Ket qua", 1)
    lcdDisplay.set(f"ur: {mango_type['ur']}    sr1: {mango_type['sr1']}",2)
    lcdDisplay.set(f"sr2: {mango_type['sr2']}   r: {mango_type['r']}",3)

def main(): 

    queue=[]
    a=None

    FLAG_POP=True
    FLAG_SENSOR_1=False
    FLAG_SENSOR_2=False
    FLAG_SENSOR_3=False
    cat_chu={'ur':0, 'sr1':0, 'sr2':0, 'r':0, 'type': 'catchu'}
    cat_hoaloc={'ur':0, 'sr1':0, 'sr2':0, 'r':0, 'type': 'cat_hoaloc'}
    cat_chu_vang={'ur':0, 'sr1':0, 'sr2':0, 'r':0, 'type': 'catchu_vang'}
    while True:

        LCD_display(cat_chu)
        if GPIO.input(sensor_camera)==0:

            im=picam2.capture_array()
            print("Da duoc chup")
            img_cut=mango_cut(im)     # Capture mango from image


            mask=detect_ripeness_by_hsv(img_cut)  # Color detection to detect ripeness of cat chu, cat hoa loc
            queue.append(classify(mask))

            # mask = catchu_vang_ripeness(img_cut)              # Color detection to detect ripeness of cat chu vang
            # queue.append(classify_catchuvang_ripeness(mask))
            time.sleep(4)

        # Classify by servoes on conveyor
        
        if FLAG_POP and len(queue)>0:
            a=queue.pop(0)
            
            print(f"Do chin: {a}")
            if a == "Chin":
                cat_chu['r']+=1
            elif a == "Chin 2":
                cat_chu['sr2']+=1
            elif a == "Chin 1":
                cat_chu['sr1']+=1
            elif a == "Song":
                cat_chu['ur']+=1

            FLAG_POP=False

        if a=="Chin":
            p1.ChangeDutyCycle(11)     #Close servo
        if GPIO.input(sensor_1)==0 and a =="Chin":
            a=None
            FLAG_SENSOR_1=True
        if FLAG_SENSOR_1 and GPIO.input(sensor_1):
            time.sleep(4)
            p1.ChangeDutyCycle(2.5) # Open servo
            FLAG_SENSOR_1=False
            FLAG_POP=True

        if a == "Chin 2":
            p2.ChangeDutyCycle(11)     #Close servo
        if GPIO.input(sensor_2)==0 and a =="Chin 2":
            a=None
            FLAG_SENSOR_2=True
        if FLAG_SENSOR_2 and GPIO.input(sensor_2):
            time.sleep(4)
            p2.ChangeDutyCycle(2.5) # Open servo
            FLAG_SENSOR_2=False
            FLAG_POP=True

        if a == "Chin 1":
            p3.ChangeDutyCycle(11)     #Close servo
            a=None
        if GPIO.input(sensor_3)==0:
            FLAG_SENSOR_3=True
        if FLAG_SENSOR_3 and GPIO.input(sensor_3):
            time.sleep(4)
            p3.ChangeDutyCycle(2.5) # Open servo
            FLAG_SENSOR_3=False
            FLAG_POP=True
            
# Using the special variable  
# __name__ 
if __name__=="__main__": 
    main() 

