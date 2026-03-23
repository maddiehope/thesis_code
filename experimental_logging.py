'''
LIVER PHANTOM EXPERIMENTAL DATA LOGGING CODE
'''

import numpy as np
import pandas as pd
import threading
import cv2
import queue
import pyvisa
import serial
import time
from datetime import datetime

# GLOBAL VARIABLES ##############################################

# CV Vars
frame_width = 640            # width of CV window
frame_height = 480           # height of CV window
target_x = 215               # target biopsy x-position (currently placeholder value)
target_y = 86               # target biopsy y-position (currently placeholder value)
pixel_error_x = 0            # signed x-position error, initial val 0
pixel_error_y = 0            # signed y-position error, initial val 0
magnitude_error = 0          # total magnitude error, initial val 0
mm_to_pixel = 0.635          # 0.635 mm/px, each mm has 1.5748 pixels

# CV Queue Initialization
'''
WORKER THREAD: CV processing, writes results into a queue, updates global error variables
MAIN THREAD : reads frames from the queue and calls cv2.imshow / cv2.waitKey
'''
raw_queue = queue.Queue(maxsize=1)        # for raw image frames, small size to keep latency low
processed_queue = queue.Queue(maxsize=1)  # for processed image frames, small size to keep latency low
stop_event = threading.Event()

# PM101 Vars
pm_power = 0.0
pm_lock = threading.Lock()
pm_stop_event = threading.Event()
pm_results = []  # keep a separate log 

# Arduino Breathing Vars
breath_time = 0.0
breath_percent = 0.0
breath_pwm = 0.0
arduino_lock = threading.Lock()

# Results Logging
results = []
i = 0 # counter that tracks # of logs

#################################################################
# CV FUNCTION ###################################################
def CV():
    global frame_height, frame_width, target_x, target_y, pixel_error_x, pixel_error_y, magnitude_error
    global raw_queue, processed_queue
    """
    WORKER THREAD: consumes frames from frame_queue,
                   does green tracking, updates global error variables,
                   and draws annotations on the frame before putting it back.
    """

    print("CV Worker: starting...")

    # Bounding 
    '''
    These are the upper and lower bounds for the specific color that will be tracked.
    Array format: [R, G, B]

    Color HSV range calculator: http://www.tydac.ch/color/
    '''
    green_lower = np.array([36, 25, 100], np.uint8)
    green_upper = np.array([70, 255, 255], np.uint8)
    kernel = np.ones((5, 5), "uint8") # kernal array of ones

    # CV Analyzation Main Loop
    while not stop_event.is_set():

        # trying to pull most recent frame from where it is saved in the main loop
        try:
            frame = raw_queue.get(timeout=0.1)
            # print("CV Worked: frame grab successful.") # DEBUGGING

        except queue.Empty:
            #print("CV Worker: queue is empty.") # DEBUGGING
            continue

        if frame is None:
            print("CV Worker: frame is None:", frame is None, ")")  # DEBUGGING
            continue # no frame read 

        # Color Initialization
        ''' 
        Converting the image frame from "BGR" (blue/green/red, default)
                                    to "HSV" (hue/saturation/value).

        HSV is preferred for color detection becuase the "hue" channel directly 
        represents color, which makes it easier to define color ranges. 
        '''
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # MASKING / MORPHOLOGICAL OPERATIONS
        green_mask = cv2.inRange(hsv_frame, green_lower, green_upper)
        green_mask = cv2.dilate(green_mask, kernel)
        res_green = cv2.bitwise_and(frame, frame, mask = green_mask)

        # Detection
        contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        '''
        Multiple green contours may be detected per frame- this can happen due to effects of shading, lighting, etc. 
        We only want to track the ONE INTENDED green contour. 

        The contours must be collected and compared (given they are present), 
        with the largest being held as the "primary" or "main" contour so that the position tracking does not get confused
        and jump around with ANY small green contour that is present in the frame. 
        '''
        if len(contours) != 0: # ONLY run if there are any green contours present
        
            areas = [] # init list to collect all contour areas for comparison

            for contour in contours:        
                area = cv2.contourArea(contour) # obtain area of each detected contour 
                areas.append(area) 

            main_area = max(areas) # find the max area from the running list
            area_index = areas.index(main_area) # find the index of said max area

            contours_list = list(contours) # tuples (default dtype) cannot be indexed as lists, so convert to list
            contour = contours_list[area_index] # grab the contour corresponding to the max area index

            if main_area > 15: # only recognize the area if it reaches threshold size 
                          # (prevents logging of insiginificant, unmeaningful detections like reflections)

                # given we've found the ONE DESIRED contour and it meets threshold size, extract positioning
                x, y, w, h = cv2.boundingRect(contour) 

                '''
                TOP LEFT-hand corner of the CV frame holds the position X=0, Y=0

                Variables 'x' & 'y' are the TOP LEFT-hand coordinates for the bounding rectangle. 
                Therefore (x+w = length of rect.) going (left -> right) and (y+h = height of rect.) going (top -> bottom). 

                Bottom right-hand corner is the tracking point (probe enters window from bottom left-hand side of screen).
                '''
                bottom_right_x = x + w
                bottom_right_y = y + h

                # DEBUGGING
                #print(f"Center: {center_x}, {center_y}") # printing x, y center coordinates in terminal for visualization

                '''
                CALCULATING PIXEL DISPLACEMENT
                Displacement or "error" is stored as a global variable and will be documented within a .csv backlog.
                '''
                # signed error calculations
                pixel_error_x = bottom_right_x - target_x
                pixel_error_y = bottom_right_y - target_y

                # magnitude of error calculation
                # Based on pythagorean theorem: distance = sqrt(x^2 + y^2)
                magnitude_error = np.sqrt(pixel_error_x**2 + pixel_error_y**2)

                log_res() # save the error results 

                frame = cv2.rectangle(frame, (x, y),(x + w, y + h),(0, 255, 0), 2)
                #cv2.putText(frame, "Green", (x, y), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0),2)

                #print(f"Green detected: {frame}") # DEBUGGING

        # Pushing to Queue
        '''
        The program will try to push the latest frame to the queue. 
        If sucessful, the older frame will be overwritten with the latest processed frame. 
        '''
        try:
            if processed_queue.full():
                processed_queue.get_nowait()
                #print("Queue is full, WAITING...") # DEBUGGING

            processed_queue.put_nowait(frame)
            #print("Pushing newest frame...") # DEBUGGING

        except queue.Full:
            #print("Queue is full, PASSING...") # DEBUGGING
            pass

    print("CV Worker: exiting...")
    
#################################################################
# PM101 FUNCTION ################################################
def PM101():
    """
    WORKER THREAD: continuously reads power from PM101 over USB.
    Stores latest reading in global variable 'pm_power' for logging.
    """
    global pm_power, pm_lock, pm_results

    print("PM101 Worker: starting...")

    try:
        # Initialize VISA
        rm = pyvisa.ResourceManager('@py')
        addr = 'USB0::4883::32886::M00667359::0::INSTR'  # replace with your actual PM101 USB address
        pm = rm.open_resource(addr)
        pm.timeout = 5000  # ms

        # Identify device
        idn = pm.query("*IDN?")
        print("PM101 connected:", idn)

    except Exception as e:
        print("PM101 connection failed:", e)
        return

    try:
        while not pm_stop_event.is_set():
            try:
                # Query power in Watts
                reading = pm.query("MEAS:POW?")
                power_val = float(reading)

                with pm_lock:
                    pm_power = power_val
                    pm_results.append({'timestamp': time.time(), 'Power_W': power_val})

            except Exception as e:
                print("PM101 read error:", e)
                time.sleep(0.1)
                continue

            time.sleep(0.1)  # adjust polling rate as needed

    except KeyboardInterrupt:
        print("PM101 Worker: stopped by user.")

    finally:
        pm.close()
        print("PM101 Worker: disconnected.")
#################################################################
# ARDUINO FUNCTION ##############################################
def ARDUINO():
    global breath_time, breath_percent, breath_pwm
    
    print("Arduino Worker: starting...")

    try:
        ser = serial.Serial('/dev/cu.usbmodem1101', 115200, timeout=1)
        time.sleep(2)  # allow Arduino reset

    except Exception as e:
        print("Arduino connection failed:", e)
        return

    while not stop_event.is_set():

        try:
            line = ser.readline().decode().strip()
            if not line:
                continue

            parts = line.split(',')

            if len(parts) != 3:
                continue

            t, percent, pwm = parts

            with arduino_lock:
                breath_time = float(t)
                breath_percent = float(percent)
                breath_pwm = float(pwm)

        except:
            continue

    ser.close()
    print("Arduino Worker: exiting...")

#################################################################
# LOGGING FUNCTION ##############################################
def log_res():
    global pixel_error_x, pixel_error_y, magnitude_error
    global breath_time, breath_percent, breath_pwm
    global results, i

    # Transforming pixels to measurement
    mm_error_x = pixel_error_x * mm_to_pixel
    mm_error_y = pixel_error_y * mm_to_pixel
    mm_error_magnitude = magnitude_error * mm_to_pixel

    # Grabbing lung data from Arduino
    with arduino_lock:
        current_breath_time = breath_time
        current_breath_percent = breath_percent
        current_breath_pwm = breath_pwm

    # Grab power data from PM101
    with pm_lock:
        current_power = pm_power

    i += 1
    
    # Append all values to results list
    results.append({
                        'Count': i,
                        'timestamp': time.time(),
                        'Pixel_X': pixel_error_x,
                        'Pixel_Y': pixel_error_y,
                        'Magnitude': magnitude_error,
                        'MM_X': mm_error_x,
                        'MM_Y': mm_error_y,
                        'MM_Magnitude': mm_error_magnitude,
                        'Breath_Time': current_breath_time,
                        'Breath_Percent': current_breath_percent,
                        'Breath_PWM': current_breath_pwm,
                        'Power_W': current_power
                    })
    
#################################################################
# HELPER PIXEL LOCATOR FUNCTION #################################

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked Pixel Location -> X: {x}, Y: {y}")

#################################################################
# MAIN FUNCTION ##################################################

def main():

    print("Main: starting...")

    # CV INITIALIZATION
    # (main thread owns the camera)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    cv2.namedWindow("Tracking")
    cv2.setMouseCallback("Tracking", mouse_callback)

    # DEBUGGING 
    '''
    print("Main: cap.isOpened() ->", cap.isOpened())
    if not cap.isOpened():
        print("Main: camera not opened.")
        return
    '''

    # STARING THREADS
    t1 = threading.Thread(target=CV, daemon=True)
    t1.start()
    t2 = threading.Thread(target=ARDUINO, daemon=True)
    t2.start()
    t3 = threading.Thread(target=PM101, daemon=True)
    t3.start()

    #   HANDLING CAMERA
    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("Main: cap.read() failed (ret:", ret, ", frame is None:", frame is None, ")")
            break

        # Push the latest raw frame to worker
        try:
            if raw_queue.full():
                raw_queue.get_nowait()
            raw_queue.put_nowait(frame.copy())

        except queue.Full:
            pass

        # For display, show the frame coming back from the queue if available,
        # or just show the raw frame.
        try:
            display_frame = processed_queue.get_nowait()
        except queue.Empty:
            display_frame = frame

        cv2.imshow("Tracking", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stop_event.set()
    pm_stop_event.set()
    t1.join()
    t2.join()
    t3.join()
    cap.release()

    # SAVE RESULTS TO CSV
    if results:  # only if there is data/results

        df = pd.DataFrame(results)

        # Generate filename with current date & time
        now = datetime.now()
        timestamp_str = now.strftime("%m_%d_%H-%M")  # Month_Day_Hour-Min
        filename = f"/Users/maddiehope/Desktop/WAVE/GUI/LOGS/Phase 2/PL_log_{timestamp_str}.csv"
        
        df.to_csv(filename, index=False)
        print(f"Saved {len(results)} rows to {filename}")
        
    else:
        print("No data to save")

    cv2.destroyAllWindows()

#################################################################
if __name__ == "__main__":
    main()
