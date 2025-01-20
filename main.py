import time
from pathlib import Path
import csv
import os
import numpy as np
import cv2
#import matplotlib.pyplot as plt
import zmq
from piezo import Piezo
from fit import fit_coords
import tkinter as tk
from tkinter import simpledialog

################ Constants ###########
WINDOW_NAME = 'AblateMate (Press Q to exit, ? for command list)'
PIEZO_SCALE = 30 # steps per pixel (40 nominal)
ROI_RADIUS = 40 # Radius in pixels for centroid fit
RESOLUTION = 16 # Resolution of the grid on the ablation target
######################################


################ Init Variables
logFlag = 0 # initialize the flag of whether to log a new shot as zero
## Init random walker
np.random.seed(int(time.time())) #initialize the 
r1 = np.random.rand()
threshold = 190000 #MOT Count threshold for what is a good/bad spot. I.e if MOT counts are below threshold for 2 shots, move
warning_number = 2
do_move = False
help_mode = True

##############################################
# CALIBRATION COEFFICIENTS
# Reported position in pixels when ablation laser is at each corner of target.
left = (239, 159.9)
right = (185.8, 162)
top = (215, 125)
bottom = (200.4, 199.4)
##############################################

############ Where to log the MOT Counts and Shot Log ###########
MOTLogFile = "MOTLog.npy"
ShotLogFile = "shotLog.npy"
heat_map_mode = False

##################################################################
#################### Settings window ############################
def open_settings_window():
    def save_settings():
        nonlocal settings
        # Collect values from the entry boxes
        settings = [entry.get() for entry in entries]
        print("Settings saved!")
        print(settings)
        root.quit()  # Close the settings window

    settings = []
    root = tk.Tk()
    root.title("Settings")

    # Example settings labels
    settings_labels = ["Low Threshold", "# of Shots Below Threshold Allowed", "MOT Log Save Filepath", "Shot Log Save Filepath"]
    defaults = [threshold, warning_number, MOTLogFile, ShotLogFile]
    # Create a list to hold entry widgets
    entries = []

    for i in range(0,len(settings_labels)):
        frame = tk.Frame(root)
        frame.pack(padx=10, pady=5)

        label = tk.Label(frame, text=settings_labels[i])
        label.pack(side=tk.LEFT)

        entry = tk.Entry(frame)
        entry.pack(side=tk.LEFT)
        entries.append(entry)
        entry.insert(0, defaults[i])

    # Add a Save button
    save_button = tk.Button(root, text="Save", command=save_settings)
    save_button.pack(pady=10)

    root.mainloop()
    root.destroy()
    return settings


#######################################################
#### Heat map functions
#######################################################
## Grid for heat map and shot logging:
def initialize_grid(file_name, height, resolution):
    if os.path.exists(file_name):
        # Load existing grid
        grid = np.load(file_name)
    else:
        # Initialize a new grid with zeros
        grid = np.ones((int(height/resolution), int(height/resolution)), dtype=int)
    return grid

def read_grid(file_name):
    return np.load(file_name, dtype=int)

def update_grid(grid, x, y, value):
    # Add the specified value to the grid point (x, y)
    grid[x, y] += value

def save_grid(filename, grid):
    # Save the grid to a .npy file
    np.save(filename, grid)

def normalize_to_grayscale(arr):
    return arr/np.max(arr)*255

def grayscale_to_heatmap(grayscale_array):
    # Ensure the array is in the range [0, 255]
    grayscale_array = np.uint8(grayscale_array)  # Convert to 8-bit unsigned integer

    # Apply a color map to the grayscale array
    heatmap = cv2.applyColorMap(grayscale_array, cv2.COLORMAP_HOT)  # You can choose any color map (e.g., COLORMAP_HOT, COLORMAP_COOL, etc.)

    return heatmap

def convert_plot_for_CV(grid):
    # Step 1: Create a sample Matplotlib plot
    colormap_arr = grayscale_to_heatmap(normalize_to_grayscale(grid))
    heatmap = cv2.resize(colormap_arr, (400,400), interpolation=cv2.INTER_NEAREST)

    # Resize the plot image to match the size of the foreground content
    #height, width = 400, 400  # Example size, adjust as needed
    #plot_img = cv2.resize(plot_img, (width, height))
    return heatmap
#####################################################

####################################################
### ZMQ Socket for recieving MOT Count data
port = '55555'
context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.connect("tcp://localhost:48766")
####################################################


######## Load the MOT grid from the log file###############
mot_grid = initialize_grid(MOTLogFile, 400, RESOLUTION)
shot_grid = initialize_grid(ShotLogFile, 400, RESOLUTION)

###########################################################
# Get the initial coordinates of the target space
mat, offset = fit_coords(left, right, top, bottom)
print(mat, offset)
###########################################################

#########################################
########### Start the video capture
vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FOCUS, 0)
vid.set(cv2.CAP_PROP_EXPOSURE, -5)
############################################


####################### Function to draw where target is
def draw_cross(img, x, y, color):
    cv2.line(img, (x-20, y), (x+20, y), color, 3)
    cv2.line(img, (x, y-20), (x, y+20), color, 3)

#####################################
# Piezo control initialization
target_x, target_y = None, None # Set the current position to none
dest_x, dest_y = None, None # set the desired positin to none
last_move = time.monotonic()

target_h = 400 # size of the target in pixels
hover_x = None # Where is the curson
hover_y = None

#Initialize MOT Logging
MOTCounts = 0
logCount = 0
move_flag =0

autoMode = False
warning = False

# Intantiate piezo
piezo = Piezo()
piezo.set_v(1, 1000)
piezo.set_v(2, 1260)

def handleMouse(e, x, y, flags, param):
    global dest_x, dest_y, csv_x, csv_y, csv_dur, hover_x, hover_y
    if e == cv2.EVENT_LBUTTONDOWN and target_x is not None and x < target_h:
        dest_x = x
        dest_y = y
        print('Destination:', dest_x, dest_y)

        csv_x = None
        csv_y = None
        csv_dur = None

    if e == cv2.EVENT_MOUSEMOVE:
        hover_x = x
        hover_y = y

    


# CSV loading
csv_x = None
csv_y = None
csv_dur = None
csv_idx = 0
last_csv = time.monotonic()



# Smoothing
spot_x_queue = []
spot_y_queue = []

plot_img = convert_plot_for_CV(mot_grid) # Load the heatmap

counter = 0 # crude counter to do some timing

########### Begin main loop

while True:
    counter = counter + 1 # Increment the counter
    #start = time.time() #Timing function for benchmarking
    
    ret, frame = vid.read() #Get a new video frame from the camera
    frame = np.array(frame, dtype=np.uint8)

    # Process image
    cropped = frame[140:340, 160:350].copy() # Crop to ROI
    h, w, _ = cropped.shape
    cropped = cv2.resize(cropped, (2*w, 2*h), interpolation=cv2.INTER_CUBIC)
    red = cropped[:,:,2] # Extract red channel

    # Extract location
    x_marginals = red.sum(axis=0)
    y_marginals = red.sum(axis=1)

    spot_x = np.argmax(x_marginals)
    spot_y = np.argmax(y_marginals)

    # Check total intensity
    intensity = red.sum()
    valid = intensity > 200e3

    if valid:
        # Compute correction from centroid fit
        d = np.arange(-ROI_RADIUS, ROI_RADIUS)
        dx = np.average(d, weights=x_marginals[spot_x-ROI_RADIUS : spot_x+ROI_RADIUS])
        dy = np.average(d, weights=y_marginals[spot_y-ROI_RADIUS : spot_y+ROI_RADIUS])
        spot_x += dx
        spot_y += dy

        # Smoothing via moving average
        spot_x_queue.append(spot_x)
        spot_y_queue.append(spot_y)
        spot_x_queue = spot_x_queue[-5:]
        spot_y_queue = spot_y_queue[-5:]
        spot_x = np.mean(spot_x_queue)
        spot_y = np.mean(spot_y_queue)

        # Visually indicate location
        draw_cross(cropped, round(spot_x), round(spot_y), (0, 0, 255))
        cv2.circle(cropped, (round(spot_x), round(spot_y)), ROI_RADIUS, (0, 0, 255), 2)

    # Convert to target coords
    target_coords = np.linalg.solve(mat, np.array([spot_x, spot_y]) - offset)

    # Display information
    h, w, _ = cropped.shape
    target_h = h
    info_display = np.zeros((h, h, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.circle(info_display, (h//2, h//2), h//2, (255, 255, 255), 2)
    cv2.putText(info_display, 'Target', (h//2-70, h//2+20), font, 1.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Visually indicate location
    if valid:
        target_x = round(h/2 + target_coords[0] * h/2)
        target_y = round(h/2 - target_coords[1] * h/2)
        draw_cross(info_display, target_x, target_y, (0, 255, 0))
    else:
        target_x, target_y = None, None

    # Draw target queue from CSV
    if csv_x is not None and not autoMode:
        autoMode = False
        for xx, yy in zip(csv_x, csv_y):
            cv2.circle(info_display, (round(xx), round(yy)), 1, (255, 0, 255), 2)
        dt = csv_dur[csv_idx] - (time.monotonic() - last_csv)
        cv2.putText(info_display, f'{dt:.2f} s', (10, 30), font, 0.6, (255, 0, 255), 1, cv2.LINE_AA)
    elif csv_x is not None:
        for xx, yy in zip(csv_x, csv_y):
            cv2.circle(info_display, (round(xx), round(yy)), 1, (255, 0, 255), 2)
        dt = csv_dur[csv_idx] - (time.monotonic() - last_csv)
        #cv2.putText(info_display, f'{dt:.2f} s', (10, 30), font, 0.6, (255, 0, 255), 1, cv2.LINE_AA)
        
    if help_mode:
        cv2.putText(cropped, "q: Quit", (10, 110), font, 1, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(cropped, "s: Open Settings", (10, 145), font, 1, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(cropped, "esc: Disable piezo", (10, 180), font, 1, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(cropped, "z: Enable Auto Mode", (10, 250), font, 1, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(cropped, "h: Toggle Heat Map", (10, 285), font, 1, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(cropped, "l: Load Raster CSV", (10, 315), font, 1, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(cropped, "?: Toggle Instructions", (10, 215), font, 1, (255, 0, 255), 1, cv2.LINE_AA)

    # Auto mode
    if autoMode:
        cv2.putText(cropped, "AutoConnor Active", (10, 350), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
        if newShot: #If there is new mot data
            if logFlag > 1: # If you have been on the spot for more than one shot
                if MOTCounts < threshold: # If the counts are below threshold
                    if warning < warning_number: # If there is no warning flag
                        warning +=1 # Set the warning flag to true
                    elif csv_x is None: #Since the warning flag was true and you were below threshold, move the spot
                #The below block generates a new coordinate, and checks that the coordinate is inside the allowed region. If it is not, it tries again until an allowed coordinate is found
                        validation_bool = False
                        while not validation_bool: 
                            np.random.seed(int(time.time()))
                            r2 = np.random.rand()
                            if r2 > 0.5:
                                np.random.seed(int(time.time()))
                                r1 = np.random.rand()
                            #print(r)
                            dest_x = target_x + 16*np.cos(2*np.pi*r1)
                            dest_y = target_y + 16*np.sin(2*np.pi*r1)
                            if 0 < dest_x and dest_x < h and 0 < dest_y and dest_y <h:
                                validation_bool = True
                    else:
                        do_move = True
                else:
                    warning = 0 # If you are above threshold, turn the warning off
            else:
                warning = 0 #If the spot has moved, turn the warning off



    # Draw current destination
    if dest_x is not None:
        draw_cross(info_display, round(dest_x), round(dest_y), (255, 0, 0))
        cv2.line(info_display, (target_x, target_y), (round(dest_x), round(dest_y)), (255, 0, 0), 1)

    cv2.putText(cropped, f'X: {spot_x:.1f}', (5, 40), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(cropped, f'Y: {spot_y:.1f}', (w//2, 40), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    
    newShot = False

# Check the socket for a new message, indicating a new shot
    try:
        message = socket.recv_string(flags=zmq.NOBLOCK)
        #print(f"Received message: {message}")
        MOTCounts = int(float(message))
        #print(MOTCounts)
        newShot = True
    except zmq.Again:
        # No message was available to be received
        #print("No message received.")
        newShot = False
    except:
        continue

    #Display the current MOT counts
    cv2.putText(cropped, 'MOT Counts: {}'.format(MOTCounts), (5, 75), font, 1, (255, 255, 255), 1, cv2.LINE_AA)


    # Show the user's curson
    if hover_x is not None:
        hx = 2*hover_x/h-1
        hy = 2*(1-hover_y/h)-1
        cv2.putText(info_display, f'Mouse X: {hx:.2f}', (h//2-100, h-80), font, 0.7, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(info_display, f'Mouse Y: {hy:.2f}', (h//2-100, h-50), font, 0.7, (180, 180, 180), 1, cv2.LINE_AA)


    # Display information
    if heat_map_mode == False:
        display = np.hstack([info_display, cropped]) #No heat map, don't show the heat map
    else:
        combined = cv2.addWeighted(info_display, 0.8, plot_img, 1, 0) #Yes heat map, show the heat map
        display = np.hstack([combined, cropped])

    dh, dw, _ = display.shape
    cv2.imshow(WINDOW_NAME, display)
    cv2.setMouseCallback(WINDOW_NAME, handleMouse)

##############################################
#Handle User key input
    key = cv2.waitKey(1) & 0xFF

    #If the user presses q, save the MOT Grid to a file and gracefully exit
    if key == ord('q'):
        np.save('old/frame.npy', frame)
        save_grid(MOTLogFile, mot_grid)
        save_grid(ShotLogFile, shot_grid)
        break

    #If the user presses l, load a raster path from a file
    elif key == ord('l'):
        autoMode = False #Don't allow auto mode and raster mode at the same time
        file = input('Load from file (xx.csv): ')
        path = Path(file)
        if not path.exists():
            print(f'Could not find {file}!')
        else:
            try:
                lines = []
                with open(path, 'r') as f:
                    for line in csv.reader(f):
                        lines.append(list(map(float, line)))
                csv_x, csv_y, csv_dur = np.array(lines).T
                csv_x = np.round(h/2 + csv_x * h/2) #Turn coordinates in [-1,1] to [0, h]
                csv_y = np.round(h/2 - csv_y * h/2)
                print(csv_x, csv_y, csv_dur) # Reads coordinates (x, y, t) from file

                # Load first point
                csv_idx = 0
                dest_x = csv_x[0]
                dest_y = csv_y[0]
            except Exception as e:
                print(f'Error reading {file}:', e)

    elif key == ord('n'):
        if autoMode == False:
            last_csv = 0 #If you push n, skip to the next point in the csv file
        else:
            do_move = True # If automode is on, allow manual override to next spot with n key
    
    #If you push z, activate auto mode
    elif key ==ord('z'):
        if autoMode == True:
            autoMode = False
        else:
            autoMode = True
    
    #If you push h, activate heat map mode
    elif key ==ord('h'):
        if heat_map_mode == True:
            heat_map_mode = False
        else:
            heat_map_mode = True
    

    #If you push s, open the settings window
    elif key ==ord('s'):
        returned_settings = open_settings_window()
        threshold = int(returned_settings[0])
        warning_number = int(returned_settings[1])
        MOTLogFile = str(returned_settings[2])
        ShotLogFile = str(returned_settings[3])

    elif key == ord('?'):
        if help_mode == False:
            help_mode = True
        else:
            help_mode = False
    


    #If you push escape, disable any destination locking, autoMove, rastering, etc.
    elif key == 27: # Esc
        dest_x = None
        dest_y = None
        csv_x = None
        csv_y = None
        csv_dur = None
        autoMode = False

##############################################################
    # Handle piezo movement

    #If the piezo target is set...
    if (
        dest_x is not None
        and target_x is not None
        and time.monotonic() - last_move > 0.1
    ):
        dx = dest_x - target_x
        dy = dest_y - target_y
        dx = round(-dx*PIEZO_SCALE)
        dy = round(dy*PIEZO_SCALE)

        if max(abs(dx), abs(dy)) > 250: # If you are more than 250 piezo units away from your target, intantiate a move
            logFlag = 0 #New move means turn of logging

            #If y > x, move x first; else, move y first
            if abs(dy) > abs(dx):\
                piezo.move_by(2, dy)
            else:
                piezo.move_by(1, dx)
            last_move = time.monotonic()

        # The spot is within ragne of destination, so no need to move
        else:
            if newShot: #If there was a new shot...
                logFlag = logFlag + 1 #increase the log flag by one
                # Log the position and mot counts of the shot to the heatmap grid
                x_grid_pos = int(target_y/RESOLUTION)
                y_grid_pos = int(target_x/RESOLUTION)
                update_grid(mot_grid,x_grid_pos, y_grid_pos, MOTCounts)
                update_grid(shot_grid,x_grid_pos, y_grid_pos, 1)
                #If heat map mode is on, update the plot
                if heat_map_mode:
                    plot_img = convert_plot_for_CV(mot_grid/shot_grid)

#Else, if the piezo target isn't set
    elif target_x is not None:
        #If there was a new shot, log where the target is
        if newShot:
                logFlag = logFlag + 1 #increase the log flag by one
                # Log the position and mot counts of the shot to the heatmap grid
                x_grid_pos = int(target_y/RESOLUTION)
                y_grid_pos = int(target_x/RESOLUTION)
                update_grid(mot_grid,x_grid_pos, y_grid_pos, MOTCounts)
                update_grid(shot_grid,x_grid_pos, y_grid_pos, 1)
                #If heat map mode is on, update the plot
                if heat_map_mode:
                    plot_img = convert_plot_for_CV(mot_grid/shot_grid)

    # Update target from CSV
    if csv_x is not None and time.monotonic() - last_csv > csv_dur[csv_idx] and not autoMode:
        csv_idx = (csv_idx + 1) % len(csv_x)
        dest_x = csv_x[csv_idx]
        dest_y = csv_y[csv_idx]
        last_csv = time.monotonic()
    elif csv_x is not None and autoMode and do_move == True:
        csv_idx = (csv_idx + 1) % len(csv_x)
        dest_x = csv_x[csv_idx]
        dest_y = csv_y[csv_idx]
        last_csv = time.monotonic()
        do_move = False

    if counter == 10000:
        counter = 0
        save_grid(MOTLogFile, mot_grid)
        save_grid(ShotLogFile, shot_grid)

    #end = time.time() #timing for benchmarking
    #print(start-end)

#Gracefully terminate the program upopn breaking from the running loop
vid.release()
cv2.destroyAllWindows()
