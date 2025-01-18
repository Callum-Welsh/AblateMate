import pygame
import sys
import pygame_gui
import numpy as np
from matplotlib.path import Path
from itertools import chain
import csv
import math
import cv2


#################
#Heat map plotting functions
#
def read_grid(file_name):
    return np.load(file_name)

def normalize_to_grayscale(arr):
    return arr/np.max(arr)*255

def grayscale_to_heatmap(grayscale_array):
    # Ensure the array is in the range [0, 255]
    grayscale_array = np.uint8(grayscale_array)  # Convert to 8-bit unsigned integer

    # Apply a color map to the grayscale array
    heatmap = cv2.applyColorMap(grayscale_array, cv2.COLORMAP_HOT)  # You can choose any color map (e.g., COLORMAP_HOT, COLORMAP_COOL, etc.)

    return heatmap

def distance(point1, point2):
    # Calculate the Euclidean distance between two points using only x and y values
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

def sort_by_distance(points):
    # Find the point closest to the origin
    origin = [0, 0] 
    closest_point = min(points, key=lambda point: distance(origin, point[:2]))
    sorted_points = [closest_point]

    # Sort the remaining points based on their distance to the closest_point
    while len(sorted_points) < len(points):
        closest_point = min((point for point in points if point not in sorted_points),
                            key=lambda point: distance(sorted_points[-1][:2], point[:2]))
        sorted_points.append(closest_point)

    return sorted_points



# Initialize Pygame
pygame.init()

# Set up display
width, height = 500, 800
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Circle Selection")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
CYAN = (0, 255, 255)

# Circle parameters
circle_radius = 240
circle_thickness = 10
circle_center = (width // 2, height // 2)

def convert_plot_for_CV(grid):
    colormap_arr = grayscale_to_heatmap(normalize_to_grayscale(grid))
    heatmap = cv2.resize(colormap_arr, (2*circle_radius,2*circle_radius), interpolation=cv2.INTER_NEAREST)
    return heatmap

# Selected points for polygon
selected_points = []
drawing_polygon = False

# Toggle for drawing polygon
draw_polygon = False

# GUI manager
manager = pygame_gui.UIManager((width, height))

# GUI elements
point_resolution_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, 10), (120, 30)),
                                                      text="Point Resolution:", manager=manager)
point_resolution_text_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((140, 10), (50, 30)),
                                                                 manager=manager)
point_resolution_text_entry.set_text("10")

fileName_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, 90), (120, 30)),
                                                      text="File Name:", manager=manager)
fileName_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((140, 90), (300, 30)),
                                                                 manager=manager)
fileName_entry.set_text("rasterPattern.csv")

time_per_point_label = pygame_gui.elements.UILabel(relative_rect=pygame.Rect((10, 50), (120, 30)),
                                                   text="Time Per Point:", manager=manager)
time_per_point_text_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((140, 50), (50, 30)),
                                                                manager=manager)
time_per_point_text_entry.set_text("30")

# Create a button to toggle drawing polygon
k_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((width - 130, 10), (120, 30)),
                                        text='Toggle Area', manager=manager)

# Create a button to print coordinates inside the polygon
print_coords_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((width - 230, 50), (200, 30)),
                                                   text='Save Points', manager=manager)

mot_counts = read_grid("MOTLog.npy")
shot_counts = read_grid("shotLog.npy")

mean_counts = mot_counts/shot_counts

mot_heatmap_cv =  cv2.cvtColor(convert_plot_for_CV(mean_counts),cv2.COLOR_BGR2RGB)
mot_heatmap_pg = pygame.image.frombuffer(mot_heatmap_cv.tostring(), mot_heatmap_cv.shape[1::-1], 'RGB')

shot_heatmap_cv =  cv2.cvtColor(convert_plot_for_CV(shot_counts),cv2.COLOR_BGR2RGB)
shot_heatmap_pg = pygame.image.frombuffer(shot_heatmap_cv.tostring(), shot_heatmap_cv.shape[1::-1], 'RGB')

mot_heatmap_mode = False
shot_heatmap_mode = False
# Main loop
running = True
while running:

    time_delta = pygame.time.Clock().tick(60) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Handle mouse events
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                # Get mouse position
                mouse_x, mouse_y = event.pos
                # Check if mouse click is inside the circle
                distance_from_center = ((mouse_x - circle_center[0]) ** 2 + (mouse_y - circle_center[1]) ** 2) ** 0.5
                if distance_from_center <= circle_radius+100 and mouse_y > height/2 - 270:
                    # Add clicked point to the list
                    selected_points.append((mouse_x, mouse_y))
                    drawing_polygon = True

        # Handle GUI events
        elif event.type == pygame.USEREVENT:
            if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == k_button:
                    draw_polygon = not draw_polygon  # Toggle drawing polygon
                    if draw_polygon and len(selected_points) >= 3:
                        # Fill the polygon with green points at Point_Resolution spacing
                        point_resolution = int(point_resolution_text_entry.get_text())
                elif event.ui_element == print_coords_button:
                    point_resolution = int(point_resolution_text_entry.get_text())
                    holdTime = int(time_per_point_text_entry.get_text())
                    if draw_polygon:
                        p=Path(selected_points)
                        coordList = []
                        for i in range(0, height, point_resolution):
                            for j in range(0, width, point_resolution):
                                coords = (j, i)
                                coordList.append(coords)

                        grid = p.contains_points(coordList)

                        coordArr = np.array(coordList)
                        rasterList = coordArr[grid].tolist()
                        for i in range(0, len(rasterList)):
                            rasterList[i][0] = np.round((rasterList[i][0]- width/2)/circle_radius,3)
                            rasterList[i][1] = -1*np.round((rasterList[i][1] - height/2)/circle_radius, 3)
                            rasterList[i].append(holdTime)
                        counter = 1
                        for i in range(0, len(rasterList)):
                                try:
                                    if rasterList[i][1] - rasterList[i+1][1] !=0:
                                        if counter ==1:
                                            counter = counter*-1
                                            for j in range(i+1, 1000):
                                                try:
                                                    if rasterList[j][1]- rasterList[j+1][1] != 0:
                                                        rasterList[i+1:j+1] = rasterList[i+1:j+1][::-1]

                                                    #print('Flipped')  
                                                        break
                                                except:
                                                    rasterList[i:-1] = rasterList[i:-1][::-1]
                                                #print('Flipped Last!')  
                                                    break
                                        else:
                                            counter = counter*-1
                                except:
                                    print("Done!")
                        rasterListWBoundary = rasterList + rasterList[::-1]

                        #print("Coordinates Inside Polygon:\n", rasterListWBoundary)
                    else:
                        sPoints = np.array(selected_points).tolist()
                        for i in sPoints:
                            i[0] = np.round((i[0]- width/2)/circle_radius,3)
                            i[1] = -1*np.round((i[1] - height/2)/circle_radius,3)
                            i.append(holdTime)
                        print("Unsorted:\n", sPoints)
                        rasterListWBoundary = sort_by_distance(sPoints)
                        print("Sorted: \n", rasterListWBoundary)

                    np.savetxt(fileName_entry.get_text(),
                        rasterListWBoundary,
                        delimiter =", ",
                        fmt ='% s')

                    


        # Handle keyboard events
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                selected_points = []  # Clear all selected points
            if event.key == pygame.K_h:
                if mot_heatmap_mode:
                    mot_heatmap_mode = False
                else:
                    shot_heatmap_mode = False
                    mot_heatmap_mode = True
            if event.key == pygame.K_j:
                if shot_heatmap_mode:
                    shot_heatmap_mode = False
                else:
                    mot_heatmap_mode = False
                    shot_heatmap_mode = True

        manager.process_events(event)

    manager.update(time_delta)

    # Clear the screen
    screen.fill(BLACK)
    # Draw the circle
    pygame.draw.circle(screen, WHITE, circle_center, circle_radius + circle_thickness)
    pygame.draw.circle(screen, (0, 0, 0, 0), circle_center, circle_radius)
    if mot_heatmap_mode:
        screen.blit(mot_heatmap_pg, (circle_center[0] - circle_radius, circle_center[1] - circle_radius))
    if shot_heatmap_mode:
        screen.blit(shot_heatmap_pg, (circle_center[0] - circle_radius, circle_center[1] - circle_radius))

    # Draw selected points
    for point in selected_points:
        pygame.draw.circle(screen, CYAN, point, 5)

    # Draw polygon if we have enough points and the draw_polygon toggle is active
    if len(selected_points) >= 2 and draw_polygon:
        pygame.draw.lines(screen, CYAN, True, selected_points, 2)

    # Draw GUI
    manager.draw_ui(screen)

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
sys.exit()
