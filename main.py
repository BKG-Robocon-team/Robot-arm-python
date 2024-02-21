import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import math


# Define parameters
L1 = 22  # Length of first segment
L2 = 22  # Length of second segment
L3 = 21  # Length of third segment
x0, y0 = 0, 21  # Base position
theta1 = np.pi / 4  # Theta1 angle
theta2 = np.pi / 4  # Theta2 angle
theta3 = np.pi * 3 / 2  # Fixed angle for third joint

# Constants for gravity
g = 9.81  # Gravity constant (m/s^2)
dt = 0.1  # Time step (s)

# Function to calculate forward kinematics
def forward_kinematics(theta1, theta2, theta3):
    x1 = L1 * np.cos(theta1) + x0
    y1 = L1 * np.sin(theta1) + y0
    x2 = x1 + L2 * np.cos(theta1 + theta2)
    y2 = y1 + L2 * np.sin(theta1 + theta2)
    x3 = x2 + L3 * np.cos( theta1 + theta2 + theta3)
    y3 = y2 + L3 * np.sin( theta1 + theta2 + theta3)
    return x1, y1, x2, y2, x3, y3

def inverse_kinematics(x3, y3, L1, L2, L3):
    theta1 = 0  # Fixed angle for third joint
    theta2 = 0  # Fixed angle for third joint
    theta3 = np.pi * 3 / 2  # Fixed angle for third joint
    isGetTheta = False
    for i in range(0,180):
        for j in range(0,360):
            theta1 = math.radians(i)
            theta2 = math.radians(j)
            x1, y1, x2, y2, x3, y3 = forward_kinematics(theta1,theta2,theta3)
            distance_to_ball = np.sqrt((x3 - ball_x )**2 + (y3 - ball_y )**2)
            if distance_to_ball< 2:
                print("1")
                isGetTheta = True
                break
        if isGetTheta:
            print("matched")
            break
    isGetTheta = False    

    return theta1, theta2, theta3

# Initialize plot
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)

plt.title('Three-Link Planar Robotic Arm Simulation | BK Galaxy')
plt.xlabel('X')
plt.ylabel('Y')
line, = ax.plot([], [], 'o-', lw=2)

# Calculate distance to the ball
ball_radius = 19 / 2
# Set the position of the ball
ball_x = 50
ball_y = ball_radius
        
# Plot the ball
circle = plt.Circle((ball_x, ball_y), ball_radius, color='red')

# Initial values for theta1 and theta2
theta1_init = np.pi / 2
theta2_init = 0

# Create sliders
axcolor = 'lightgoldenrodyellow'
ax_theta1 = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
ax_theta2 = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

s_theta1 = Slider(ax_theta1, 'Theta1', 0, 2*np.pi, valinit=theta1_init)
s_theta2 = Slider(ax_theta2, 'Theta2', 0, 2*np.pi, valinit=theta2_init)

# Create a button to drop the ball
ax_drop_button = plt.axes([0.81, 0.025, 0.1, 0.04], facecolor=axcolor)
button_drop = Button(ax_drop_button, 'Drop')# Create a button to drop the ball
ax_pick_button = plt.axes([0.25, 0.025, 0.1, 0.04], facecolor=axcolor)
button_pick = Button(ax_pick_button, 'Pick')


# Function to update plot
def update(val):
    global ball_x, ball_y, circle
    theta1_val = s_theta1.val
    theta2_val = s_theta2.val
    x1, y1, x2, y2, x3, y3 = forward_kinematics(theta1_val, theta2_val, theta3)
    line.set_data([x0, x1, x2, x3], [y0, y1, y2, y3])
    
    # Check if the end-effector (x3, y3) touches the ball
    distance_to_ball = np.sqrt((x3 - ball_x )**2 + (y3 - ball_y )**2)
    
    # Print distance to the ball
##    print("Distance to ball:", distance_to_ball)
    
    # If the end-effector is at the center of the ball, move the ball along with it
    if distance_to_ball <= ball_radius:
        ball_x = x3
        ball_y = y3 
        circle.center  = (ball_x, ball_y)

    # Plot the ball
    ax.add_artist(circle)
    
    return line, circle


# Function to handle dropping the ball
def drop_ball(event):
    global ball_x, ball_y
    ball_y = ball_radius
    circle.center  = (ball_x, ball_radius)
    # Plot the ball
    ax.add_artist(circle)
    update(None)

def pick_ball(event):
    global ball_x, ball_y
    theta1, theta2, theta3 = inverse_kinematics(ball_x, ball_y, L1, L2, L3)

    theta1_val = theta1
    theta2_val = theta2
    print(math.degrees(theta1), math.degrees(theta2))
    s_theta1.val = theta1
    s_theta2.val = theta2
    x1, y1, x2, y2, x3, y3 = forward_kinematics(theta1_val, theta2_val, theta3)
    line.set_data([x0, x1, x2, x3], [y0, y1, y2, y3])
    return line
        
# Set plot limits
ax.set_xlim(-80, 80)
ax.set_ylim(-10, 80)
ax.set_aspect('equal')
ax.grid(True)

update(None)
s_theta1.on_changed(update)
s_theta2.on_changed(update)

# Register the button's callback function
button_drop.on_clicked(drop_ball)
# Register the button's callback function
button_pick.on_clicked(pick_ball)

# Show plot
plt.show()
