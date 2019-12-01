'''
Program to make Baxter sort red and blue objects
Written by Tony Lin and Jack Noyes
'''

import rospy, cv2, cv_bridge
import numpy as np
import baxter_interface
import baxter_external_devices
from sensor_msgs.msg import Image
from baxter_pykdl import baxter_kinematics
from baxter_interface import CHECK_VERSION

bridge = cv_bridge.CvBridge()

def image_callback(ros_img):
    '''
    Convert the image to HSV and based on boundaries, create masks to detect all red and blue objects
    and then return the first red or blue object detected with a flag indicating the object color.
    '''
    
    # Convert the ROS image to an OpenCV image
    cv_image = bridge.imgmsg_to_cv2(ros_img, desired_encoding="passthrough")
    height = image.shape[0]
    width = image.shape[1]

    # Convert to an HSV image
    hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

    # Define HSV boundaries for red and blue
    red_hsv_boundaries = [([0,160,80], [5,255,255]),
                          ([170,160,80], [180,255,255])]
    blue_hsv_boundaries = ([100,180,100], [140,255,255])

    # Detect red and blue colors in the image and create masks
    red_mask = None
    for (lower, upper) in red_hsv_boundaries:
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")

        # Check if in range of boundaries
        if red_mask is None:
            red_mask = cv2.inRange(hsv_image, lower, upper)
        else:
            red_mask = cv2.bitwise_or(red_mask, cv2.inRange(hsv_image, lower, upper))

    blue_mask = None
    (lower, upper) = blue_hsv_boundaries
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")
    blue_mask = cv2.inRange(hsv_image, lower, upper)

    # Reduce noise
    kernel = np.ones((5,5),np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)

    # Get the center coordinates of the first object that we detected
    IS_RED = True
    mask = cv2.bitwise_or(red_mask, blue_mask)
    q = deque()
    visited = [[0 for i in range(width)] for j in range(height)]
    dr = [0, 1, 0, -1]
    dc = [1, 0, -1, 0]
    for r in range(height):
        for c in range(width):
            # print("{}, {}\n".format(r, c))
            if mask[r, c] > 0 and visited[r][c] == 0:
                if red_mask[r, c] > 0:
                    IS_RED = True
                else:
                    IS_RED = False
                # Initialize boundaries
                left = width
                right = 0
                top = height
                down = 0
                # BFS through the entire object
                q.appendleft((r, c))
                visited[r][c] = 1
                while q:
                    (cur_row, cur_col) = q.pop()
                    # Update boundaries
                    top = min(top, cur_row)
                    down = max(down, cur_row)
                    left = min(left, cur_col)
                    right = max(right, cur_col)
                    # Add neighbors
                    for i in range(4):
                        if (0 <= cur_row+dr[i] and cur_row+dr[i] < height
                            and 0 <= cur_col+dc[i] and cur_col+dc[i] < width
                            and mask[cur_row+dr[i], cur_col+dc[i]] > 0
                            and visited[cur_row+dr[i]][cur_col+dc[i]] == 0):
                            q.appendleft((cur_row+dr[i], cur_col+dc[i]))
                            visited[cur_row+dr[i]][cur_col+dc[i]] = 1
                # Return center
                center_r = (top+down)/2
                center_c = (left+right)/2
                # print("{}, {}\n".format(center_r, center_c))
                return center_r, center_c, IS_RED

    # cv2.imwrite('test_image.jpg', cv_image)
    # cv2.imshow('Image', cv_image)
    # cv2.waitKey(60000)
    print('Did not find any red or blue objects')
    return None

def movePos():
    left = baxter_interface.Limb('left')
    cur=left.endpoint_pose()
    print(cur)
    
    #pos=left.joint_angles()
    #j='left_w0'
    #pos[j]=pos[j]-0.25
    #left.move_to_joint_positions(pos)
    

def goToPos(goal,pixels):
    left = baxter_interface.Limb('left')
    myJacobian = baxter_kinematics('left')
    speed=0.025
    curPose=left.endpoint_pose()
    cur=[curPose['position'].x,curPose['position'].y,curPose['position'].z]
    if pixels:
        factor=[0.001,0.0009]
        goal[0]=-(goal[0]*factor[0]) + cur[0] + (200*factor[0])
        goal[1]=-(goal[1]*factor[1]) + cur[1] + (320*factor[1])
    
    print("\nGoal Position:{}\n".format(goal))
    print("Starting Position:{}\n".format(cur))
    while(abs(goal[0]-cur[0])>0.005 or abs(goal[1]-cur[1])>0.005 or abs(goal[2]-cur[2])>0.005):
        # print("{}, {}, {}\n".format(goal[0]-cur[0], goal[1]-cur[1], goal[2]-cur[2]))
        direc=np.array([goal[0]-cur[0],goal[1]-cur[1],goal[2]-cur[2]])
        norm=np.linalg.norm(direc)
        vel=direc*speed/norm
        vel=np.append(vel,[0,0,0])
        # finds Jacobian with current state
        J = myJacobian.jacobian()

        # perform related rates
        # finds pseudo-inverse with current state
        J_inv = myJacobian.jacobian_pseudo_inverse()
        joint_angles = np.dot(J_inv, vel)

        # Actuate baxter
        velocities = left.joint_velocities()

        velocities['left_s0'] = joint_angles[0, 0]
        velocities['left_s1'] = joint_angles[0, 1]
        velocities['left_e0'] = joint_angles[0, 2]
        velocities['left_e1'] = joint_angles[0, 3]
        velocities['left_w0'] = joint_angles[0, 4]
        velocities['left_w1'] = joint_angles[0, 5]
        velocities['left_w2'] = joint_angles[0, 6]

        left.set_joint_velocities(velocities)
        
        curPose=left.endpoint_pose()
        cur=[curPose['position'].x,curPose['position'].y,curPose['position'].z]
    print("End Position:{}".format(cur))


def main():
    print("\nInitializing node... ")
    rospy.init_node("moveTest_Node", anonymous=True)
    print("\nGetting robot state... ")
    rs = baxter_interface.RobotEnable(CHECK_VERSION)
    init_state = rs.state().enabled
    
    def clean_shutdown():
        print("\nExiting example...")
        if not init_state:
            print("Disabling robot...")
            rs.disable()
            cv2.destroyAllWindows()
    rospy.on_shutdown(clean_shutdown)

    print("\nEnabling robot... ")
    rs.enable()
    left = baxter_interface.Limb('left')
    ang=left.joint_angles()
    ang['left_w2']=-0.75
    left.move_to_joint_positions(ang)
    
    print("\nDetect...")
    img = rospy.wait_for_message('/cameras/left_hand_camera/image', Image)
    goal_r, goal_c = image_callback(img)
    leftg=baxter_interface.Gripper('left')
    leftg.calibrate()
    leftg.open()
    
    print("\nMove...")
    goToPos([goal_r, goal_c, 0.1],1)
    print("\nSecond Detect...")
    img = rospy.wait_for_message('/cameras/left_hand_camera/image', Image)
    goal_r, goal_c = image_callback(img)
    print("\nMove...")
    goToPos([goal_r, goal_c, -0.11],1)
    leftg.close()
    
    print("\nPlace...")
    goToPos([0.5, 0.5, 0],0)
    leftg.open()
    goToPos([0.5, 0.2, 0.1],0)
    # rospy.Subscriber('/cameras/left_hand_camera/image', Image, image_callback)
    #rospy.spin()

    #moveForward()
    #movePos()
    
#    done = False
#    while not done and not rospy.is_shutdown():
#        c = baxter_external_devices.getch()
#        if c:
#            # catch Esc or ctrl-c
#            if c in ['\x1b', '\x03']:
#                done = True
    rospy.signal_shutdown("Example finished.")
    print("Done.")


if __name__ == '__main__':
    main()
