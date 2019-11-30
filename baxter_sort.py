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
    # Convert the ROS image to an OpenCV image
    cv_image = bridge.imgmsg_to_cv2(ros_img, desired_encoding="passthrough")

    # Find center coordinates of a red object
    row_size = cv_image.shape[0]
    col_size = cv_image.shape[1]
    left = col_size
    right = 0
    top = row_size
    down = 0
    for r in range(1,row_size):
        for c in range(1,col_size):
            # Check if current pixel within RGB threshold values
            pixel_b, pixel_g, pixel_r, _ = cv_image[r, c]
            if pixel_b < 100 and pixel_g < 100 and pixel_r > 130:
                if r < top:
                    top = r
                if r > down:
                    down = r
                if c < left:
                    left = c
                if c > right:
                    right = c
    center_r = (top+down)/2
    center_c = (left+right)/2
    print("{}, {}\n".format(center_r, center_c))
    # cv2.imwrite('test_image.jpg', cv_image)
    # cv2.imshow('Image', cv_image)
    # cv2.waitKey(60000)
    return center_r, center_c

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