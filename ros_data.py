from data_wrapper import data_wrapper
from PIL import Image as PIL_Image
import cv2
import rospy
from std_msgs.msg import Image as ROS_Image
from cv_bridge import CvBridge
from enum import Enum

## Inside ROS
# Cameras publishes to images/ ROS topic
# \/
## This code
# await roscore?
# connects to the image topic specified
# In the callback saves the latest frame in the "latest" variable as a PIL image.
# set "new image" flag true
# loop above replacing the previous frame as images are receieved
# call function basically needs to return new image, and sleep/loop until it can. sets "new_image" to false
# \/
## ML Code
# get image from ros_data (could incur delay)
# process
# loop

class ImgFormat(Enum):
    PIL = 1 # works for code as written
    CV2 = 2 # potentially faster

class ros_loader(data_wrapper):

    def callback(self, data):
        # rospy.loginfo(rospy.get_caller_id() + 'I heard %s', data)
        self.latest_image = data
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        if self.img_format == ImgFormat.PIL:
            self.latest_image = PIL_Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        else:
            self.latest_image = cv_image
        self.new_image = True

    def __init__(self):
        self.image_list = []
        self.new_image = False
        self.rate = rospy.Rate(10)
        self.bridge = CvBridge()
        self.img_format = ImgFormat.PIL
        image_topic = "/cameras/image"
        rospy.init_node('ros_ml_listener', anonymous=True)
        rospy.Subscriber(image_topic, ROS_Image, self.callback)
    
    def has_next(self):
        return not rospy.is_shutdown()

    def next(self):
        if not self.has_next():
            raise Exception("calling next when roscore is not running")            
        else:
            if self.new_image is False:
                self.rate.sleep() # sleep node to wait for image.
            else:
                yield [ self.latest_image, None ,  None ] # [ image, split layer, filename] # this class should only be sending the image
                self.new_image = False
                # [val, filename] = self.image_list.pop()
                # print(f"Testing \"{filename}\": # images left={len(self.image_list)}")            
                # for i in range(1,21): # hardcoded split layers for AlexNet - no full processing yet. Layer 0 would be full server, 21 would be full client
                #     yield [ val, i, filename ]