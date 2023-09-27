#! /usr/bin/python3

import os
import rospy
import cv2
import torch
import sys
import numpy


from pathlib import Path
from datetime import datetime
from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage

class ImageSaver():
    def __init__(self, use_for_data_gathering=True, store_location=f"{Path.home()}/duckietown_dataset", robot_name="duck7") -> None:
        # get image from ROS
        self.img_subscriber = rospy.Subscriber(f"/{robot_name}/camera_node/image/compressed", CompressedImage, self.image_handler)
        
        self.save_dir_path = store_location
        self.create_save_dir(store_location)
        self.image_counter = 0
        self.use_for_dataset = use_for_data_gathering
        self.bridge = CvBridge()
        self.save_flag = False
        print("Initialization complete")
        print("To save an image, enter 3 numbers with spaces in between and press enter")
        print("The first is the number of Ducks, Duckiebots, then cones")

    def create_save_dir(self, dir_path):
        given_path = Path(dir_path)
        if not given_path.exists():
            os.mkdir(given_path)

    def run(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            user_in = str(input())
            if ' ' in user_in:
                self.num_items = user_in
                self.save_flag = True
            rate.sleep()


    def image_handler(self, data:CompressedImage):
        np_arr = numpy.frombuffer(data.data, numpy.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if self.save_flag:
            self.save_flag = False
            self.save_image(cv_image)

    def save_image(self, image):
        if self.use_for_dataset:
            self.num_items = self.num_items.split(" ")
            file_name = f"{self.save_dir_path}/{self.num_items[0]}DK_{self.num_items[1]}RB_{self.num_items[0]}CN_{datetime.now().strftime('%H-%M')}.jpg"
        else:
            file_name = f"{self.save_dir_path}/{datetime.now().strftime('%d-%m-%Y_%H-%M')}-IM{self.image_counter}.jpg"
        self.image_counter += 1
        cv2.imwrite(file_name, image)
        rospy.loginfo("Saved image succcessfully")

if __name__ == "__main__":
    rospy.init_node(name="image_saver_node")
    node = ImageSaver()
    node.run()
    rospy.spin()
    