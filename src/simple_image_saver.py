#! /usr/bin/python3
# Very similar to image_saver.py but separating the annotation task to handle it after data collection.
# Thsi script saves images to a specified folder, and annotate_post_data_collection.py handles the annotation of images from a specific folder
# To help with annotations later, make sure that each folder is consistent (ex: only duckies, only 2 duckies, only 3 duckies in a specific distance range, 1 duckie and 1 duckiebot but in a specific distance range each)
# Be sure to include a .txt file afterwards with details of how far the distance range of the annotated objects is.
import os
import rospy
import cv2
import torch
import sys
import numpy
import csv
import json
import subprocess
import zipfile
import time

from subprocess import CompletedProcess
from pathlib import Path
from datetime import datetime
from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage


class SimpleImageSaver:
    def __init__(self, n = 100, use_for_data_gathering=False, store_location=f"{Path.home()}/duckietown_dataset",
                 robot_name="duck7", location="Ames", tag="TRIAL") -> None:
        # n: The nth batch of images collected in a particular day.
        # get image from ROS
        # self.img_subscriber = rospy.Subscriber(f"/{robot_name}/camera_node/image/compressed", CompressedImage,
        #                                        self.image_handler)
        self.n = n
        if self.n:
            self.str_prefix_ann = "annotations"+str(self.n)
            self.str_prefix_im = "images"+str(self.n)
        else:
            self.str_prefix_ann = "annotations"
            self.str_prefix_im = "images"

        self.save_dir_path = store_location
        self.image_counter = 0
        self.use_for_dataset = use_for_data_gathering
        self.bridge = CvBridge()
        self.save_flag = False

        self.robot_name = robot_name
        self.location = location
        self.tag = tag

        print("Initialization complete")
        print("To save an image, enter 3 numbers with spaces in between and press enter")
        print("The first is the number of Ducks, Duckiebots, then cones")


    def run(self):

        user_in = str(input("Enter the number of duckies, duckiebots, and cones in frame when you are ready\n"))
        
        if len(user_in) > 0:
            if user_in == 'q':
                self.csv_file.close()
                rospy.signal_shutdown()
                exit()
            self.num_items = user_in
            self.save_flag = True
            print("Getting the image")
            recvd_image = rospy.wait_for_message(f"/{self.robot_name}/camera_node/image/compressed", CompressedImage, timeout=None)
            self.image_handler(recvd_image)

    def image_handler(self, data: CompressedImage):
        np_arr = numpy.frombuffer(data.data, numpy.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if self.save_flag:
            self.save_flag = False
            self.save_image(cv_image)

    
    def save_image(self, image):
        time_str = datetime.now().strftime('%H-%M')

        if self.n:
            str_prefix = "images"+str(self.n)
        else:
            str_prefix = "images"

        image_dir_path = Path(f"{self.save_dir_path}/{self.str_prefix_im}_{datetime.now().strftime('%d-%m-%Y')}")
        if not image_dir_path.exists():
            os.mkdir(str(image_dir_path))

        if False and self.use_for_dataset:
            self.num_items = self.num_items.split(" ")
            file_name = f"{self.save_dir_path}/{self.num_items[0]}DK_{self.num_items[1]}RB_{self.num_items[0]}CN_{time_str}.jpg"
        else:
            file_name = f"{image_dir_path }/{time_str}-IM{self.image_counter}.jpg"
            cv2.imwrite(file_name, image)
        self.image_counter += 1
        rospy.loginfo("Saved image succcessfully")
        self.run()


if __name__ == "__main__":
    rospy.init_node(name="image_saver_node")
    # Enter the correct batch number "n"
    node = SimpleImageSaver(n=37)
    node.run()