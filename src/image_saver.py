#! /usr/bin/python3

import os
import rospy
import cv2
import torch
import sys
import numpy
import csv
import json
import subprocess


from pathlib import Path
from datetime import datetime
from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage

class ImageSaver():
    def __init__(self, use_for_data_gathering=False, store_location=f"{Path.home()}/duckietown_dataset", robot_name="duck7", location="Ames") -> None:
        # get image from ROS
        self.img_subscriber = rospy.Subscriber(f"/{robot_name}/camera_node/image/compressed", CompressedImage, self.image_handler)
        
        self.save_dir_path = store_location
        self.csv_file = None
        self.create_save_dir(store_location)
        self.image_counter = 0
        self.use_for_dataset = use_for_data_gathering
        self.bridge = CvBridge()
        self.save_flag = False
        self.location = location
        print("Initialization complete")
        print("To save an image, enter 3 numbers with spaces in between and press enter")
        print("The first is the number of Ducks, Duckiebots, then cones")

    def create_save_dir(self, dir_path):
        given_path = Path(dir_path)
        if not given_path.exists():
            os.mkdir(given_path)
        if not Path(f"{self.save_dir_path}/annotations_{datetime.now().strftime('%d-%m-%Y')}.csv").exists():
            csv_file = open(f"{self.save_dir_path}/annotations_{datetime.now().strftime('%d-%m-%Y')}.csv", 'w')
            FIELDS = ["imgID", "imgLocation", "timestamp", "location", "numDucks", "duckJSON", "numRobots", "robotJSON", "numCones", "coneJSON"]
            self.csv_file = csv.writer(csv_file)
            self.csv_file.writerow(FIELDS)
        else:
            csv_file = open(f"{self.save_dir_path}/annotations_{datetime.now().strftime('%d-%m-%Y')}.csv", 'a')
            self.csv_file = csv.writer(csv_file)

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
            
    def write_to_csv(self, imageID, imageLocation, timestamp):
        # FIELDS = ["imgID", "imgLocation", "timestamp", "location", "numDucks", "duckJSON", "numRobots", "robotJSON", "numCones", "coneJSON"]
        img_data = [imageID, imageLocation, timestamp, self.location, "", "", "", "", "", ""]
        self.csv_file.writerow(img_data)

    def run_cvat(self, path_to_img):
        task_name = None
        labels = '[{"name":"duckiebot"},{"name":"duckie"},{"name":"cone"}]'
        std_cmd = ['cvat-cli', '--auth', 'duckie:quackquack', '--server-host', 'localhost', '--server-port', '8080']
        create_cmd = ['create', f"{task_name}", "--labels", labels, "local", f"{path_to_img}"]
        cvat_task_id = -1
        os.mkdir(f"{path_to_img}/temp")
        #TODO capture output of create command
        dump_cmd = ['dump', 'format', '"COCO 1.0"', f'{cvat_task_id}', f"{path_to_img}/temp/output.zip"]
        starter_comp_proc = subprocess.run(std_cmd.append(create_cmd), capture_output=True)

        

    def save_image(self, image):
        if self.use_for_dataset:
            self.num_items = self.num_items.split(" ")
            time_str = datetime.now().strftime('%H-%M')
            file_name = f"{self.save_dir_path}/{self.num_items[0]}DK_{self.num_items[1]}RB_{self.num_items[0]}CN_{time_str}.jpg"
        else:
            file_name = f"{self.save_dir_path}/{datetime.now().strftime('%d-%m-%Y_%H-%M')}-IM{self.image_counter}.jpg"
            self.write_to_csv(f"IM{self.image_counter}", file_name, time_str)
        self.image_counter += 1
        cv2.imwrite(file_name, image)
        rospy.loginfo("Saved image succcessfully")

if __name__ == "__main__":
    rospy.init_node(name="image_saver_node")
    node = ImageSaver()
    node.run()
    rospy.spin()
    