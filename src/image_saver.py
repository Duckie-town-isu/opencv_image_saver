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
import zipfile
import time

from subprocess import CompletedProcess
from pathlib import Path
from datetime import datetime
from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage


class ImageSaver:
    def __init__(self, use_for_data_gathering=False, store_location=f"{Path.home()}/duckietown_dataset",
                 robot_name="duck7", location="Ames", tag="TRIAL") -> None:
        # get image from ROS
        self.img_subscriber = rospy.Subscriber(f"/{robot_name}/camera_node/image/compressed", CompressedImage,
                                               self.image_handler)

        self.save_dir_path = store_location
        self.csv_file = None
        self.create_save_dir(store_location)
        self.image_counter = 0
        self.use_for_dataset = use_for_data_gathering
        self.bridge = CvBridge()
        self.save_flag = False

        self.location = location
        self.tag = tag

        print("Initialization complete")
        print("To save an image, enter 3 numbers with spaces in between and press enter")
        print("The first is the number of Ducks, Duckiebots, then cones")

    def create_save_dir(self, dir_path):
        given_path = Path(dir_path)

        if not given_path.exists():
            os.mkdir(given_path)

        if not Path(f"{self.save_dir_path}/annotations_{datetime.now().strftime('%d-%m-%Y')}.csv").exists():
            csv_file = open(f"{self.save_dir_path}/annotations_{datetime.now().strftime('%d-%m-%Y')}.csv", 'w')
            FIELDS = ["imgID", "imgLocation", "timestamp", "tag", "numDucks", "duckJSON", "numRobots", "robotJSON",
                      "numCones", "coneJSON"]
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

    def image_handler(self, data: CompressedImage):
        np_arr = numpy.frombuffer(data.data, numpy.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if self.save_flag:
            self.save_flag = False
            self.save_image(cv_image)

    def write_to_csv(self, imageID, imageLocation, timestamp, tag):
        # FIELDS = ["imgID", "imgLocation", "timestamp", "tag", "numDucks", "duckJSON", "numRobots", "robotJSON", "numCones", "coneJSON"]

        numDucks, numRobots, numCones, retJSON = self.read_write_anntn()
        for i in range(int(numDucks)):
            img_data = [imageID, imageLocation, timestamp, tag, numDucks, retJSON["duckie"][i], "", "", "", ""]
            self.csv_file.writerow(img_data)
        for i in range(int(numRobots)):
            img_data = [imageID, imageLocation, timestamp, tag, "", "", numRobots, retJSON["duckiebot"][i], "", ""]
            self.csv_file.writerow(img_data)
        for i in range(int(numCones)):
            img_data = [imageID, imageLocation, timestamp, tag, "", "", "", "", numCones, retJSON["cone"][i]]
            self.csv_file.writerow(img_data)

    def user_io(self, annotation_list):
        """gets a list of annotations from CVAT output file and parses them to put in the CSV

        Args:
            annotation_list (list): list of annotations from CVAT
        """
        numDucks = input("How many DUCKS did you annotate in the image")
        numRobots = input("How many ROBOTS did you annotate in the image")
        numCones = input("How many cones did you annotate in the image")

    def run_cvat(task_name, path_to_img):

        img_dir = Path(path_to_img)

        temp_dir = Path(f"{img_dir.parent}/temp")
        if not temp_dir.exists():
            os.mkdir(str(temp_dir))

        if not img_dir.exists():
            raise FileNotFoundError(f"Image file {path_to_img} does not exist")

        labels = '[{"name":"duckiebot"},{"name":"duckie"},{"name":"cone"}]'
        std_cmd = ['cvat-cli', '--auth', 'duckie:quackquack', '--server-host', 'localhost', '--server-port', '8080']
        create_cmd = ['create', f"{task_name}", "--labels", labels, "local", f"{path_to_img}"]

        starter_comp_proc = subprocess.run(std_cmd + create_cmd, capture_output=True)
        # starter_comp_proc = CompletedProcess()
        string = starter_comp_proc.stdout.decode()
        index_id = starter_comp_proc.stdout.decode().find("task ID: ")
        next_index_id = starter_comp_proc.stdout.decode().find(" ", index_id + 9)
        cvat_task_id = int(string[index_id + 9:next_index_id])

        print("sleeping")
        x = input("hit space and enter when done annotating and you have hit save")

        # cvat_task_id = 100
        dump_cmd = ['dump', '--format', '"COCO 1.0"', f"{cvat_task_id}", f"{temp_dir}/output.zip"]
        anotation_proc = subprocess.run(std_cmd + dump_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # check = subprocess.check_output(std_cmd + dump_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(anotation_proc.stdout.decode())
        print(anotation_proc.stderr.decode())
        # # print(check.stdout.decode())
        # # print(check.stderr.decode())

    def read_write_anntn(temp_dir: Path):
        """       
        #unzip this file
        #read file
        #parse json
        #get bbox
        #ask user to type distance
        #send to csv

        Returns:
            _type_: _description_
        """

        with zipfile.ZipFile(f"{temp_dir}/output.zip", 'r') as zip_ref:
            zip_ref.extractall(f"{temp_dir}")
            zip_ref.close()

        with open(f"{temp_dir}/annotations/instances_default.json") as f:
            data = json.load(f)

        os.remove(str(f"{temp_dir}/output.zip"))
        print(data['images'])

        with open(f"annotations/instances_default.json") as f:
            data = json.load(f)
            annotation_list = data['annotations']
            retJSON = {"duckiebot": [], "duckie": [], "cone": []}
            for annotation in annotation_list:
                d_param_anntn = {}

                class_num = annotation['category_id']
                if class_num == 1:
                    d_param_anntn['class_name'] = "duckiebot"
                elif class_num == 2:
                    d_param_anntn['class_name'] = "duckie"
                elif class_num == 3:
                    d_param_anntn['class_name'] = "cone"
                else:
                    d_param_anntn['class_name'] = "INVALID"

                bbox = annotation['bbox']

                d_param_anntn['bbox'] = annotation['bbox']
                d_param_anntn['distance'] = input(
                    f"how far is the {d_param_anntn['class_name']} with x = ({bbox[0]}, {bbox[1]}) and l = {bbox[2]} and w = {bbox[3]} ")
                d_param_anntn['annotation_id'] = f"{d_param_anntn['class_name']}{annotation['id']}"

                retJSON[d_param_anntn['class_name']].append(d_param_anntn)

        return len(retJSON["duckie"]), len(retJSON["duckiebot"]), len(retJSON["cone"]), retJSON

    def save_image(self, image):
        time_str = datetime.now().strftime('%H-%M')
        if self.use_for_dataset:
            self.num_items = self.num_items.split(" ")
            file_name = f"{self.save_dir_path}/{self.num_items[0]}DK_{self.num_items[1]}RB_{self.num_items[0]}CN_{time_str}.jpg"
        else:
            file_name = f"{self.save_dir_path}/{time_str}-IM{self.image_counter}.jpg"
            self.write_to_csv(f"IM{self.image_counter}", file_name, time_str, self.tag)
        self.image_counter += 1
        cv2.imwrite(file_name, image)
        rospy.loginfo("Saved image succcessfully")


if __name__ == "__main__":
    rospy.init_node(name="image_saver_node")
    node = ImageSaver()
    node.run()
    rospy.spin()
