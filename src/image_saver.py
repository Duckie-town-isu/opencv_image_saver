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
        # self.img_subscriber = rospy.Subscriber(f"/{robot_name}/camera_node/image/compressed", CompressedImage,
        #                                        self.image_handler)

        self.save_dir_path = store_location
        self.temp_dir_path = None
        self.csv_writer = None
        self.csv_file = None
        self.create_save_dir(store_location)
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

    def create_save_dir(self, dir_path):
        given_path = Path(dir_path)

        if not given_path.exists():
            os.mkdir(given_path)

        self.temp_dir_path = Path(f"{self.save_dir_path}/temp")
        if not self.temp_dir_path.exists():
            os.mkdir(str(self.temp_dir_path))

        if not Path(f"{self.save_dir_path}/annotations_{datetime.now().strftime('%d-%m-%Y')}.csv").exists():
            self.csv_file = open(f"{self.save_dir_path}/annotations_{datetime.now().strftime('%d-%m-%Y')}.csv", 'w')
            FIELDS = ["imgID", "imgLocation", "timestamp", "tag", "numDucks", "duckJSON", "numRobots", "robotJSON",
                      "numCones", "coneJSON"]
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(FIELDS)
        else:
            self.csv_file = open(f"{self.save_dir_path}/annotations_{datetime.now().strftime('%d-%m-%Y')}.csv", 'a')
            self.csv_writer = csv.writer(self.csv_file)

    def run(self):

        user_in = str(input("Enter number when you are ready\n"))
        
        if len(user_in) > 0:
            if user_in == 'q':
                rospy.signal_shutdown()
                exit()
            self.num_items = user_in
            self.save_flag = True
            recvd_image = rospy.wait_for_message(f"/{self.robot_name}/camera_node/image/compressed", CompressedImage, timeout=None)
            self.image_handler(recvd_image)

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
            self.csv_writer.writerow(img_data)
        for i in range(int(numRobots)):
            img_data = [imageID, imageLocation, timestamp, tag, "", "", numRobots, retJSON["duckiebot"][i], "", ""]
            self.csv_writer.writerow(img_data)
        for i in range(int(numCones)):
            img_data = [imageID, imageLocation, timestamp, tag, "", "", "", "", numCones, retJSON["cone"][i]]
            self.csv_writer.writerow(img_data)

        self.csv_file.flush()

    # def user_io(self, annotation_list):
    #     """gets a list of annotations from CVAT output file and parses them to put in the CSV

    #     Args:
    #         annotation_list (list): list of annotations from CVAT
    #     """
    #     numDucks = input("How many DUCKS did you annotate in the image")
    #     numRobots = input("How many ROBOTS did you annotate in the image")
    #     numCones = input("How many cones did you annotate in the image")

    def run_cvat(self, task_name, path_to_img):

        img_dir = Path(path_to_img)

        if not self.temp_dir_path.exists():
            raise FileNotFoundError(f"Temp Path {self.temp_dir_path} does not exist")

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

        x = input("hit space and enter when done annotating and you have hit save")
        print("sleeping for 2 seconds because rest is important")
        time.sleep(2)

        # cvat_task_id = 76
        # dump_cmd = ['dump', '--format', '"COCO 1.0"', f"{cvat_task_id}", f"{self.temp_dir_path}/output.zip"]
        dump_cmd = f'cvat-cli --auth duckie:quackquack --server-host localhost --server-port 8080 dump --format "COCO 1.0" {cvat_task_id} {self.temp_dir_path}/output.zip'
        anotation_proc = subprocess.run(dump_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        # check = subprocess.check_output(std_cmd + dump_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(anotation_proc.args)
        print(anotation_proc.stdout.decode())
        print(anotation_proc.stderr.decode())
        # # print(check.stdout.decode())
        # # print(check.stderr.decode())

    def read_write_anntn(self):
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
        print("ZIP FILE IS EXTRACTING" +  f"{str(self.temp_dir_path)}/output.zip")
        with zipfile.ZipFile(f"{str(self.temp_dir_path)}/output.zip", 'r') as zip_ref:
            zip_ref.extractall(f"{self.temp_dir_path}")
            zip_ref.close()

        with open(f"{self.temp_dir_path}/annotations/instances_default.json") as f:
            data = json.load(f)
            print(data["annotations"])

        os.remove(str(f"{self.temp_dir_path}/output.zip"))

        with open(f"{self.temp_dir_path}/annotations/instances_default.json") as f:
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
        
        os.remove(str(f"{self.temp_dir_path}/annotations/instances_default.json"))
        os.rmdir(str(f"{self.temp_dir_path}/annotations"))
        return len(retJSON["duckie"]), len(retJSON["duckiebot"]), len(retJSON["cone"]), retJSON

    def save_image(self, image):
        time_str = datetime.now().strftime('%H-%M')

        if False and self.use_for_dataset:
            self.num_items = self.num_items.split(" ")
            file_name = f"{self.save_dir_path}/{self.num_items[0]}DK_{self.num_items[1]}RB_{self.num_items[0]}CN_{time_str}.jpg"
        else:
            file_name = f"{self.save_dir_path}/{time_str}-IM{self.image_counter}.jpg"
            cv2.imwrite(file_name, image)
            self.run_cvat(f"{time_str}-IM{self.image_counter}.jpg", f"{self.save_dir_path}/{time_str}-IM{self.image_counter}.jpg")
            self.write_to_csv(f"IM{self.image_counter}", file_name, time_str, self.tag)
        self.image_counter += 1
        rospy.loginfo("Saved image succcessfully")
        self.run()


if __name__ == "__main__":
    rospy.init_node(name="image_saver_node")
    node = ImageSaver()
    node.run()
    