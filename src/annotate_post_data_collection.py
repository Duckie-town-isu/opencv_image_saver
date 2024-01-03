#! /usr/bin/python3
# This handles the annotation of images from a specific folder.

import os
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


class AnnotateImages:
    def __init__(self, img_dir_path, use_for_data_gathering=False, store_location=f"{Path.home()}/duckietown_dataset",
                 robot_name="duck7", location="Ames", tag="TRIAL") -> None:
        
        # get image from ROS
        # self.img_subscriber = rospy.Subscriber(f"/{robot_name}/camera_node/image/compressed", CompressedImage,
        #                                        self.image_handler)

        self.img_dir_path = img_dir_path
        self.annotation_save_path = store_location
        
        self.n, self.timestamp = self.parse_dir_names()
        assert self.n is not None

        if self.n:
            self.str_prefix_ann = "annotations" + str(self.n)
        else:
            self.str_prefix_ann = "annotations"
        self.images, self.image_files = self.load_images_from_folder() # Stores image filenames as an array in self.images
        
        self.temp_dir_path = None
        self.csv_writer = None
        self.csv_file = None
        self.create_save_dir(store_location)
        self.use_for_dataset = use_for_data_gathering

        self.robot_name = robot_name
        self.location = location
        self.tag = tag

        # print("Initialization complete")
        # print("To save an image, enter 3 numbers with spaces in between and press enter")
        # print("To save an image, enter 3 numbers with spaces in between and press enter")
        # print("The first is the number of Ducks, Duckiebots, then cones")

    def parse_dir_names(self):
        annotation_save_path = Path(self.annotation_save_path)
        if not annotation_save_path.exists():
            print(f"Save directory {annotation_save_path} does not exist", file=sys.stderr)
            sys.exit(1)
        
        if not annotation_save_path.is_absolute():
            annotation_save_path = annotation_save_path.absolute()
        self.annotation_save_path = annotation_save_path.absolute()
        
        img_dir_path = Path(self.img_dir_path)
        if not img_dir_path.exists():
            print(f"Image directory {self.img_dir_path} does not exist", file=sys.stderr)
            sys.exit(1)
        
        if not img_dir_path.is_absolute():
            img_dir_path = img_dir_path.absolute()
        batch_name, timestamp = img_dir_path.name.split('_')
        n = int(batch_name.replace('images','')) # getting the integer part of the batch_name
        self.img_dir_path = img_dir_path.absolute()
        return n, timestamp
    
    def load_images_from_folder(self):
        images = []
        image_files = []
        for filename in os.listdir(self.img_dir_path):
            img = cv2.imread(os.path.join(img_dir_path, filename))
            if img is not None:
                images.append(img)
                image_files.append(filename)
        return images, image_files

    def create_save_dir(self, dir_path):
        given_path = Path(dir_path)

        if not given_path.exists():
            os.mkdir(given_path)

        self.temp_dir_path = Path(f"{self.annotation_save_path}/temp")
        if not self.temp_dir_path.exists():
            os.mkdir(str(self.temp_dir_path))
            
        csv_file_name = f"{str(self.annotation_save_path)}/{self.str_prefix_ann}_{self.timestamp}.csv"

        if not Path(csv_file_name).exists():
            self.csv_file = open(csv_file_name, 'w')
            FIELDS = ["imgID", "imgLocation", "timestamp", "tag", "numDucks", "duckJSON", "numRobots", "robotJSON",
                      "numCones", "coneJSON"]
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(FIELDS)
        else:
            self.csv_file = open(csv_file_name, 'a')
            self.csv_writer = csv.writer(self.csv_file)

    def run(self):
        for image_file in self.image_files:
            self.annotate_image(image_file)
        self.csv_file.close()

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

        print(f"------------------- Running for image {img_dir.name} ------------------------")
        labels = '[{"name":"duckiebot"},{"name":"duckie"},{"name":"cone"}]'
        std_cmd = ['cvat-cli', '--auth', 'duckietown:QuackQuack1', '--server-host', 'localhost', '--server-port', '8080']
        create_cmd = ['create', f"{task_name}", "--labels", labels, "local", f"{path_to_img}"]

        starter_comp_proc = subprocess.run(std_cmd + create_cmd, capture_output=True)
        # starter_comp_proc = CompletedProcess()
        # print("Return Code:", starter_comp_proc.returncode)
        # print("stdout:", starter_comp_proc.stdout)
        # print("stderr:", starter_comp_proc.stderr)
        
        string = starter_comp_proc.stdout.decode()
        index_id = starter_comp_proc.stdout.decode().find("task ID: ")

        next_index_id = starter_comp_proc.stdout.decode().find(" ", index_id + 9)

        cvat_task_id = int(string[index_id + 9:next_index_id])

        _ = input("hit space and enter when done annotating and you have hit save")
        print("sleeping for 1 seconds because rest is important")
        time.sleep(1)

        # cvat_task_id = 76
        # dump_cmd = ['dump', '--format', '"COCO 1.0"', f"{cvat_task_id}", f"{self.temp_dir_path}/output.zip"]
        dump_cmd = f'cvat-cli --auth duckietown:QuackQuack1 --server-host localhost --server-port 8080 dump --format "COCO 1.0" {cvat_task_id} {self.temp_dir_path}/output.zip'
        anotation_proc = subprocess.run(dump_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

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
            # print(data["annotations"])

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

    def annotate_image(self, image_file):
        img_dir_path = Path(self.img_dir_path)
        file_relative = f"{img_dir_path.relative_to('/home/ranai/duckietown_dataset')}/{image_file}"
        self.run_cvat(f"{image_file}", f"{img_dir_path}/{image_file}")
        self.write_to_csv(f"{image_file}", file_relative, self.timestamp, self.tag)

if __name__ == "__main__":
    print("Enter img directory and then annotation save directory separated by a space")
    img_dir_path = sys.argv[1]
    # img_dir_path = "/home/ranai/duckietown_dataset/images_duckie/images1_15-12-2023"
    # annotation_dir = "/home/ranai/duckietown_dataset/images_duckie/images1_15-12-2023"
    annotation_dir = sys.argv[2]
    if not img_dir_path:
        print("Enter the correct image directory")
        assert img_dir_path is not None
    node = AnnotateImages(img_dir_path=img_dir_path, store_location=annotation_dir)
    node.run()