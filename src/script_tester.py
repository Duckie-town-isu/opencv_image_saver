#! /usr/bin/python3

import os
# import rospy
import cv2
import torch
import sys
import numpy
import csv
import json
import subprocess
import zipfile
import time
# import image_saver

from subprocess import CompletedProcess
from pathlib import Path
from datetime import datetime
# from cv_bridge import CvBridge
# from std_msgs.msg import String
# from sensor_msgs.msg import CompressedImage

csv_file = csv.writer(open(f"annotations_{datetime.now().strftime('%d-%m-%Y')}.csv", 'w'))

def write_to_csv(imageID, imageLocation, timestamp, tag):
    # FIELDS = ["imgID", "imgLocation", "timestamp", "tag", "numDucks", "duckJSON", "numRobots", "robotJSON", "numCones", "coneJSON"]
    numDucks, numRobots, numCones, retJSON = user_io()
    for i in range(int(numDucks)):
        img_data = [imageID, imageLocation, timestamp, tag, numDucks, retJSON["duckie"][i], "", "", "", ""]
        csv_file.writerow(img_data)
    for i in range(int(numRobots)):
        img_data = [imageID, imageLocation, timestamp, tag, "", "", numRobots, retJSON["duckiebot"][i], "", ""]
        csv_file.writerow(img_data)
    for i in range(int(numCones)):
        img_data = [imageID, imageLocation, timestamp, tag, "", "", "", "", numCones, retJSON["cone"][i]]
        csv_file.writerow(img_data)
        


def user_io():
    
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
            d_param_anntn['distance'] = input(f"how far is the {d_param_anntn['class_name']} with x = ({bbox[0]}, {bbox[1]}) and l = {bbox[2]} and w = {bbox[3]} ")
            d_param_anntn['annotation_id'] = f"{d_param_anntn['class_name']}{annotation['id']}"
            
            retJSON[d_param_anntn['class_name']].append(d_param_anntn)
    
    return len(retJSON["duckie"]), len(retJSON["duckiebot"]), len(retJSON["cone"]), retJSON
    
    

def run_cvat(task_name, path_to_img):

    img_dir = Path(path_to_img)

    temp_dir = Path(f"{img_dir.parent}/temp")
    if not temp_dir.exists():
        os.mkdir(str(temp_dir))

    if not img_dir.exists():
        raise FileNotFoundError(f"Image file {path_to_img} does not exist")

    labels = '[{"name":"duckiebot"},{"name":"duckie"},{"name":"cone"}]'
    std_cmd = ['cvat-cli', '--auth', 'duckie:quackquack', '--server-host', 'http://localhost', '--server-port', '8090']
    create_cmd = ['create', f"{task_name}", "--labels", labels, "local", f"{path_to_img}"]
    
    starter_comp_proc = subprocess.run(std_cmd + create_cmd, capture_output=True)
    print(starter_comp_proc.stderr.decode())
    string = starter_comp_proc.stdout.decode()
    print(string)
    index_id = starter_comp_proc.stdout.decode().find("task ID: ")
    print(index_id)
    next_index_id = starter_comp_proc.stdout.decode().find(" ", index_id + 9)
    cvat_task_id = int(string[index_id+10:next_index_id])

    print(f"Go annotate Task {cvat_task_id}. \n sleeping.....")
    x = input("hit space and enter when done annotating and you have hit save")
    print("Continuing")

    # cvat_task_id = 100
    dump_cmd = ['dump', '--format', '"COCO 1.0"', f"{cvat_task_id}", f"{temp_dir}/output.zip"]
    anotation_proc = subprocess.run(std_cmd + dump_cmd,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # check = subprocess.check_output(std_cmd + dump_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(anotation_proc.stdout.decode())
    print(anotation_proc.stderr.decode())
    # print(check.stdout.decode())
    # print(check.stderr.decode())

    #unzip this file
    #read file
    #parse json
    #get bbox
    #ask user to type distance
    #send to csv
    
    temp_dir = Path("Duckietown_COCO.zip")
    
    if not temp_dir.exists():
        raise FileNotFoundError()

    with zipfile.ZipFile(temp_dir, 'r') as zip_ref:
        zip_ref.extractall()
    zip_ref.close()
        
    # get the location of this directory
    
    write_to_csv("IM0", "Ames", "2023Oct16", "TRIAL")

    # os.remove(str(f"{temp_dir}/output.zip"))
    # print(data['images'])


if __name__ == "__main__":
    run_cvat("IM0", "/home/ranai/duckietown_dataset/15-39-IM0.jpg")
    # run_cvat("IM0", "/home/ranai/duckietown_dataset/0DK_0RB_0CN_18-42.jpg")