#! /usr/bin/python

import os
import cv2
import csv
import json
import torch
import numpy
import seaborn


from pathlib import Path
from matplotlib import pyplot as plt
from typing import List

############ PARAMS
user = "autosys"                # One of "ranai", "apurva", "autosys"
LOWER_DIST_THRESH = -1          # Distance threshold for matching bounding boxes. Set to -1 to disable and UPPER_DIST_THRESH to numpy.inf
UPPER_DIST_THRESH = numpy.inf   # Distance threshold for matching bounding boxes. Set to numpy.inf to disable and LOWER_DIST_THRESH to -1
DO_PROPOSITION_LABELLED = True  # If True, make prop labelled confusion matrix
GEN_PERCENTAGE = False          # If True, generate percentage confusion matrix. If False, generate absolute confusion matrix
YOLO_CONF_THRESHOLD = 0.4       # Confidence threshold for trained model
IOU_THRESHOLD = 0.6             # IOU threshold for matching bounding boxes
REMOVE_CONES  = True            # If True, remove cones from confusion matrix
DO_VALIDATION = True            # If True, run validation
############ END PARAMS

home = str(Path.home())
if user == "ranai":
    FOLDER_PATH = f"{home}/duckietown_dataset/test"
    YOLOv5_PATH = f"{home}/software/yolov5"
elif user == "apurva":
    FOLDER_PATH = f"{home}/duckietown_dataset/test"
    YOLOv5_PATH = f"{home}/software/yolov5"
    
elif user == "autosys":
    FOLDER_PATH = f"{home}/duckietown_dataset/test"
    YOLOv5_PATH = f"{home}/software/yolov5"
    trained_model = torch.hub.load(YOLOv5_PATH, 'custom',
                                   path=f"{YOLOv5_PATH}/trained_models/best.pt", source='local',
                                   force_reload=True)
    trained_model.conf = YOLO_CONF_THRESHOLD
else:
    raise ValueError("Invalid user")

#  duckiebot, duckie, cone, empty
confusion_matrix_freq = [
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]
                        ]

#  duckiebot, duckie, cone, duckie_duckiebot, duckie_cone, duckiebot_cone, all, empty
propn_labelled_confusion_matrix = [
                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                  ]

DUCKIEBOT = 0
DUCKIE = 1
CONE = 2
EMPTY = 3
DUCKIE_DUCKIEBOT = 3
DUCKIE_CONE = 4
DUCKIEBOT_CONE = 5
ALL = 6
PROPN_EMPTY = 7


def bb_intersection_over_union(boxA, boxB):
    ### From https://gist.github.com/meyerjo

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

# Read the CSV file
def get_pytorch_bbox(img_path, trained_model):
    home = str(Path.home())
    # trained_model = torch.hub.load(home + '/Ranai/Research/yolov5', 'custom',
    #                                path=home + '/Ranai/Research/yolov5/duckietown_weights/best.pt', source='local',
    #                                force_reload=True)    
    cv_img = cv2.imread(img_path)
    im_ann = cv_img.copy()

    results = trained_model(cv_img)
    pred_list = results.xyxy[0].cpu().numpy()

    counter = {"duckie": 0, "duckiebot": 0, "cone": 0}

    predictions = {"duckiebot": [], "duckie": [], "cone": []}

    if len(pred_list) > 0:
        for pred in pred_list:
            topleft = (int(pred[0]), int(pred[1]))
            botright = (int(pred[2]), int(pred[3]))
            conf = pred[4]
            classes = results.names
            obj_class = classes[int(pred[5])]
            if obj_class == "duckiebot": class_id = DUCKIEBOT
            elif obj_class == "duckie": class_id = DUCKIE
            elif obj_class == "cone": class_id = CONE
            predictions[obj_class].append({"tag": f"{obj_class}{counter[obj_class]}", "class": f"{obj_class}", "class_id": class_id,
                                           "bbox": [topleft[0], topleft[1], botright[0], botright[1]]})
            
            if DO_VALIDATION:
                im_ann = cv2.rectangle(im_ann, topleft, botright, color=(255, 0, 0), thickness=4)
                im_ann = cv2.putText(im_ann, f"{obj_class}", topleft, color=(255, 0, 0), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75)

    return predictions, im_ann

# getting things from the CSV file
def get_ground_truth_bbox(csv_path, img_name):
    
    ground_truth_json = {"duckiebot": [], "duckie": [], "cone": []}
    with open(csv_path, "r") as file:
        csv_reader = csv.reader(file)
        for line in csv_reader:
            if line[0] == img_name:
                location = line[1]
                timestamp = line[2]
                tag = line[3]
                numDucks = line[4]
                if len(numDucks) != 0:
                    duckJSON = json.loads(line[5].replace("'", '"'))
                    duckJSON["location"] = location
                    duckJSON["timestamp"] = timestamp
                    duckJSON["tag"] = tag
                    duckJSON["class_id"] = DUCKIE
                    duckJSON["class"] = "duckie"
                    x,y,w,h = duckJSON["bbox"]
                    duckJSON["bbox"] = [x,y, x+w, y+h]
                    ground_truth_json["duckie"].append(duckJSON)
                numRobots = line[6]                
                if len(numRobots) != 0:
                    robotJSON = json.loads(line[7].replace("'", '"'))
                    robotJSON["location"] = location
                    robotJSON["timestamp"] = timestamp
                    robotJSON["tag"] = tag
                    robotJSON["class_id"] = DUCKIEBOT
                    robotJSON["class"] = "duckiebot"
                    x,y,w,h = robotJSON["bbox"]
                    robotJSON["bbox"] = [x,y, x+w, y+h]
                    ground_truth_json["duckiebot"].append(robotJSON)
                numCones = line[8]
                if len(numCones) != 0:
                    coneJSON = json.loads(line[9].replace("'", '"'))
                    coneJSON["location"] = location
                    coneJSON["timestamp"] = timestamp
                    coneJSON["tag"] = tag
                    coneJSON["class_id"] = CONE
                    coneJSON["class"] = "cone"
                    x,y,w,h = coneJSON["bbox"]
                    coneJSON["bbox"] = [x,y, x+w, y+h]
                    ground_truth_json["cone"].append(coneJSON)

    return ground_truth_json

def calculate(img_path:str, annotation_path, lower_dist_threshold, upper_dist_threshold):
    preds, _ = get_pytorch_bbox(img_path, trained_model)
    img_name = img_path[img_path.rfind("/")+1:]
    truth = get_ground_truth_bbox(annotation_path, img_name)

    max_IOU = 0
    best_match_pred = None
    best_match_anntn = None
    gt_propositions = [False] * 4
    pred_propositions = [False] * 4

    pred_list = preds["duckiebot"] + preds["duckie"] + preds["cone"]
    true_list = truth["duckiebot"] + truth["duckie"] + truth["cone"]
    matched_list = []

    for true in true_list:
        if float(true["distance"]) > lower_dist_threshold and float(true["distance"]) < upper_dist_threshold:
            
            if len(true_list) != 0:
                if true["class_id"] == DUCKIE: gt_propositions[DUCKIE] = True
                if true["class_id"] == DUCKIEBOT: gt_propositions[DUCKIEBOT] = True
                if true["class_id"] == CONE: gt_propositions[CONE] = True
            
            if len(pred_list) == 0:
                confusion_matrix_freq[EMPTY][true["class_id"]] += 1
                pred_propositions[EMPTY] = 0
                continue
            
            for pred in pred_list:
                if pred["class"] != true["class_name"]:
                    continue
                pred_bbox = pred["bbox"]
                true_bbox = true["bbox"]

                curr = bb_intersection_over_union(pred_bbox, true_bbox)
                if curr > max_IOU:
                    max_IOU = curr
                    best_match_pred = pred
                    best_match_anntn = true

            if max_IOU < 0.60:
                confusion_matrix_freq[EMPTY][true["class_id"]] += 1
                pred_propositions[EMPTY] = True
            else:
                pred_propositions[best_match_pred["class_id"]] = True
                preds[best_match_pred["class"]].remove(best_match_pred)
                truth[best_match_anntn["class_name"]].remove(best_match_anntn)
                matched_list.append(best_match_pred)

                confusion_matrix_freq[best_match_pred["class_id"]][best_match_anntn["class_id"]] += 1

            max_IOU = 0
            
    for pred in pred_list:
        if len(pred_list) and (pred not in matched_list):
            confusion_matrix_freq[pred["class_id"]][EMPTY] += 1
            pred_propositions[best_match_pred["class_id"]] = True
            
    populate_proposition_confusion_matrix(gt_propositions, pred_propositions)

def populate_proposition_confusion_matrix(gt_propositions:List[bool], pred_propositions:List[bool]):
    # Belongs in programming horror but IDK a better way just yet
    
    if not gt_propositions[DUCKIE] and not gt_propositions[DUCKIEBOT] and not gt_propositions[CONE]: # GT Empty
        if not pred_propositions[DUCKIE] and not pred_propositions[DUCKIEBOT] and not pred_propositions[CONE]:
            propn_labelled_confusion_matrix[PROPN_EMPTY][PROPN_EMPTY] += 1
        elif not pred_propositions[DUCKIE] and not pred_propositions[DUCKIEBOT] and pred_propositions[CONE]:
            propn_labelled_confusion_matrix[CONE][PROPN_EMPTY] += 1
        elif not pred_propositions[DUCKIE] and pred_propositions[DUCKIEBOT] and not pred_propositions[CONE]: 
            propn_labelled_confusion_matrix[DUCKIEBOT][PROPN_EMPTY] += 1
        elif not pred_propositions[DUCKIE] and pred_propositions[DUCKIEBOT] and pred_propositions[CONE]:
            propn_labelled_confusion_matrix[DUCKIEBOT_CONE][PROPN_EMPTY] += 1
        elif pred_propositions[DUCKIE] and not pred_propositions[DUCKIEBOT] and not pred_propositions[CONE]: 
            propn_labelled_confusion_matrix[DUCKIE][PROPN_EMPTY] += 1
        elif pred_propositions[DUCKIE] and not pred_propositions[DUCKIEBOT] and pred_propositions[CONE]: 
            propn_labelled_confusion_matrix[DUCKIE_CONE][PROPN_EMPTY] += 1
        elif pred_propositions[DUCKIE] and pred_propositions[DUCKIEBOT] and not pred_propositions[CONE]: 
            propn_labelled_confusion_matrix[DUCKIE_DUCKIEBOT][PROPN_EMPTY] += 1
        elif pred_propositions[DUCKIE] and pred_propositions[DUCKIEBOT] and pred_propositions[CONE]: 
            propn_labelled_confusion_matrix[ALL][PROPN_EMPTY] += 1
            
    elif not gt_propositions[DUCKIE] and not gt_propositions[DUCKIEBOT] and gt_propositions[CONE]: # GT cone
        if not pred_propositions[DUCKIE] and not pred_propositions[DUCKIEBOT] and not pred_propositions[CONE]:
            propn_labelled_confusion_matrix[PROPN_EMPTY][CONE] += 1
        elif not pred_propositions[DUCKIE] and not pred_propositions[DUCKIEBOT] and pred_propositions[CONE]:
            propn_labelled_confusion_matrix[CONE][CONE] += 1
        elif not pred_propositions[DUCKIE] and pred_propositions[DUCKIEBOT] and not pred_propositions[CONE]: 
            propn_labelled_confusion_matrix[DUCKIEBOT][CONE] += 1
        elif not pred_propositions[DUCKIE] and pred_propositions[DUCKIEBOT] and pred_propositions[CONE]:
            propn_labelled_confusion_matrix[DUCKIEBOT_CONE][CONE] += 1
        elif pred_propositions[DUCKIE] and not pred_propositions[DUCKIEBOT] and not pred_propositions[CONE]: 
            propn_labelled_confusion_matrix[DUCKIE][CONE] += 1
        elif pred_propositions[DUCKIE] and not pred_propositions[DUCKIEBOT] and pred_propositions[CONE]: 
            propn_labelled_confusion_matrix[DUCKIE_CONE][CONE] += 1
        elif pred_propositions[DUCKIE] and pred_propositions[DUCKIEBOT] and not pred_propositions[CONE]: 
            propn_labelled_confusion_matrix[DUCKIE_DUCKIEBOT][CONE] += 1
        elif pred_propositions[DUCKIE] and pred_propositions[DUCKIEBOT] and pred_propositions[CONE]: 
            propn_labelled_confusion_matrix[ALL][CONE] += 1
            
    elif not gt_propositions[DUCKIE] and gt_propositions[DUCKIEBOT] and not gt_propositions[CONE]: # GT duckiebot
        if not pred_propositions[DUCKIE] and not pred_propositions[DUCKIEBOT] and not pred_propositions[CONE]:
            propn_labelled_confusion_matrix[PROPN_EMPTY][DUCKIEBOT] += 1
        elif not pred_propositions[DUCKIE] and not pred_propositions[DUCKIEBOT] and pred_propositions[CONE]:
            propn_labelled_confusion_matrix[CONE][DUCKIEBOT] += 1
        elif not pred_propositions[DUCKIE] and pred_propositions[DUCKIEBOT] and not pred_propositions[CONE]: 
            propn_labelled_confusion_matrix[DUCKIEBOT][DUCKIEBOT] += 1
        elif not pred_propositions[DUCKIE] and pred_propositions[DUCKIEBOT] and pred_propositions[CONE]:
            propn_labelled_confusion_matrix[DUCKIEBOT_CONE][DUCKIEBOT] += 1
        elif pred_propositions[DUCKIE] and not pred_propositions[DUCKIEBOT] and not pred_propositions[CONE]: 
            propn_labelled_confusion_matrix[DUCKIE][DUCKIEBOT] += 1
        elif pred_propositions[DUCKIE] and not pred_propositions[DUCKIEBOT] and pred_propositions[CONE]: 
            propn_labelled_confusion_matrix[DUCKIE_CONE][DUCKIEBOT] += 1
        elif pred_propositions[DUCKIE] and pred_propositions[DUCKIEBOT] and not pred_propositions[CONE]: 
            propn_labelled_confusion_matrix[DUCKIE_DUCKIEBOT][DUCKIEBOT] += 1
        elif pred_propositions[DUCKIE] and pred_propositions[DUCKIEBOT] and pred_propositions[CONE]: 
            propn_labelled_confusion_matrix[ALL][DUCKIEBOT] += 1
            
    elif not gt_propositions[DUCKIE] and gt_propositions[DUCKIEBOT] and gt_propositions[CONE]: # GT cone and duckiebot
        if not pred_propositions[DUCKIE] and not pred_propositions[DUCKIEBOT] and not pred_propositions[CONE]:
            propn_labelled_confusion_matrix[PROPN_EMPTY][DUCKIEBOT_CONE] += 1
        elif not pred_propositions[DUCKIE] and not pred_propositions[DUCKIEBOT] and pred_propositions[CONE]:
            propn_labelled_confusion_matrix[CONE][DUCKIEBOT_CONE] += 1
        elif not pred_propositions[DUCKIE] and pred_propositions[DUCKIEBOT] and not pred_propositions[CONE]: 
            propn_labelled_confusion_matrix[DUCKIEBOT][DUCKIEBOT_CONE] += 1
        elif not pred_propositions[DUCKIE] and pred_propositions[DUCKIEBOT] and pred_propositions[CONE]:
            propn_labelled_confusion_matrix[DUCKIEBOT_CONE][DUCKIEBOT_CONE] += 1
        elif pred_propositions[DUCKIE] and not pred_propositions[DUCKIEBOT] and not pred_propositions[CONE]: 
            propn_labelled_confusion_matrix[DUCKIE][DUCKIEBOT_CONE] += 1
        elif pred_propositions[DUCKIE] and not pred_propositions[DUCKIEBOT] and pred_propositions[CONE]: 
            propn_labelled_confusion_matrix[DUCKIE_CONE][DUCKIEBOT_CONE] += 1
        elif pred_propositions[DUCKIE] and pred_propositions[DUCKIEBOT] and not pred_propositions[CONE]: 
            propn_labelled_confusion_matrix[DUCKIE_DUCKIEBOT][DUCKIEBOT_CONE] += 1
        elif pred_propositions[DUCKIE] and pred_propositions[DUCKIEBOT] and pred_propositions[CONE]: 
            propn_labelled_confusion_matrix[ALL][DUCKIEBOT_CONE] += 1
            
    elif gt_propositions[DUCKIE] and not gt_propositions[DUCKIEBOT] and not gt_propositions[CONE]: # GT duckie
        if not pred_propositions[DUCKIE] and not pred_propositions[DUCKIEBOT] and not pred_propositions[CONE]:
            propn_labelled_confusion_matrix[PROPN_EMPTY][DUCKIE] += 1
        elif not pred_propositions[DUCKIE] and not pred_propositions[DUCKIEBOT] and pred_propositions[CONE]:
            propn_labelled_confusion_matrix[CONE][DUCKIE] += 1
        elif not pred_propositions[DUCKIE] and pred_propositions[DUCKIEBOT] and not pred_propositions[CONE]: 
            propn_labelled_confusion_matrix[DUCKIEBOT][DUCKIE] += 1
        elif not pred_propositions[DUCKIE] and pred_propositions[DUCKIEBOT] and pred_propositions[CONE]:
            propn_labelled_confusion_matrix[DUCKIEBOT_CONE][DUCKIE] += 1
        elif pred_propositions[DUCKIE] and not pred_propositions[DUCKIEBOT] and not pred_propositions[CONE]: 
            propn_labelled_confusion_matrix[DUCKIE][DUCKIE] += 1
        elif pred_propositions[DUCKIE] and not pred_propositions[DUCKIEBOT] and pred_propositions[CONE]: 
            propn_labelled_confusion_matrix[DUCKIE_CONE][DUCKIE] += 1
        elif pred_propositions[DUCKIE] and pred_propositions[DUCKIEBOT] and not pred_propositions[CONE]: 
            propn_labelled_confusion_matrix[DUCKIE_DUCKIEBOT][DUCKIE] += 1
        elif pred_propositions[DUCKIE] and pred_propositions[DUCKIEBOT] and pred_propositions[CONE]: 
            propn_labelled_confusion_matrix[ALL][DUCKIE] += 1
            
    elif gt_propositions[DUCKIE] and not gt_propositions[DUCKIEBOT] and gt_propositions[CONE]: # GT duckie and cone
        if not pred_propositions[DUCKIE] and not pred_propositions[DUCKIEBOT] and not pred_propositions[CONE]:
            propn_labelled_confusion_matrix[PROPN_EMPTY][DUCKIE_CONE] += 1
        elif not pred_propositions[DUCKIE] and not pred_propositions[DUCKIEBOT] and pred_propositions[CONE]:
            propn_labelled_confusion_matrix[CONE][DUCKIE_CONE] += 1
        elif not pred_propositions[DUCKIE] and pred_propositions[DUCKIEBOT] and not pred_propositions[CONE]: 
            propn_labelled_confusion_matrix[DUCKIEBOT][DUCKIE_CONE] += 1
        elif not pred_propositions[DUCKIE] and pred_propositions[DUCKIEBOT] and pred_propositions[CONE]:
            propn_labelled_confusion_matrix[DUCKIEBOT_CONE][DUCKIE_CONE] += 1
        elif pred_propositions[DUCKIE] and not pred_propositions[DUCKIEBOT] and not pred_propositions[CONE]: 
            propn_labelled_confusion_matrix[DUCKIE][DUCKIE_CONE] += 1
        elif pred_propositions[DUCKIE] and not pred_propositions[DUCKIEBOT] and pred_propositions[CONE]: 
            propn_labelled_confusion_matrix[DUCKIE_CONE][DUCKIE_CONE] += 1
        elif pred_propositions[DUCKIE] and pred_propositions[DUCKIEBOT] and not pred_propositions[CONE]: 
            propn_labelled_confusion_matrix[DUCKIE_DUCKIEBOT][DUCKIE_CONE] += 1
        elif pred_propositions[DUCKIE] and pred_propositions[DUCKIEBOT] and pred_propositions[CONE]: 
            propn_labelled_confusion_matrix[ALL][DUCKIE_CONE] += 1
            
    elif gt_propositions[DUCKIE] and gt_propositions[DUCKIEBOT] and not gt_propositions[CONE]: # GT duckie and duckiebot
        if not pred_propositions[DUCKIE] and not pred_propositions[DUCKIEBOT] and not pred_propositions[CONE]:
            propn_labelled_confusion_matrix[PROPN_EMPTY][DUCKIE_DUCKIEBOT] += 1
        elif not pred_propositions[DUCKIE] and not pred_propositions[DUCKIEBOT] and pred_propositions[CONE]:
            propn_labelled_confusion_matrix[CONE][DUCKIE_DUCKIEBOT] += 1
        elif not pred_propositions[DUCKIE] and pred_propositions[DUCKIEBOT] and not pred_propositions[CONE]: 
            propn_labelled_confusion_matrix[DUCKIEBOT][DUCKIE_DUCKIEBOT] += 1
        elif not pred_propositions[DUCKIE] and pred_propositions[DUCKIEBOT] and pred_propositions[CONE]:
            propn_labelled_confusion_matrix[DUCKIEBOT_CONE][DUCKIE_DUCKIEBOT] += 1
        elif pred_propositions[DUCKIE] and not pred_propositions[DUCKIEBOT] and not pred_propositions[CONE]: 
            propn_labelled_confusion_matrix[DUCKIE][DUCKIE_DUCKIEBOT] += 1
        elif pred_propositions[DUCKIE] and not pred_propositions[DUCKIEBOT] and pred_propositions[CONE]: 
            propn_labelled_confusion_matrix[DUCKIE_CONE][DUCKIE_DUCKIEBOT] += 1
        elif pred_propositions[DUCKIE] and pred_propositions[DUCKIEBOT] and not pred_propositions[CONE]: 
            propn_labelled_confusion_matrix[DUCKIE_DUCKIEBOT][DUCKIE_DUCKIEBOT] += 1
        elif pred_propositions[DUCKIE] and pred_propositions[DUCKIEBOT] and pred_propositions[CONE]: 
            propn_labelled_confusion_matrix[ALL][DUCKIE_DUCKIEBOT] += 1
            
    elif gt_propositions[DUCKIE] and gt_propositions[DUCKIEBOT] and gt_propositions[CONE]: # GT duckie and duckiebot and cone
        if not pred_propositions[DUCKIE] and not pred_propositions[DUCKIEBOT] and not pred_propositions[CONE]:
            propn_labelled_confusion_matrix[PROPN_EMPTY][ALL] += 1
        elif not pred_propositions[DUCKIE] and not pred_propositions[DUCKIEBOT] and pred_propositions[CONE]:
            propn_labelled_confusion_matrix[CONE][ALL] += 1
        elif not pred_propositions[DUCKIE] and pred_propositions[DUCKIEBOT] and not pred_propositions[CONE]: 
            propn_labelled_confusion_matrix[DUCKIEBOT][ALL] += 1
        elif not pred_propositions[DUCKIE] and pred_propositions[DUCKIEBOT] and pred_propositions[CONE]:
            propn_labelled_confusion_matrix[DUCKIEBOT_CONE][ALL] += 1
        elif pred_propositions[DUCKIE] and not pred_propositions[DUCKIEBOT] and not pred_propositions[CONE]: 
            propn_labelled_confusion_matrix[DUCKIE][ALL] += 1
        elif pred_propositions[DUCKIE] and not pred_propositions[DUCKIEBOT] and pred_propositions[CONE]: 
            propn_labelled_confusion_matrix[DUCKIE_CONE][ALL] += 1
        elif pred_propositions[DUCKIE] and pred_propositions[DUCKIEBOT] and not pred_propositions[CONE]: 
            propn_labelled_confusion_matrix[DUCKIE_DUCKIEBOT][ALL] += 1
        elif pred_propositions[DUCKIE] and pred_propositions[DUCKIEBOT] and pred_propositions[CONE]: 
            propn_labelled_confusion_matrix[ALL][ALL] += 1
                    

def run_validation(csv_file, img_folder):
    
    csv_file = Path(csv_file)
    img_folder = Path(img_folder)

    if csv_file is None or not csv_file.exists():
        raise ValueError("No CSV file found")
    if img_folder is None or not img_folder.exists():
        raise ValueError("No image folder found")

    # Match names of the two files to ensure they refer to the same set of images
    csv_file_name = str(csv_file.name)[12:-4]
    img_folder_name = str(img_folder.name)[7:]

    if csv_file_name != img_folder_name:
        raise ValueError("CSV file and image folder do not match")
    
    # Iterate through the images in the folder
    for img in Path(img_folder).iterdir():  
        # Get the predicted bounding boxes
        predictions, im_ann = get_pytorch_bbox(str(img), trained_model)
        # Get the ground truth bounding boxes
        ground_truth_dict = get_ground_truth_bbox(str(csv_file.absolute()), img.name)
        for gt in ground_truth_dict["duckiebot"] + ground_truth_dict["duckie"] + ground_truth_dict["cone"]:
            gt_class = gt["class"]
            gt_bbox = gt["bbox"]
            gt_dist = gt["distance"]
            
            im_ann = cv2.rectangle(im_ann, (int(gt_bbox[0]), int(gt_bbox[1])), (int(gt_bbox[2]), int(gt_bbox[3])), color=(0, 0, 255), thickness=4)
            im_ann = cv2.putText(im_ann, f"{gt_class}/{gt_dist}", (int(gt_bbox[0]-15), int(gt_bbox[3]+10)), color=(0, 0, 255), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.755)
            
        cv2.imwrite(f"{home}/duckietown_dataset/validating_conf_mat/{img.name}_validation.jpg", im_ann)

def calculateAll(filePath, lower_distance_threshold = 0, upper_distance_threshold = numpy.inf):
    
    fileList = os.listdir(filePath)
    folders = []
    csv_files = []


    for file in fileList:
        if file.startswith("images") and os.path.isdir(os.path.join(filePath, file)): folders.append(file)
        if file.endswith('.csv'): csv_files.append(file)

    if not csv_files:
        raise ValueError("No CSV files found")
    if not folders:
        raise ValueError("No image folders found")

    for file in csv_files:
        csvdate = file.split("_")[1].split(".")[0]

        for imagefolder in folders:
            imagedate = imagefolder.split("_")[1]

            if (csvdate == imagedate):
                imageList = os.listdir(os.path.join(filePath, imagefolder))
                imagePath = os.path.join(filePath, imagefolder)
                csvPath = os.path.join(filePath, file)

                for image in imageList:
                    calculate(f"{imagePath}/{image}", csvPath, lower_dist_threshold=lower_distance_threshold, upper_dist_threshold=upper_distance_threshold)
            # calculate(f"{imagePath}/{'23-59-IM4.jpg'}", csvPath, lower_dist_threshold=lower_distance_threshold, upper_dist_threshold=upper_distance_threshold)

if __name__ == "__main__":
    run_validation(img_folder=f"{FOLDER_PATH}/images_13-12-2023", csv_file=f"{FOLDER_PATH}/annotations_13-12-2023.csv")
    
    # remove distance threshold parameter to calculate full confusion matrix
    if DO_VALIDATION: calculateAll(FOLDER_PATH, LOWER_DIST_THRESH, UPPER_DIST_THRESH)

    confusion_matrix_freq = numpy.array(confusion_matrix_freq)
    propn_labelled_confusion_matrix = numpy.array(propn_labelled_confusion_matrix)
    classes = ["Duckiebot", "Duckie", "Cone", "Empty"]
    
    if(REMOVE_CONES):
        # removes rows
        confusion_matrix_freq = numpy.delete(confusion_matrix_freq, (2), axis=0)
        # removes columns
        confusion_matrix_freq = numpy.delete(confusion_matrix_freq, (2), axis=1)
        # select rows
        propn_labelled_confusion_matrix = propn_labelled_confusion_matrix[[0, 1, 3, 7], :]
        # select columns
        propn_labelled_confusion_matrix = propn_labelled_confusion_matrix[:, [0, 1, 3, 7]]
        
        classes = ["Duckiebot", "Duckie", "Empty"]

    plt.cla()
    plt.clf()

    if GEN_PERCENTAGE:
        heatmap = seaborn.heatmap(confusion_matrix_freq/numpy.sum(confusion_matrix_freq, axis=0), cmap='Blues', yticklabels=classes, xticklabels=classes, annot=True)
    else:
        heatmap = seaborn.heatmap(confusion_matrix_freq, cmap='Blues', yticklabels=classes, xticklabels=classes, annot=True)
    heatmap.set(xlabel = "Ground Truth", ylabel="Predictions")
    
    #to show
    plt.show()
    
    name = "ConfusionMatrix.png"
    if GEN_PERCENTAGE:
        name = "ConfusionMatrix_percentages.png"
    if LOWER_DIST_THRESH > 0 and UPPER_DIST_THRESH != numpy.inf:
        name = f"ConfusionMatrix_{LOWER_DIST_THRESH}_{UPPER_DIST_THRESH}.png"
    # to save
    plt.savefig(f"{FOLDER_PATH}/{name}")
    
    classes = ["DB", "DK", "DK & DB", "Empty"]

    plt.cla()
    plt.clf()

    if GEN_PERCENTAGE:
        heatmap = seaborn.heatmap(propn_labelled_confusion_matrix/numpy.sum(propn_labelled_confusion_matrix, axis=0), cmap='Blues', yticklabels=classes, xticklabels=classes, annot=True)
    else:
        heatmap = seaborn.heatmap(propn_labelled_confusion_matrix, cmap='Blues', yticklabels=classes, xticklabels=classes, annot=True)
    heatmap.set(xlabel = "Ground Truth", ylabel="Predictions")
    
    name = "PropositionLabelledConfusionMatrix.png"
    if LOWER_DIST_THRESH > 0 and UPPER_DIST_THRESH != numpy.inf:
        name = f"PropositionLabelledConfusionMatrix_{LOWER_DIST_THRESH}_{UPPER_DIST_THRESH}.png"
        
    plt.savefig(f"{FOLDER_PATH}/{name}")
    
