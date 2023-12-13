#! /usr/bin/python3

import os
import cv2
import csv
import json
import torch
import seaborn

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DUCKIEBOT = 0
DUCKIE = 1
CONE = 2
EMPTY = 3

trained_model = None
confusion_matrix_freq = [
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]
                        ]

i = 0


home = str(Path.home())
trained_model = torch.hub.load(home + '/software/yolov5', 'custom',
                                   path=home + '/software/yolov5/trained_models/best.pt', source='local',
                                   force_reload=True)
trained_model.conf = 0.4
print("Done loading YOLOv5 trained model")

# getting things from pytorch inference
def get_pytorch_bbox(img_path, trained_model):
    home = str(Path.home())
    # trained_model = torch.hub.load(home + '/Ranai/Research/yolov5', 'custom',
    #                                path=home + '/Ranai/Research/yolov5/duckietown_weights/best.pt', source='local',
    #                                force_reload=True)    

    cv_img = cv2.imread(img_path)
    cv2.imwrite(f"confusion_matrix_trial{i}.jpeg", cv_img)
    print(f"confusion_matrix_trial{i}.jpeg")
    i += 1

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

    return predictions


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
                    x,y,w,h = coneJSON["bbox"]
                    coneJSON["bbox"] = [x,y, x+w, y+h]
                    ground_truth_json["cone"].append(coneJSON)

    return ground_truth_json


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


def calculate(img_path:str, annotation_path):
    # img_path = "img_1569225723.195207118.png"


    preds = get_pytorch_bbox(img_path, trained_model)
    img_name = img_path[img_path.rfind("/")+1:]
    truth = get_ground_truth_bbox(annotation_path, img_name)

    max_IOU = 0
    best_match_pred = None
    best_match_anntn = None #TODO have the default class EMPTY and ONLY IF one is found, then change it

    pred_list = preds["duckiebot"] + preds["duckie"] + preds["cone"]
    true_list = truth["duckiebot"] + truth["duckie"] + truth["cone"]
    matched_list = []
    
    for true in true_list:
        if len(pred_list) == 0:
            confusion_matrix_freq[EMPTY][true["class_id"]] += 1
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
        else:
            preds[best_match_pred["class"]].remove(best_match_pred)
            truth[best_match_anntn["class_name"]].remove(best_match_anntn)
            matched_list.append(best_match_pred)

            confusion_matrix_freq[best_match_pred["class_id"]][best_match_anntn["class_id"]] += 1

        max_IOU = 0
        
        
    for pred in pred_list:
        if len(pred_list) and (pred not in matched_list):
            confusion_matrix_freq[pred["class_id"]][EMPTY] += 1


def calculate_distance_parameterized(img_path:str, annotation_path, distance):
# img_path = "img_1569225723.195207118.png"

    preds = get_pytorch_bbox(img_path, trained_model)
    img_name = img_path[img_path.rfind("/")+1:]
    truth = get_ground_truth_bbox(annotation_path, img_name)

    max_IOU = 0
    best_match_pred = None
    best_match_anntn = None #TODO have the default class EMPTY and ONLY IF one is found, then change it

    pred_list = preds["duckiebot"] + preds["duckie"] + preds["cone"]
    true_list = truth["duckiebot"] + truth["duckie"] + truth["cone"]
    matched_list = []

    for true in true_list:
        if float(true["distance"]) < distance:
            if len(pred_list) == 0:
                confusion_matrix_freq[EMPTY][true["class_id"]] += 1
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
            else:
                preds[best_match_pred["class"]].remove(best_match_pred)
                truth[best_match_anntn["class_name"]].remove(best_match_anntn)
                matched_list.append(best_match_pred)

                confusion_matrix_freq[best_match_pred["class_id"]][best_match_anntn["class_id"]] += 1

            max_IOU = 0
            
            
            for pred in pred_list:
                if len(pred_list) and (pred not in matched_list):
                    confusion_matrix_freq[pred["class_id"]][EMPTY] += 1

        

def calculateAll(filePath, distance_threshold = np.inf):
    fileList = os.listdir(filePath)
    folders = []
    csvfile = []
    for file in fileList:
        if file.startswith("images") and os.path.isdir(os.path.join(filePath, file)): folders.append(file)
        if file.endswith('.csv'): csvfile.append(file)
    
    for file in csvfile:
        csvdate = file.split("_")[1].split(".")[0]

        for imagefolder in folders:
            imagedate = imagefolder.split("_")[1]

            if (csvdate == imagedate):
                imageList = os.listdir(os.path.join(filePath, imagefolder))
                imagePath = os.path.join(filePath, imagefolder)
                csvPath = os.path.join(filePath, file)

                for image in imageList:
                    calculate(imagePath+"/"+image, csvPath, distance_threshold)
                
                

    # TODO
    # calculate the IOUs and determine if it greater than a threshold

    # TODO
    # add to a counter based on the class of the object

    # TODO 
    # make this main method a function that takes in 
    #   the path to the image and
    #   the path to the csv file and adds it to the counter than this class maintains

    # cv_img = cv2.imread(img_path)

    # for key in preds.keys():
    #     anntn = preds[key]
    #     print(anntn)

    #     tag = anntn["tag"]
    #     top_left = anntn["top_left"]
    #     bot_right = anntn["bot_right"]
    #     obj_class = anntn["class"]

    #     im_ann = cv2.rectangle(cv_img, top_left, bot_right, color=(255, 0, 0), thickness=4)
    #     im_ann = cv2.putText(im_ann, f"{obj_class}", top_left, color=(255, 255, 0), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1)

    #     cv2.imshow("Image", im_ann)
    #     cv2.waitKey(0)


    

if __name__ == '__main__':
    home = str(Path.home())
    
    # remove distance threshold parameter to calculate fll confusion matrix
    calculateAll(home + "/duckietown_dataset/images_08-11-2023")
    # calculate(home + "/duckietown_dataset/01-43-IM0.jpg", home + "/duckietown_dataset/annotations_24-10-2023.csv")
    # calculate(home + "/duckietown_dataset/19-17-IM1.jpg", home + "/duckietown_dataset/annotations_08-11-2023.csv")
    
    print(confusion_matrix_freq)

    #to remove cones
    confusion_matrix_freq = np.array(confusion_matrix_freq)
    # removes rows
    confusion_matrix_freq = np.delete(confusion_matrix_freq, (2), axis=0)
    # removes columns
    confusion_matrix_freq = np.delete(confusion_matrix_freq, (2), axis=1)

    plt.cla()
    plt.clf()
    # classes = ["Duckiebot", "Duckie", "Cone", "Empty"]
    classes = ["Duckiebot", "Duckie", "Empty"]
    #For percentages
    # heatmap = seaborn.heatmap(confusion_matrix_freq/np.sum(confusion_matrix_freq, axis=0), cmap='Blues', yticklabels=classes, xticklabels=classes, annot=True)
    #For counts
    heatmap = seaborn.heatmap(confusion_matrix_freq, cmap='Blues', yticklabels=classes, xticklabels=classes, annot=True)
    heatmap.set(xlabel = "Ground Truth", ylabel="Predictions")
    
    # plt.plot()
    #to show
    plt.show()
    
    # to save
    plt.savefig("ConfusionMatrix_percentages_30.png")

