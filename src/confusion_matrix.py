#! /usr/bin/python3

import cv2
import csv
import json
import torch

from pathlib import Path


# getting things from pytorch inference
def get_pytorch_bbox(img_path):
    home = str(Path.home())
    # trained_model = torch.hub.load(home + '/Ranai/Research/yolov5', 'custom',
    #                                path=home + '/Ranai/Research/yolov5/duckietown_weights/best.pt', source='local',
    #                                force_reload=True)    
    
    trained_model = torch.hub.load(home + '/software/yolov5', 'custom',
                                   path=home + '/home/ranai/software/yolov5', source='local',
                                   force_reload=True)
    trained_model.conf = 0.64

    cv_img = cv2.imread(img_path)

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

            predictions[obj_class].append({"tag": f"{obj_class}{counter[obj_class]}", "class": f"{obj_class}",
                                           "bbox": [topleft[0], topleft[1], botright[0], botright[1]]})

    return predictions


# getting things from the CSV file
def get_ground_truth_bbox(csv_path, img_name):
    ground_truth_bbox = {"duckiebot": [], "duckie": [], "cone": []}
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
                    ground_truth_bbox["duckie"].append(duckJSON)
                numRobots = line[6]
                if len(numRobots) != 0:
                    robotJSON = json.loads(line[7].replace("'", '"'))
                    ground_truth_bbox["duckiebot"].append(robotJSON)
                numCones = line[8]
                if len(numCones) != 0:
                    coneJSON = json.loads(line[9].replace("'", '"'))
                    ground_truth_bbox["cone"].append(coneJSON)

    return ground_truth_bbox


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


if __name__ == '__main__':
    img_path = "img_1569225723.195207118.png"

    preds = get_pytorch_bbox(img_path)
    truth = get_ground_truth_bbox("annotations_17-10-2023.csv", img_path)

    # bbox will be in the form of (x_topleft, y_topleft, x_botright, y_botright)
    for class_name in ["duckiebot", "duckie", "cone"]:

        # TODO
        # iterate through the predictions and ground truth and find the relevant
        # bounding boxes
        max_IOU = 0
        best_match_pred = None
        best_math_anntn = None

        pred_list = preds[class_name]
        true_list = truth[class_name]
        for pred in pred_list:
            for true in true_list:
                pred_bbox = pred["bbox"]
                true_bbox = true["bbox"]

                curr = bb_intersection_over_union(pred_bbox, true_bbox)
                if curr > max_IOU:
                    max_IOU = curr
                    best_match_pred = pred
                    best_match_anntn = true

        preds[class_name].remove(best_match_pred)
        truth[class_name].remove(best_match_anntn)

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
