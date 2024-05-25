import numpy as np
import torch
import cv2
import json
import torch
torch.backends.quantized.engine = 'qnnpack'
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, faster_rcnn
import warnings
from shapely import Polygon, box
from rtree import index
from shapely.ops import unary_union
from datetime import datetime
import time


class parkingLot:
    """
    Class that loads a map and an AI model for a parking lot occupancy detection.
    Contains decision algorithms for parking Lot occupancy.
    Takes images as inputs and returns occupancy status with images.
    """
    def __init__(self, parking_locations, path, img_scale = 1):
        #Reads the initial parking lot locations from the map json
        self.img_scale = img_scale
        self.img_size = (1920, 1080)
        self.reload_JSON(parking_locations)
        
        #Getting Torch vision ready
        self.device = self.get_device()
        self.preprocess = transforms.Compose([
        transforms.ToTensor(),
        ])
        
        #Creates a FasterRCNNV3_Large model with 2 classes and loads state dict
        self.model = fasterrcnn_mobilenet_v3_large_fpn(weights = "DEFAULT")
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, 2)
        self.load_model(path) #Loads a model from state dict
        self.infertime = 0
        
    def reload_JSON(self, path):
        """
        Reloads the JSON map file for the parking lot.

        Args:
            path (str): Path to the JSON file
        """
        with open(path) as f:
            self.lots = json.load(f)
        if not self.img_scale == 1:
            self.lots = np.array(self.lots, np.int32) #Convert to numpy array
            self.lots = self.lots * self.img_scale
            
        self.lots = np.array(self.lots, np.int32) #Convert to numpy array
        self.parking_spaces = [{"name": i+1, "cords": lot, "status": False} for i, lot in enumerate(self.lots)] #List of dictionaries for parking spaces enumerated with status set to False
        #Converts the cords to polygons for easier intersection
        for i, lot in enumerate(self.parking_spaces):
            points = lot["cords"]
            self.parking_spaces[i]["polygons"] = Polygon([(points[0][0],points[0][1]), (points[1][0],points[1][1]), (points[2][0],points[2][1]), (points[3][0],points[3][1])])
            
    def get_device(debug = False) -> torch.device:
        """
        Gets device to be used for machine learning.
        
        Args:
            debug (bool, optional): Prints selected device. Defaults to False.

        Returns:
            torch.device: Device for torch to use
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if debug:
            if device.type == "cpu":
                print("Using CPU")
                print("Recommend using GPU for faster training")
            else:
                print("Using GPU")
        return device
    
    def load_model(self, path):
        """
        Loads a state dict onto a model.

        Args:
            path (str): Path to the state dict
        """
        if self.device.type == 'cpu':
            self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        else:
            self.model.load_state_dict(torch.load(path))
            self.model.cuda()
        self.model.eval() # Sets model to evaluation mode since no training will be done
        self.model = torch.jit.script(self.model)
    
    #({}, [{'boxes': tensor([], size=(0, 4)), 'labels': tensor([], dtype=torch.int64), 'scores': tensor([])}])
    #[{'boxes': tensor([], size=(0, 4)), 'labels': tensor([], dtype=torch.int64), 'scores': tensor([])}]
    
    def make_pred(self, img, treshold = 0.9):
        """
        Makes a prediction on a single image in a batch.

        Args:
            model (torch): Model to inference with
            img_batch (torch.tensor): Image batch containing single image
            treshold (float, optional): Detection treshold, will throw warning if below. Defaults to 0.9.

        Returns:
            pred_boxes: Two X and Y coordinates for the bounding box
            pred_score: Score of individual detections 0-1
        """
        input_tensor = self.preprocess(img)
        img_batch = input_tensor.unsqueeze(0)
        #Send image to device (would cause problem if it were missing on GPU)
        images = list(image.to(self.device) for image in img_batch)
        with torch.no_grad():
            tic = time.time()
            pred = self.model(images)
            pred = pred[1]
            toc = time.time()
        pred_boxes = [[(x[0], x[1]), (x[2], x[3])] for x in list(pred[0]["boxes"].detach().cpu().numpy())]
        pred_class = list(pred[0]["labels"].detach().cpu().numpy())
        pred_score = list(pred[0]["scores"].detach().cpu().numpy())
        try:
            over_treshold = [pred_score.index(x) for x in pred_score if x>treshold][-1]
        except IndexError:
            warnings.warn(f"Didn't detect anything over threshold {treshold}")
            over_treshold = 0
        pred_boxes = pred_boxes[:over_treshold+1]
        pred_class = pred_class[:over_treshold+1]
        self.infertime = toc-tic
        return pred_boxes, pred_score
    
    def resize_image(self, img):
        """
        Resizes an image to the desired size.

        Args:
            img (np matrix): Image to resize

        Returns:
            np matrix: Resized image
        """
        return cv2.resize(img, (int(self.img_size[0]*self.img_scale), int(self.img_size[1]*self.img_scale)))
    
    def plot_to_image(self, debug = False):
        """
        Draws polygons over parking places according to the active mapping and occupancy.
        Args:
            img (np matrix): A cv2 image to draw on.
            debug (bool, optional): If true, the function will save the image. Defaults to False.
        Returns:
            image (np matrix): Image with colored polygons drawn over parking spaces.
        """
        try:
            image = self.img
        except AttributeError:
            print("No image loaded yet, run inference")
            return None
        
        for i, lot in enumerate(self.parking_spaces):
            lot["cords"] = lot["cords"].reshape((-1,1,2))
            cv2.putText(image, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), (20,40), cv2.LINE_AA, 1.2, (255, 255, 255), 2)
            cv2.putText(image, str(round(self.infertime, 2)) + " s", (20, 1080-40), cv2.LINE_AA, 1.2, (255, 255, 255), 2)
            if lot["status"]:
                cv2.putText(image, f"{i}", lot["cords"][0][0], cv2.LINE_AA, 1.2, (0,0,255), 2)
                cv2.polylines(image, [lot["cords"]], isClosed = True, color = (0,0,255), thickness = 2)
            else:
                cv2.putText(image, f"{i}", lot["cords"][0][0], cv2.LINE_AA, 1.2, (0,255,0), 2)
                cv2.polylines(image, [lot["cords"]], isClosed = True, color = (0,255,0), thickness = 2)
        if debug:
            cv2.imwrite("parking_lot.png", image)
        return image
    
    def show_inference(self, treshold = 0.9, debug = False):
        """
        Shows the inference on an image, with marked parking slots. Goes through the whole inference process.

        Args:
            img (np matrix) Image ready by cv2
            treshold (float, optional): Treshold to flag as a detection. Defaults to 0.9.
            debug (bool, optional): When True saves image. Defaults to False.

        Returns:
            np matrix: Image with detetions marked
        """
        try:
            img = self.plot_to_image(self.img)
        except AttributeError:
            print("No image loaded yet, run inference")
            return None
        boxes, score = self.make_pred(self.img, treshold)
        #Top left corner info text
        for i, x in enumerate(boxes):
            #Car detection boxes
            cv2.rectangle(img, (int(x[0][0]),int(x[0][1])), (int(x[1][0]),int(x[1][1])), color=(0, 0, 255), thickness=2)
            #cv2.putText(img, str(round(score[i],2)), (int(x[0][0]),int(x[0][1])), cv2.LINE_AA, 1.5, (0,255,0), 1)
            cv2.putText(img, "Busy", (int(x[0][0]),int(x[0][1])), cv2.LINE_AA, 1.2, (0,0,255), 2)
        if debug:
            cv2.imwrite("parking_lot.png", img)
        return img
    
    def evaulate_occupancy(self, img, overlap = 0.8):
        """
        Evaluates the occupancy of the parking lot.

        Args:
            img (np matrix): Image to evaluate
            overlap (float, optional): Overlap treshold for detection. Defaults to 0.75.
            debug (bool, optional): Saves image if True. Defaults to False.

        Returns:
            int: Number of occupied parking spaces
        """
        if not self.img_scale == 1:
            self.img = self.resize_image(img)
        else:
            self.img = img
        idx = index.Index()
        boxes, score = self.make_pred(self.img)
        boxes = [box(x[0][0], x[0][1], x[1][0], x[1][1]) for x in boxes]
        
        for pos, cell in enumerate(boxes):
            idx.insert(pos, cell.bounds)
        
        self.calculate_iou(idx, boxes)
        
        return self.parking_spaces
    
    def calculate_iou(self, idx, boxes, area_treshold = 0.75):
        """
        Calculates the intersection over union of a box and a polygon.

        Args:
            idx (rtree.index): Index of intersected boxes
            boxes (list): List of detection boxes
        """
        for i, lot in enumerate(self.parking_spaces):
            merged_cells = unary_union([boxes[pos] for pos in idx.intersection(lot["polygons"].bounds)])
            if lot["polygons"].intersection(merged_cells).area/lot["polygons"].area > area_treshold:
                lot["status"] = True
                lot["iou"] = lot["polygons"].intersection(merged_cells).area/lot["polygons"].area
            else:
                lot["status"] = False
                lot["iou"] = 0