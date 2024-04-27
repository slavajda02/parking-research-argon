import cv2
from PIL import Image
from baselines.utils.inter_utils import *
from baselines.utils.common_utils import get_device
from baselines.intersection_based.inter_models import *
from torchvision import transforms
from picamera2 import Picamera2
import timeit
import os

model_path = r"Saved_Models/Low_epoch_T10LOT/state_dict_final.pth"
preprocess = transforms.Compose([
    transforms.ToTensor(),
])
picam2 = Picamera2()
camera_config = picam2.create_still_configuration({"size" : (3200, 1800)})
picam2.configure(camera_config)
picam2.start()

while True:
    tic = timeit.default_timer()
    image = picam2.capture_array()
    image = image[:,:, [2, 1, 0]]
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    device = get_device()
    model = get_model(faster_rcnn_mobilenetV3_Large_params, True)
    load_model(model, device, model_path)
    pred_boxes, pred_score = make_pred(model, device, input_batch, 0.8)
    toc = timeit.default_timer()
    os.system("clear")
    print("Cars detected: ", len(pred_boxes))
    print(f"Time of one frame: {toc-tic} s")
    
cv2.imwrite(filename = "test.png", img = image)
