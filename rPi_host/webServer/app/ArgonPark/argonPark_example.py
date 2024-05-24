from rPi_host.webServer.app.ArgonPark.argonPark import *
from picamera2 import Picamera2


picam2 = Picamera2()
camera_config = picam2.create_still_configuration({"size" : (3200, 1800)})
picam2.configure(camera_config)
picam2.start()
image = picam2.capture_array()
input_img = image[:,:, [2, 1, 0]]
input_img = cv2.resize(input_img, (1920, 1080))

#Example usage of argonPark
#input_img = cv2.imread(r"Models/test.jpg")
parking_lot = r"map.json"
model_path = r"state_dict_final.pth"


test = parkingLot(parking_lot, model_path)
status = test.evaulate_occupancy(input_img)
test.plot_to_image(debug = True)
for lot in status:
    if lot["status"]:
        print(f"Lot {lot['name']} is occupied")
        print(f"Intersecting area: {round(lot['iou'],2)}")