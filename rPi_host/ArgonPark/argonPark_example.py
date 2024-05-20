from baselines.utils.argonPark import *

#Example usage of argonPark
input_img = cv2.imread(r"Models/test.jpg")
parking_lot = r"Models/map.json"
model_path = r"Models/Saved_Models/Low_epoch_T10LOT/state_dict_final.pth"


test = parkingLot(parking_lot, model_path)
status = test.evaulate_occupancy(input_img)
test.plot_to_image(input_img, debug = True)
test.show_inference(input_img)
for lot in status:
    if lot["status"]:
        print(f"Lot {lot['name']} is occupied")
        print(f"Intersecting area: {round(lot['iou'],2)}")