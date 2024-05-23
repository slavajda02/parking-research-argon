from app.argonPark import *
from multiprocessing import Queue, Event, Process, Value
import cv2
import os
import time
from datetime import datetime
#from picamera2 import Picamera2

##Flag variables for communication between processes
image_save = Value('b', False) #Updates image each iteration
image_raw = Value('b', False) #Saves a raw image for map creation
json_reload = Value('b', False) #Reloads the json map
state_dict_reload = Value('b', False) #Reloads the state dict

##DEBUG DEVELOPMENT
DELAY = 3 #Fetch new image every 5 seconds

##Multiprocessing class everyting running here is seperate process!
##Acess only through queue and other multiprocessing tools
class parkingProcess(Process):
    def __init__(self, process_queue, stop_event, image_save, image_raw, json_reload, state_dict_reload, db):
        Process.__init__(self)
        self.queue = process_queue
        self.stop_event = stop_event
        self.parking = parkingLot('uploads/map.json', 'uploads/state_dict_final.pth')
        self.image_save = image_save
        self.image_raw = image_raw
        self.json_reaload = json_reload
        self.state_dict_reload = state_dict_reload
        self.collection = db['parking_data_test']
    
    def run(self):
        i = 0
        image_names = []
        for filename in os.listdir('T10LOT/T10LOT/Images'):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_names.append('T10LOT/T10LOT/Images/' + filename)
        images_names = image_names[100:]
        timer = DELAY #First inference runs instantly
        
        #picam2 = Picamera2()
        #camera_config = picam2.create_still_configuration({"size" : (3200, 1800)})
        #picam2.configure(camera_config)
        #picam2.start()
        
        while not self.stop_event.is_set():
            #self.image = picam2.capture_array()
            #self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            
            if timer >= DELAY:
                self.image = cv2.imread(images_names[i])
                self.image = self.image[:,:, [2, 1, 0]]
                self.image = cv2.resize(self.image, (1920, 1080))
                #Inference
                status_dict = self.parking.evaulate_occupancy(self.image)
                
                ##Prepares data for database
                data_out = {"timestamp": datetime.now(datetime.UTC).isoformat(), "parking_lot": {}}
                for lot in status_dict:
                    lot.pop('polygons', None)
                    data_out["parking_lot"].append(lot)
                self.collection.insert_one(data_out)
                
                #Sends data to the webserver
                self.queue.put(status_dict)
                #Result image save
                if self.image_save.value:
                    try:
                        cv2.imwrite('static/img/output.jpg', self.parking.plot_to_image())
                    except AttributeError:
                        print("No image yet loaded, run inference first")
                #Restart image cycling
                if i == len(images_names) - 1:
                    i = 0
                else:
                    i += 1
                timer = 0
            
                
            #Reloads the map json
            if self.json_reaload.value:
                self.parking.reload_JSON('map.json')
                self.json_reaload.value = False
            #Reloads the state dict
            if self.state_dict_reload.value:
                self.parking.load_model('uploads/state_dict_final.pth')
                self.state_dict_reload.value = False
            #Raw image save
            if self.image_raw.value:
                cv2.imwrite('static/img/raw.jpg', self.image)
                self.image_raw.value = False

            time.sleep(0.1)
            timer += 0.1
        
    #Stops the process, freezes if not called on p.join()
    def stop(self):
        self.stop_event.set()