import app
from multiprocessing import Queue, Event, Process, Value
from datetime import datetime, timezone, timedelta
import cv2
from picamera2 import Picamera2
import numpy as np

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
    def __init__(self, process_queue, stop_event, task_start, task_done, image_save, image_raw, json_reload, state_dict_reload, db):
        Process.__init__(self)
        self.queue = process_queue
        self.stop_event = stop_event
        self.task_done = task_done
        self.task_start = task_start
        self.parking = app.parkingLot('uploads/map.json', 'uploads/state_dict_final.pth')
        self.image_save = image_save
        self.image_raw = image_raw
        self.json_reaload = json_reload
        self.state_dict_reload = state_dict_reload
        self.collection = db['parking_data_testing']
    
    def run(self):
        ##DEBUG
        #i = 0
        #image_names = []
        #for filename in os.listdir('T10LOT/T10LOT/Images'):
        #    if filename.endswith('.jpg') or filename.endswith('.png'):
        #        image_names.append('T10LOT/T10LOT/Images/' + filename)
        #images_names = image_names[100:]
        #timer = DELAY #First inference runs instantly
        
        db_update_time = datetime.now() - timedelta(minutes=1) #So the first inference gets send
        picam2 = Picamera2()
        camera_config = picam2.create_still_configuration({"size" : (3200, 1800)})
        picam2.configure(camera_config)
        picam2.start()
        
        while not self.stop_event.is_set():
            #Obtaining image
            self.image = picam2.capture_array()
            self.image = cv2.resize(self.image, (1920, 1080))
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.raw = self.image.copy()
            
            #self.image = cv2.imread(images_names[i])
            #self.image = self.image[:,:, [2, 1, 0]]
            #Inference
            status_dict = self.parking.evaulate_occupancy(self.image)
            
            #Make sure to send only up to date data
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except self.queue.Empty:
                    continue
            self.queue.put(status_dict)
            
            #Result image save
            if self.image_save.value:
                try:
                    cv2.imwrite('app/static/img/output.jpg', self.parking.plot_to_image())
                except AttributeError:
                    print("No image yet loaded, run inference first")
            #Raw image save
            if self.image_raw.value:
                cv2.imwrite('app/static/img/raw.jpg', self.raw)
                self.image_raw.value = False
                self.task_start.clear()
            #Reloads the map json
            if self.json_reaload.value:
                self.parking.reload_JSON('uploads/map.json')
                self.json_reaload.value = False
                self.task_start.clear()
            #Reloads the state dict
            if self.state_dict_reload.value:
                self.parking.load_model('uploads/state_dict_final.pth')
                self.state_dict_reload.value = False
                self.task_start.clear()
            
            if datetime.now() - db_update_time >= timedelta(minutes=1):
                ##Prepares and sends data to database
                data_out = {"timestamp": datetime.now(timezone.utc).isoformat(), "parking_lot": []}
                for lot in status_dict:
                    lot_copy = lot.copy()
                    lot_copy.pop('polygons', None)
                    # Convert numpy arrays to lists
                    for key, value in lot_copy.items():
                        if isinstance(value, np.ndarray):
                            lot_copy[key] = value.tolist()
                    data_out["parking_lot"].append(lot_copy)
                self.collection.insert_one(data_out)            
                db_update_time = datetime.now()
                
            #Signaling that an inference loop was done
            if not self.task_start.is_set():
                self.task_done.set()
            
        
    #Stops the process, freezes if not called on p.join()
    def stop(self):
        self.stop_event.set()