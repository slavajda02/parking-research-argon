from flask import Flask, render_template, flash, redirect
import queue
from config import Config
from app.argonPark import *
from app.forms import LoginForm
import multiprocessing
#from picamera2 import Picamera2

import time
import cv2
import os

web = Flask(__name__)
web.config.from_object(Config)
p = None

#DEBUGING purposes
path = r"T10LOT/T10LOT/Images"
images_dir = []

##Multiprocessing class everyting running here is seperate process!
##Acess only through queue and other multiprocessing tools
class parkingProcess(multiprocessing.Process):
    def __init__(self, process_queue, stop_event):
        multiprocessing.Process.__init__(self)
        self.img_pointer = 0
        self.queue = process_queue
        self.stop_event = stop_event
        self.parking = parkingLot('map.json', 'state_dict_final.pth')
        self.stop_flag = False
        self.image_flag = True
    
    def run(self):
        while not self.stop_event.is_set():
            self.image = getImage(self.img_pointer)
            status_dict = self.parking.evaulate_occupancy(self.image)
            self.queue.put(status_dict)
            self.img_pointer += 1
            if self.stop_flag:
                break
            if self.image_flag:
                self.image = self.parking.plot_to_image(self.image)
                cv2.imwrite('static/img/output.jpg', self.image)
            # Sleep for 3 seconds, but check the stop_event every 0.1 seconds
            for _ in range(30):
                time.sleep(0.1)
                if self.stop_event.is_set():
                    break
        
    def stop(self):
        self.stop_flag = True
        self.stop_event.set()


##Helper functions
#Starts Picamera2 and configures it
def prepareCamera():
    picam2 = Picamera2()
    camera_config = picam2.create_still_configuration({"size" : (3200, 1800)})
    picam2.configure(camera_config)
    picam2.start()

#Gets an image form Picamera2 in a correct format
def getImageCamera():
    image = picam2.capture_array()
    image = image[:,:, [2, 1, 0]]
    return image

#Getting an image from a directory for testing purposes
def getImage(pointer):
    if not images_dir:
        for p in os.listdir(path):
            images_dir.append(path + '/' + p)
    image = cv2.imread(images_dir[pointer])
    return image


##Flask routes and functions
@web.route('/')
def index():
    return render_template('index.html')

@web.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        flash('Login requested for user {}, remember_me={}'.format(
            form.username.data, form.remember_me.data))
        return redirect('/')
    return render_template('login.html', form=form)

@web.route('/parking')
def show_parking():
    global p
    if p is not None:
        try:
            status = p.queue.get_nowait()
        except queue.Empty:
            status = None
        if status:
            i = 0
            for lot in status:
                if lot["status"]:
                    i+=1
            flash(f"Request sucessful, {i} parking spaces occupied")
            return redirect('/')
    flash("Request failed, no data")
    return redirect('/')

@web.route('/parking/start')
def start_parking():
    global p
    if p is None or not p.is_alive():
        p = parkingProcess(process_queue, stop_event)
        p.start()
        flash("Parking inference started")
    else:
        flash("Parking inference already running")
    return redirect('/')

@web.route('/parking/stop')
def stop_parking():
    global p
    if p is not None and p.is_alive():
        p.stop()
        p.join()
        p = None
        stop_event.clear()
        flash("Parking inference stopped")
    else:
        flash("Parking inference not running")
    return redirect('/')

@web.route('/parking/image')
def show_image():
    global p
    if p is not None and p.is_alive():
        flash("Image updated")
    else:
        flash("No inference running")
    return redirect('/')

##Main
if __name__ == '__main__':
    #Creates a multiprocessing object and runs it's loop
    process_queue = multiprocessing.Queue()
    stop_event = multiprocessing.Event()
    try:
        web.run(debug=True, host='0.0.0.0', port=80, use_reloader=False) #Web server launch
    except KeyboardInterrupt:
        p.stop() #Stops the inference loop