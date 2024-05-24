import os
import time
from flask import Flask, render_template, flash, redirect, url_for, request
from flask_pymongo import PyMongo
from werkzeug.utils import secure_filename
from app.process import *
import queue
from config import Config
import plotly
import json
import plotly.express as px
import pandas as pd

##Flask setup
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'json', 'pth'}
web = Flask(__name__)
web.config.from_object(Config)
web.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
web.config["MONGO_URI"] = "mongodb+srv://240648:GRziVfkVYYMGaQWa@t10lot.mzbydiy.mongodb.net/T10LOT?retryWrites=true&w=majority&appName=T10LOT"

#Setup mongodb
mongodb_client = PyMongo(web)
db = mongodb_client.db
collection = db['parking_data_testing']

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#Gets all of the data so far and puts them in a list
def fetch_database():
    data = collection.find()
    data_list = []
    for item in data:
        data_list.append(item)
    return data_list

#Empty global variable for multiprocessing object
p = None


##Flask routes and functions
#Index page
@web.route('/')
def index():
    return render_template('index.html')

#Parking database and histogram
@web.route('/parking')
def show_parking():
    global p
    info = ""
    if p is not None:
        try:
            status = p.queue.get_nowait()
        except queue.Empty:
            status = None
            info = "Queue empty"
        if status:
            i = 0
            unoccupied = []
            for lot in status:
                if lot["status"]:
                    i+=1
                else:
                    unoccupied.append(int(lot["name"])-1)
            flash(f"{i} parking spaces occupied, closest free parking space is {min(unoccupied)}")
            return redirect('/')
    flash(f"Request failed, no new data or inference not running")
    return redirect('/')

@web.route('/history')
def show_graph():
    ##Data extraction
    data = fetch_database()
    time = []
    occupancy = []
    for entry in data:
        i = 0
        timestamp = datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00"))
        formatted_time = timestamp.strftime('%Y-%m-%d %H:%M')
        time.append(formatted_time)
        for lot in entry["parking_lot"]:
            if lot["status"]:
                i += 1
        occupancy.append(i)
    occupancy_df = pd.DataFrame({'Time' : time, 'Number of cars' : occupancy})
    
    ##Graph
    fig = px.bar(occupancy_df, x='Time', y='Number of cars', title="Parking occupancy over time", template='plotly_dark')
    graphJson = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('history.html', graphJSON=graphJson)
    
@web.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file to upload')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            if filename.rsplit('.', 1)[1].lower() == "json":
                file.save(os.path.join(web.config['UPLOAD_FOLDER'], "map.json"))
                json_reload.value = True
            else:
                file.save(os.path.join(web.config['UPLOAD_FOLDER'], "state_dict_final.pth"))
                state_dict_reload.value = True
            flash('File uploaded')
            image_raw.value = True
            task_start.set()
            task_done.clear()
            task_done.wait()
            return redirect(request.url)
        else:
            flash('Invalid file!')
            return redirect(request.url)
    image_raw.value = True
    task_start.set()
    task_done.clear()
    task_done.wait()
    return(render_template('upload.html'))

#Developer tools age
@web.route('/dev')
def show_dev():
    return(render_template('dev.html'))

#Inference start withs the webserver, used only when manualy stopping and starting the inference
@web.route('/dev/start')
def start_parking():
    global p
    if p is None or not p.is_alive():
        p = parkingProcess(process_queue, stop_event, task_start, task_done, image_save, image_raw, json_reload, state_dict_reload, db)
        p.start()
        flash("Parking inference started")
    else:
        flash("Parking inference already running")
    return redirect('/dev')

#Unsafe, don't really recommend
@web.route('/dev/stop')
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
    return redirect('/dev')

##Main
if __name__ == '__main__':
    #Creates a multiprocessing object and runs it's loop
    process_queue = Queue()
    stop_event = Event()
    task_done = Event()
    task_start = Event()
    image_save.value = True
    p = parkingProcess(process_queue, stop_event, task_start, task_done, image_save, image_raw, json_reload, state_dict_reload, db)
    p.start()
    try:
        web.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False) #Web server launch
    except KeyboardInterrupt:
        p.stop() #Stops the inference loop
