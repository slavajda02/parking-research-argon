import app
import queue
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
from flask import render_template, flash, redirect, request
from datetime import datetime
import pandas as pd
import json
from werkzeug.utils import secure_filename
import os

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.ALLOWED_EXTENSIONS

#Gets all of the data so far and puts them in a list
def fetch_database():
    data = app.collection.find()
    data_list = []
    for item in data:
        data_list.append(item)
    return data_list


##Flask routes and functions
#Index page
@app.web.route('/')
def index():
    return render_template('index.html')

#Parking database and histogram
@app.web.route('/parking')
def show_parking():
    info = ""
    if app.p is not None:
        try:
            status = app.p.queue.get_nowait()
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

@app.web.route('/history')
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
    fig = px.histogram(occupancy_df, x='Time', y='Number of cars', title="Parking occupancy over time", template='plotly_dark', nbins=100000)
    fig.update_layout(bargap=0.2)
    
    #fig.update_layout(xaxis=dict(range=[occupancy_df.index[occupancy_df.shape(0)-1440], occupancy_df.index[occupancy_df.shape(0)]]))
    graphJson = json.dumps(fig, cls=PlotlyJSONEncoder)
    return render_template('history.html', graphJSON=graphJson)
    
@app.web.route('/upload', methods=['GET', 'POST'])
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
                file.save(os.path.join(app.web.config['UPLOAD_FOLDER'], "map.json"))
                app.json_reload.value = True
            else:
                file.save(os.path.join(app.web.config['UPLOAD_FOLDER'], "state_dict_final.pth"))
                app.state_dict_reload.value = True
            flash('File uploaded')
            app.image_raw.value = True
            app.task_start.set()
            app.task_done.clear()
            app.task_done.wait()
            return redirect(request.url)
        else:
            flash('Invalid file!')
            return redirect(request.url)
    app.image_raw.value = True
    app.task_start.set()
    app.task_done.clear()
    app.task_done.wait()
    return(render_template('upload.html'))

#Developer tools age
@app.web.route('/dev')
def show_dev():
    return(render_template('dev.html'))

#Inference start withs the webserver, used only when manualy stopping and starting the inference
@app.web.route('/dev/start')
def start_parking():
    if app.p is None or not app.p.is_alive():
        app.p = app.parkingProcess(app.process_queue, app.stop_event, app.task_start, app.task_done, app.image_save, app.image_raw, app.json_reload, app.state_dict_reload, app.db)
        app.p.start()
        flash("Parking inference started")
    else:
        flash("Parking inference already running")
    return redirect('/dev')

#Unsafe, don't really recommend
@app.web.route('/dev/stop')
def stop_parking():
    if app.p is not None and app.p.is_alive():
        app.p.stop()
        app.p.join()
        app.p = None
        app.stop_event.clear()
        flash("Parking inference stopped")
    else:
        flash("Parking inference not running")
    return redirect('/dev')