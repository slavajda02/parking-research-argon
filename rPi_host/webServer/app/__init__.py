from flask import Flask, render_template, flash, redirect, url_for, request
from app.ArgonPark.argonPark import *
from flask_pymongo import PyMongo
from config import Config


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

from app.ArgonPark.argonPark import *
from app.process import *
from app import routes

#Flags
process_queue = Queue()
stop_event = Event()
task_done = Event()
task_start = Event()

#Subprocess start
p = parkingProcess(process_queue, stop_event, task_start, task_done, image_save, image_raw, json_reload, state_dict_reload, db)
