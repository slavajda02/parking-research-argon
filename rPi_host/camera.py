from picamera2 import Picamera2
from time import sleep
from datetime import datetime
import os

picam2 = Picamera2()

camera_config = picam2.create_still_configuration({"size" : (3200, 1800)})
picam2.configure(camera_config)
picam2.start()
sleep(2)
n = 100


if not os.path.isdir("images"):
    os.mkdir("images")

for x in range(n):
    time = datetime.now().strftime('%Y-%m-%d %H_%m_%S')
    picam2.capture_file("images/" + time + ".jpg")
    print("Sucessfully captured image in:", time)
    print(f'{x+1} of {n}')
    sleep(600)
