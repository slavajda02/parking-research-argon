from os import listdir
from os.path import isfile, join
import json

import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ipywidgets import widgets, Dropdown, Box, Label, HBox, VBox

def pa_widget(img_path = "map.jpg", output_dir = ""):

    class Annotator(object):
        def __init__(self, axes):
            self.axes = axes

            self.xdata = []
            self.ydata = []
            
            self.annotation_dict = dict()
            self.image = ""
            self.img = []

        def mouse_click(self, event):
            if not event.inaxes:
                return
            if annotatation_flag.value == '1':
                x, y = event.xdata, event.ydata

                self.xdata.append(x)
                self.ydata.append(y)
                #If first click then dont't draw anything and just save cords
                try:
                    line = Line2D([self.xdata[-2], self.xdata[-1]], [self.ydata[-2], self.ydata[-1]])
                except:
                    return
                line.set_color('r')
                self.axes.add_line(line)

                #When clicking forth time, finish poly and clear cords
                if len(self.xdata) % 4 == 0:
                    self.annotation_dict[self.image].append([self.xdata.copy(), self.ydata.copy()])
                    self.xdata.append(self.xdata[-4])
                    self.ydata.append(self.ydata[-4])

                    line = Line2D([self.xdata[-2], self.xdata[-1]], [self.ydata[-2], self.ydata[-1]])
                    line.set_color('r')
                    self.axes.add_line(line)
                    plt.draw()

                    self.xdata, self.ydata = [], []
    #Returns image
    def load_image(path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if path not in metadata:
            metadata[0] = img.shape
        return img
    
    #Enables label painting with am annotation_flag
    def paint_button_clicked(b):
        if annotatation_flag.value == '0':
            annotatation_flag.value = '1'
            button_paint.style = {'button_color': 'red', 'color': 'white'}
        else:
            annotatation_flag.value = '0'
            button_paint.style = {'button_color': '#eeeeee'}

    #Removes last drawn line or square
    def trash_button_clicked(b):
        annotator.xdata, annotator.ydata = [], []
            
        if len(annotator.axes.lines) > 0:
            if len(annotator.axes.lines) % 4 == 0:
                annotator.annotation_dict[annotator.image].pop(-1)
                for i in range(4):
                    annotator.axes.lines[-1].remove()
            else:
                while(len(annotator.axes.lines) % 4 != 0):
                    annotator.axes.lines[-1].remove()
            axes.imshow(annotator.img)

    def download_button_clicked(b):
        print(annotator.annotation_dict)
        processed = []
        for polygon in annotator.annotation_dict[0]:
            reshaped_polygon = []
            points = []
            for i in range(len(polygon[0])):
                reshaped_polygon.append([round(polygon[0][i]), round(polygon[1][i])])
            processed.append(reshaped_polygon)
        with open(output_dir + "map.json", "w") as outfile:
            json.dump(processed, outfile)
       
        with open(output_dir + "metadata.json", "w") as outfile:
            json.dump(metadata, outfile)
        print("Map saved to map.json!")

    metadata = dict()

    selected_image = img_path

    annotatation_flag = widgets.Label(value='0')

    img = load_image(selected_image)
        
    fig, axes = plt.subplots(figsize=[16,9], num='Parking map creator v1.0')
    axes.imshow(img)
    plt.axis("off")
    fig.get_tight_layout()

    annotator = Annotator(axes)
    annotator.annotation_dict[0], annotator.image = [], 0
    annotator.img = img

    fig.canvas.mpl_connect('button_press_event', annotator.mouse_click)

    button_paint = widgets.Button(description="Annotate", style={'button_color': '#eeeeee'}, layout={'width': '80px'})
    button_trash = widgets.Button(description="Delete", layout={'width': '60px'})
    button_download = widgets.Button(description="Save", layout={'width': '60px'})

    tool_box = HBox([button_paint, button_trash, button_download])
    menu_box = VBox([tool_box])

    button_paint.on_click(paint_button_clicked)
    button_trash.on_click(trash_button_clicked)
    button_download.on_click(download_button_clicked)
    display(menu_box)