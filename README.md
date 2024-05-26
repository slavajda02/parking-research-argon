# Detection of parking space availability based on video
This is a repo that contains the scripts for *''Detection of parking space availability based on video''* thesis and bachelor work.

## This repo now consists of two parts!
1. **Model training and testing scripts and tools**
    - Contains tools to create your own dataset, train your selected network and run tests.
2. **Flask WebServer application for occupancy detection**
    - Consists of a web application using ArgonPark library to evaluate free parking spaces on a parking lot


## Introduction

This repo originated as a fork of a repository from [*''Revising deep learning methods in parking lot occupancy detection''*](https://arxiv.org/abs/2306.04288). I greatly thank the authors for their work. And I recommend checking the repository along with their paper, their reasearch helped me significantly.

The authors are:
Anastasia Martynova, [Mikhail Kuznetsov](https://github.com/mmkuznecov), [Vadim Porvatov](https://www.researchgate.net/profile/Vadim-Porvatov), Vladislav Tishin, [Natalia Semenova](https://www.researchgate.net/profile/Natalia-Semenova-7).

And their repo is: [parking-reseach](https://github.com/Eighonet/parking-research)

The original goal of this fork is to port the tools to newest python libraries and to serve as a personal playground  for the work. But after a lot of changes and modifications to the code. It is now used as a proof of concept to the final work.

The training and testing scripts support only object recognition models for now.
Scripts will work with datasets used in Parking-Research as they use the same format.

## What has been added / changed
- I written a training script that lets the user choose how to train a model with the dataset format from their work as both as a training script that uses the testing images from the dataset to test the trained model.
- The creation of your own dataset was a bit simplified and reworked. Consult the README located in [annotating](annotating/) directory.
- All of the scripts are able to run on both CPU and GPU.
- Tested on Python 3.11
- Created a new dataset T10LOT shot on multiple cameras
- ArgonPark library

## Downloads
[Images containing testing results with trained models and T10LOT dataset](https://drive.google.com/drive/folders/1Jvvc7PKZTQi63PJnOjMKW9x3qeNipSYl?usp=drive_link)


## Prerequisites

To run the code, you need to install the requirements using the following command:

```bash
pip install -r requirements.txt
```

Alternatively, you can create and activate the conda enviroment:

```bash
conda env create -f environment.yml
conda activate parking-research
```

The trainig script is setup to log information to [Comet](https://comet.com).
The training script looks for an `pi.key`` file containing a key that you can obtain in your user settings after registering.


# Part one: **Models/~**
## Training
Create a `datasets` directory containing the datasets in the same location as the training script. Add an `api.key` file containing your comet API key next to your `datasets` folder. Then run the script and follow to onscreen prompts:
```bash 
python train.py
```
For an average size dataset you can use the default values for learning speed, batch size and number of epochs. Always check the validation batch progress to make sure that the model is not overfitting! 

Every training is run with a different seed for the random number generator, this will make it so that no model is the same. To get rid of this, set a permanent seed in the setting dictionary instead of the one geneerated from a current time:
```python
settings = {
    "batch_size" : int(answers["batch"]),
    "learning_rate": float(answers["rate"]),
    "model_type" : answers["model"],
    "seed" : int(datetime.now().timestamp()), #Set the seed here
    "save_rate" : int(answers["save_rate"]),
    "pretrained" : answers["pretrained"],
    "dataframe" : "datasets/"+datasets[0]+"/"+datasets[0]+"/"+datasets[0]+"_dataframe.csv",
    "path": "datasets/"+datasets[0]+'/'+datasets[0]+'/',
    "epochs" : int(epochs[0])
}
```

## Running tests
After training a model you can test it using the `test.py` script. Run it and follow the prompts:
```bash 
python test.py
```
The testing function can print out the time of one inference and save all of the images that were tested.


# Part two: **rPi_host/~**

## ArgonPark library

This library was made to utilize the trained model from previous steps with an algorithm to evaluate occupancy on a dedicated parking lot.
It acts as a base for a raspberry pi host script that has a web interface which interfaces with this library and its methods.
[Documentation](https://slavajda02.github.io/parking-research-argon/) for this library is available on githubs pages.

## Webserver
### How to:
1. cd into the ```rPi_host/webServer``` directory
2. populate the ```upload/``` directory with ```state_dict_final.pth``` obtained from the training script
3. run: 

```bash
python server.py 
```

This should launch a developement webserver running on an ip adress that will be printed out into the terminal. The home page should show the current raspberry Pi camera view with plotted parking spaces.
To create a new map, head to Tools->Map, download the raw image and use the jupyter script in ```app/ArgonPark/map_Creator``` to create a new ```map.json``` which can then be uploaded straight from the site. After the upload, images should refresh with the upated information.

## Database setup
The subprocess of this application sends the curent status of all parking space alongside with aditional information to a MongoDB database configured in the ```___init___.py```. Which is now connecting to an Atlas service. You can reconfigure the database to connect to.

# Contact me

If you have some questions about the code, you are welcome to open an issue or contact me through an email or through social link on my GitHub profile. Please do not contact the original authors with questions regarding this fork.

# License

Established code released as open-source software under the MIT license.

# Citation

```
@misc{martynova2023revising,
      title={Revising deep learning methods in parking lot occupancy detection}, 
      author={Anastasia Martynova and Mikhail Kuznetsov and Vadim Porvatov and Vladislav Tishin and Andrey Kuznetsov and Natalia Semenova and Ksenia Kuznetsova},
      year={2023},
      eprint={2306.04288},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
@misc{kuzelaParkDetection,
      title={Detection of parking space availability based on video},
      author={Miloslav Kužela and Tomáš Frýza}
      year={2024}
}
```
