# Task 3 (Vibrio Detection)
This is a folder to manage vibrio detection project. This project process the image of vibrio from a microsope, then uses object detection to count all vibrio that detected

## Prerequisites
Before you begin, ensure you have met the following requirements:
* Python 3.6.5 environment

## Setup & Installations
To setup and install all required files, follow these steps:
1. Make sure you have installed the requirements needed for this project.
```
pip install -r requirements.txt
```

2. Download the weights before run the progran in this [link](https://drive.google.com/file/d/1chNf6ih1GEHEleGGhjqrMNQWO3eypxUU/view?usp=sharing)

3. Place the weights inside the `task_3/` directory

## How to Run
To run this project, follow these steps:
1. open a terminal inside the folder `task_3/`
2. type this command
```
export FLASK_ENV=development
export FLASK_APP=api.py
```
3. Then, run the flask by typing this command
```
flask run
```

4. Open this URL `http://127.0.0.1:5000/` at your browser

5. Upload the image, the you will get the result