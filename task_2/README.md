# Task 2 (Shrimp Ponds Information)
This is a folder to manage shrimp ponds infomation. This project process the input the get the data of geolocation and imagery satellite from given coordinate

## Prerequisites
Before you begin, ensure you have met the following requirements:
* Python 3.6.5 environment

## Setup & Installations
To setup and install all required files, follow these steps:
1. Make sure you have installed the requirements needed for this project.
```
pip install -r requirements.txt
```

2. Download the weights before run the progran in this [link](https://drive.google.com/file/d/1-Q1zsJxs7XR37mN0bjsuLisaPzJim2st/view?usp=sharing)

3. Place the weights inside the `task_2/` directory

## How to Run
To run this project, follow these steps:
1. open a terminal inside the folder `task_2/`
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

5. input the name, island, langitude, and longitude coordinate