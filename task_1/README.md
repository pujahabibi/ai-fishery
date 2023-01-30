# Task 1 (Document OCR)
This is a folder to manage Document OCR. This project extract all text from a document using deep leaning

## Prerequisites
Before you begin, ensure you have met the following requirements:
* Python 3.6.5 environment

## Setup & Installations
To setup and install all required files, follow these steps:
1. Make sure you have installed the requirements needed for this project.
```
pip install -r requirements.txt
```

2. Download the weights before run the progran in this [link](https://drive.google.com/file/d/1LCOC8g0bfI8FbNG312GdQHL1lmlovvkH/view?usp=sharing)

3. Place the weights inside the `task_1/` directory

## How to Run
To run this project, follow these steps:
1. open a terminal inside the folder `task_1/`
2. run gunicorn
```
gunicorn -w 1 -t 600 -b 127.0.0.1:8811 api:app
```
3. Then, open Postman and copy this URL `http://127.0.0.1:8811`

4. Choose POST

5. Choose Body below the input URL

6. set a key name called `image` and the input is File