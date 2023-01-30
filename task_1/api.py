import flask
from flask import Flask, render_template, request
import os
import predict
import time

app = Flask(__name__)

print("API READY !!!")

@app.route("/predict", methods=["POST"])
def detect():
    #if request.method == "POST":
    ts = time.time()
    ts_str = "{:0<20}".format(str(ts).replace(".",""))
    image = flask.request.files.get("image")
    filename = ts_str + ".jpg"
    if not os.path.exists("static/images/"):
        os.makedirs("static/images/")
    image.save(os.path.join("static/images", filename))
    out = predict.predict(os.path.join("static/images", filename))
    
    return(flask.jsonify(out), 200)


# @app.route("/")
# def index():
#     return render_template("index.html")

# if __name__ == "__main__":
#     app.run(debug=True)
