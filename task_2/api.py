import flask
from flask import Flask, render_template, request
import os
import geolocation as gl

app = Flask(__name__)

print("API READY !!!")

@app.route("/predict", methods=["POST", "GET"])
def detect():
    if request.method == "POST":
        lat = request.form.get("lat")
        lng = request.form.get("lng")

        name = request.form['name']
        island = request.form['island']

        print(lat, lng)

        out = gl.geolocation_info(float(lat), float(lng))
        return render_template("result.html", result=out, name=name, island=island, lat=lat, lng=lng)
    return render_template("index.html")

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
