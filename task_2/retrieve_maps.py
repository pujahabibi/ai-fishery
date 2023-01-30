import requests
import time
import os

# Your Google Maps API key
API_KEY = 'AIzaSyAqpKaOnjKHYLlpqF0bA_zwZApIn0YXYZM'


def get_map(coordinate):
    lat = coordinate[0]
    lng = coordinate[1]

    ts = time.time()
    ts_str = "{:0<20}".format(str(ts).replace(".",""))

    zoom = 16

    size = "1000x1000"

    # URL for the Static Maps API
    url = f"https://maps.googleapis.com/maps/api/staticmap?center={lat},{lng}&zoom={zoom}&size={size}&maptype=satellite&key={API_KEY}"

    # Send a GET request to the API
    response = requests.get(url)

    if not os.path.exists("static/images/"):
        os.makedirs("static/images/")

    with open(os.path.join("static/images", ts_str+".png"), "wb") as f:
        f.write(response.content)

    return {'map_image':os.path.join("static/images", ts_str+".png")}