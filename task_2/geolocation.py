from geopy.geocoders import Nominatim
from shapely.geometry import Polygon
from geohash2 import encode
import predict
import retrieve_maps

# # Inisialisasi GeoPy
geolocator = Nominatim(user_agent="geoapiExercises")

def geolocation_info(lat, lng):
    coordinate = (lat, lng)
    get_geohash = encode(coordinate[0], coordinate[1])
    exact_location = geolocator.reverse(coordinate, timeout=10)
    alamat = exact_location.address
    boundingbox = exact_location.raw['boundingbox']
    polygon = Polygon([[float(boundingbox[2]), float(boundingbox[0])], 
        [float(boundingbox[2]), float(boundingbox[1])], [float(boundingbox[3]), float(boundingbox[1])], 
        [float(boundingbox[3]), float(boundingbox[0])]])
    polygon_coordinates = [list(p) for p in polygon.exterior.coords]
    estimasi_luas = polygon.area

    image_map = retrieve_maps.get_map(coordinate)
    output_predict = predict.predict(image_map['map_image'])
    img_viz = output_predict['image_visualization']
    jumlah_kolam = output_predict['jumlah_kolam']
    return {'geohash':get_geohash, 'alamat':alamat, 'bbox_loc':boundingbox, 
    'polygon_coordinates':polygon_coordinates, 'estimasi_luas':estimasi_luas, 'jumlah_kolam':jumlah_kolam, 
    'image_visualization':img_viz}