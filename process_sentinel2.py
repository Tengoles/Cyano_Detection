from osgeo import gdal
from datetime import datetime
import numpy as np
import utils
import matplotlib.pyplot as plt
import os
import cv2

S2A_wavelengths = {"B1": 443, "B2": 492, "B3": 560, "B4": 665, "B5": 704, "B6": 740, "B7": 783, "B8": 833, 
                   "B8A": 865, "B9": 945, "B11": 1373, "B12": 1614, "B13": 2202}

S2B_wavelengths = {"B1": 442, "B2": 492, "B3": 559, "B4": 665, "B5": 704, "B6": 739, "B7": 780, "B8": 833, 
                   "B8A": 864, "B9": 943, "B11": 1377, "B12": 1610, "B13": 2186}

class day_data():
    def __init__(self, path):
        self._relevant_bands = ["B2", "B3", "B4", "B5", "B6", "B7"]
        #path to directory with acolite output
        self.data_path = path
        # datetime of captured data
        self.date = self._get_date()
        # get relevant bands as dictionary with keys as band name and values as np arrays
        self.bands = self._get_bands_data()
        # make display-ready rgb array from band data
        self.rgb = self._get_rgb_array()
        # make array with latitude and longitude for every pixel
        self.latitude, self.longitude = self._get_lat_lon()
    
    def _get_date(self):
        for file_name in os.listdir(self.data_path):
            if "MSI" in file_name:
                return datetime.strptime(file_name[8:27], '%Y_%m_%d_%H_%M_%S')
            
    def _get_bands_data(self):
        output = {}
        for file_name in os.listdir(self.data_path):
            try:
                band_wavelength = int(file_name.split("_")[-1].split(".")[0])
            except ValueError:
                continue
            if "S2A" in file_name:
                band_name = list(S2A_wavelengths.keys())[list(S2A_wavelengths.values()).index(band_wavelength)]
            elif "S2B" in file_name: 
                band_name = list(S2A_wavelengths.keys())[list(S2B_wavelengths.values()).index(band_wavelength)]
            else:
                continue
            if band_name in self._relevant_bands:
                tif_path = os.path.join(self.data_path, file_name)
                ds = gdal.Open(tif_path)
                band = ds.GetRasterBand(1)
                arr = band.ReadAsArray()
                output[band_name] = arr
        return output
                
    def _get_rgb_array(self):
        r = self.bands["B4"]
        g = self.bands["B3"]
        b = self.bands["B2"]
        
        r_stretched = utils.stretch_to_MinMax(r)
        g_stretched = utils.stretch_to_MinMax(g)
        b_stretched = utils.stretch_to_MinMax(b)
        
        rgb_stretched = np.dstack([r_stretched, g_stretched, b_stretched])
        rgb_stretched = rgb_stretched.astype('uint8')
        return rgb_stretched
    
    def _get_lat_lon(self):
        for file_name in os.listdir(self.data_path):
            if file_name.endswith("lat.tif"):
                tif_path = os.path.join(self.data_path, file_name)
                ds = gdal.Open(tif_path)
                band = ds.GetRasterBand(1)
                latitude = band.ReadAsArray()
            if file_name.endswith("lon.tif"):
                tif_path = os.path.join(self.data_path, file_name)
                ds = gdal.Open(tif_path)
                band = ds.GetRasterBand(1)
                longitude = band.ReadAsArray()
        #return np.array([latitude, longitude])
        return latitude, longitude
    
    def show_rgb(self):
        plt.figure(figsize=(10,10))
        plt.imshow(self.rgb)
        plt.show()
    
    def dominant_color(self, n_colors=3):
        pixels = np.float32(self.rgb.reshape(-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS

        _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
        _, counts = np.unique(labels, return_counts=True)
        
        dominant = palette[np.argmax(counts)]
        return dominant
    
    def get_pos_index(self, lat, lon):
        dist_array = np.zeros(self.lat_lon.shape[:2])
        for i, row in enumerate(dist_array):
            for j, dist in enumerate(row):
                dist_array[i, j] = np.linalg.norm(self.lat_lon[i,j] - np.array([lat, lon], dtype=np.float32))
        result = np.where(dist_array == np.amin(dist_array))
        return list(zip(result[0], result[1]))[0]
    
    def paint_coords(self, coords, color):
        for coord in coords:
            index = self.get_pos_index(coord[0], coord[1])
            self.rgb[index[0], index[1]] = color

def laguna_data_generator(start_date, end_date, date_format, data_path):
    start_datetime = datetime.strptime(start_date, date_format)
    end_datetime = datetime.strptime(end_date, date_format)

    data_directorys = sorted(os.listdir(data_path))
    data_directorys = [date for date in data_directorys if (not date.startswith(".") and
                                                            datetime.strptime(date, date_format) >= start_datetime and 
                                                            datetime.strptime(date, date_format) <= end_datetime)]
    for directory in data_directorys:
        print(directory)
        if "acolite_output" in os.listdir(os.path.join(data_path, directory)):
            try:
                instance = day_data(os.path.join(data_path, directory, "acolite_output"))
                yield instance
            except Exception as e:
                print("Error in %s: %s" % (directory, str(e)))
                yield None