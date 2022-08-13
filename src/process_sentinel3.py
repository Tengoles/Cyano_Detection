import sys
import glob
from shapely.geometry import Polygon, Point
import os
import json
from datetime import datetime
import numpy as np
import cv2
import matplotlib.pyplot as plt

import settings
import snappy_utils
import utils
from mph import MPH

class OLCIdata():
    def __init__(self, path, mask_coordinates):        
        self.product_path = path
        
        self._product = snappy_utils.read_product(self.product_path)
        
        # product name from metadata
        self.pName = str(self._product.getMetadataRoot().getElement('history').getElement('SubsetInfo').getAttribute('SourceProduct.name').getData())
        # datetime of captured data
        self.date = datetime.strptime(self.pName.split("____")[1][0:15], '%Y%m%dT%H%M%S')
        
        self.quality_flags = snappy_utils.get_bands(self._product, ["quality_flags"])["quality_flags"]
        
        self.rgb = self._get_rgb_array()
        
        self.water_mask = utils.make_flags_mask(self.quality_flags, ["fresh_inland_water"])
        
        self.mph = self._get_mph()
        
        latitude = snappy_utils.get_bands(self._product, ["latitude"])["latitude"]
        longitude = snappy_utils.get_bands(self._product, ["longitude"])["longitude"]
        # make array of shape (len(lat), len(lon), 2)
        self.lat_lon = np.dstack([latitude, longitude])
        
        self.duplicated = utils.make_flags_mask(self.quality_flags, ["duplicated"])
        
        self.mask_coordinates = mask_coordinates
        
        self.laguna_mask = self._get_laguna_mask()
        
        # path to json with metadata
        data_index = [int(s) for s in os.path.basename(foo) if s.isdigit()][0]
        self.metadata_path = os.path.join(os.path.dirname(self.product_path), "metadata.json")
        # load metadata of day
        self.metadata = self._get_metadata()
    
    def _get_rgb_array(self):
        rgb_bands = ["Oa09_radiance", "Oa08_radiance", "Oa06_radiance",
                    "Oa14_radiance", "Oa04_radiance"]
        rgb_bands_dict = snappy_utils.get_bands(self._product, rgb_bands)
        trueColor_array = utils.enhanced_true_color(rgb_bands_dict["Oa09_radiance"], 
                        rgb_bands_dict["Oa08_radiance"], rgb_bands_dict["Oa06_radiance"],
                        rgb_bands_dict["Oa14_radiance"], rgb_bands_dict["Oa04_radiance"])
        trueColor_array = utils.histogram_equalization(utils.normalize_array(trueColor_array))
        return trueColor_array
    
    def _get_mph(self):
        mph_bands = ["Oa07_radiance", "Oa08_radiance", "Oa10_radiance",
                    "Oa11_radiance", "Oa12_radiance", "Oa18_radiance"]
        
        brrs_product = snappy_utils.apply_rayleigh_correction(self._product, mph_bands)
        brr_bands = ["rBRR_07", "rBRR_08", "rBRR_10", "rBRR_11", "rBRR_12", "rBRR_18"]
        brrs_arrays = snappy_utils.get_bands(brrs_product, brr_bands)
        
        return MPH(brrs_arrays)
    
    def show_rgb(self):
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
    
    def _get_laguna_mask(self):
        laguna_coords = []
        laguna_mask = self.water_mask.copy()
        selected_coords_poly = Polygon(self.mask_coordinates)
        for i, row in enumerate(self.lat_lon):
            for j, val in enumerate(row):
                point = self.lat_lon[i,j].tolist()
                point = [point[1], point[0]]
                point = Point(point)
                if selected_coords_poly.contains(point):
                    laguna_mask[i,j] = True
                else:
                    laguna_mask[i,j] = False
        return laguna_mask
    
    def save_geotiff(self, path):
        #Algunas opciones de p_type: "GeoTiff":.tif y "BEAM-DIMAP":.dim
        snappy_utils.write_product(self._product, path, "GeoTiff")
    
    def _get_metadata(self):
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path) as f:
                return json.load(f)
        else:
            return {}

class OLCIdataGenerator():
    def __init__(self, data_path, date_format, start_date=None, end_date=None, dates_list=None, 
                     skip_invalid=False, tagging=False, cloud_level_th=0, mask_coordinates=None):
        self.date_format = date_format
        self.cloud_level_th = cloud_level_th
        if start_date != None and end_date != None:
            self.start_datetime = datetime.strptime(start_date, date_format)
            self.end_datetime = datetime.strptime(end_date, date_format)
        self.dates_list = dates_list
        self.data_path = data_path
        self.skip_invalid = skip_invalid
        self.tagging_mode = tagging
        self.mask_coordinates = mask_coordinates
        self.data_directories = self._get_data_directories()
            
        
    def _get_data_directories(self):
        data_directories = sorted(os.listdir(self.data_path))
        
        if self.dates_list == None:
            data_directories = [date for date in data_directories if (not date.startswith(".") and
                                                                datetime.strptime(date, self.date_format) >= self.start_datetime and 
                                                                datetime.strptime(date, self.date_format) <= self.end_datetime)]
        else:
            data_directories = [date_str for date_str in data_directories if date_str in self.dates_list]
        
        data_directories_temp = []
        for i, directory in enumerate(data_directories):
            metadata_path = os.path.join(self.data_path, directory, "OLCI", "metadata.json")
            # days without metadata are ignored
            if os.path.exists(metadata_path):
                with open(metadata_path) as f:
                    metadata =  json.load(f)
                if self.skip_invalid:
                    if int(metadata["cloud level"]) <= self.cloud_level_th:
                        data_directories_temp.append(directory)
                else:
                    data_directories_temp.append(directory)
            elif self.tagging_mode:
                data_directories_temp.append(directory)
                    
        data_directories = data_directories_temp
                
        return data_directories
    
    def __iter__(self):
        for directory in self.data_directories:                
            try:
                day_data_paths = glob.glob(os.path.join(self.data_path, directory, "OLCI")+"/subselection_*.dim")
                for d in day_data_paths:
                    instance = OLCIdata(day_data_path, self.mask_coordinates)
                    yield instance
            except Exception as e:
                print("Error in %s: %s" % (directory, str(e)))
                continue
    
    def __len__(self):
        return len(self.data_directories)