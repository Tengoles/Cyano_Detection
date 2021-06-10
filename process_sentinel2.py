from osgeo import gdal
from datetime import datetime
import numpy as np
from shapely.geometry import Polygon, Point
import utils
import matplotlib.pyplot as plt
import os
import json
import cv2
from tqdm import tqdm

S2A_wavelengths = {"B1": 443, "B2": 492, "B3": 560, "B4": 665, "B5": 704, "B6": 740, "B7": 783, "B8": 833, 
                   "B8A": 865, "B9": 945, "B11": 1373, "B12": 1614, "B13": 2202}

S2B_wavelengths = {"B1": 442, "B2": 492, "B3": 559, "B4": 665, "B5": 704, "B6": 739, "B7": 780, "B8": 833, 
                   "B8A": 864, "B9": 943, "B11": 1377, "B12": 1610, "B13": 2186}

class DayData():
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
        # path to json with metadata
        self.metadata_path = os.path.join(self.data_path, "metadata.json")
        # load metadata of day
        self.metadata = self._get_metadata()
    
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
    
    def get_NDCI(self):
        ndci = (self.bands["B5"] - self.bands["B4"])/(self.bands["B5"] + self.bands["B4"])
        return ndci
    
    def _get_metadata(self):
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path) as f:
                return json.load(f)
        else:
            return {}

class DayDataGenerator():
    def __init__(self, start_date, end_date, date_format, data_path, skip_invalid=False, tagging=False):
        self.date_format = date_format
        self.start_datetime = datetime.strptime(start_date, date_format)
        self.end_datetime = datetime.strptime(end_date, date_format) 
        self.data_path = data_path
        self.skip_invalid=skip_invalid
        self.tagging_mode = tagging
        self.data_directories = self._get_data_directories()
        
    def _get_data_directories(self):
        data_directories = sorted(os.listdir(self.data_path))
        data_directories = [date for date in data_directories if (not date.startswith(".") and
                                                                datetime.strptime(date, self.date_format) >= self.start_datetime and 
                                                                datetime.strptime(date, self.date_format) <= self.end_datetime)]
        
        data_directories_temp = []
        for i, directory in enumerate(data_directories):
            metadata_path = os.path.join(self.data_path, directory, "acolite_output", "metadata.json")
            # days without metadata are ignored
            if os.path.exists(metadata_path):
                with open(metadata_path) as f:
                    metadata =  json.load(f)
                if self.skip_invalid:
                    if metadata["valid"] == True:
                        data_directories_temp.append(directory)
                else:
                    data_directories_temp.append(directory)
            elif self.tagging_mode == True:
                data_directories_temp.append(directory)
                    
        data_directories = data_directories_temp
                
        return data_directories
    
    def __iter__(self):
        for directory in self.data_directories:                
            try:
                day_data_path = os.path.join(self.data_path, directory, "acolite_output")
                instance = DayData(day_data_path)
                yield instance
            except Exception as e:
                print("Error in %s: %s" % (directory, str(e)))
                continue
    
    def __len__(self):
        return len(self.data_directories)
        

class Mask():
    def __init__(self, mask_path, width, height):
        self.width = width
        self.height = height
        self.path = mask_path
        self.annotation = self._load_json()
        self.array, self.polygon = self._load_mask()
        self.rgb = self.make_rgb()
        
    def _load_json(self):
        with open(self.path) as f:
            return json.load(f)
            
    def _load_mask(self):
        #output = np.zeros((self.width, self.height), dtype=np.uint8)
        output = np.zeros((self.height, self.width), dtype=np.uint8)

        with open(self.path) as f:
            data = json.load(f)

        mask_absolute_coords = [[p['x']*self.width, p['y']*self.height] for p in data["valid water"]["relative points"]]
        mask_polygon = Polygon(mask_absolute_coords)
        
        mask_rectangle = mask_polygon.minimum_rotated_rectangle
        mask_bounding_coords = list(mask_rectangle.exterior.coords)
        
        pbar = tqdm(total=self.height*self.width)
        for i in range(self.height):
            for j in range(self.width):                
                if mask_polygon.contains(Point([j, i])):
                    output[i, j] = 255
                        
                pbar.update(1)
        pbar.close()
        return output, mask_polygon
    
    def make_rgb(self):
        mask_rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        mask_rgb[:, :, 0] = self.array
        mask_rgb[:, :, 1] = self.array
        mask_rgb[:, :, 2] = self.array
        
        return mask_rgb
    
    def display_mask_img(self, img):
        output = np.zeros_like(img)
        for i, row in enumerate(img):
            for j, rgb in enumerate(row):
                if self.array[i,j] == 255:
                    output[i,j,:] = np.array([255, 255, 255])
                else:
                    output[i,j,:] = img[i, j, :]

        return output
    
    def display_mask_contour(self, img):
        pts = np.array([[p['x']*self.width, p['y']*self.height] for p in self.annotation["valid water"]["relative points"]], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img,[pts],True, (0,255,255))
        
        return img                        
    
    def get_pixel_count(self):
        pixel_count = 0
        for i in range(self.height):
            for j in range(self.width):
                if self.array[i, j] == 255:
                    pixel_count += 1
        return pixel_count
        
    def reduce_mask(self, method="skip"):
        #output = np.zeros_like(self.array)
        if method == "skip":
            for i in range(self.height):
                for j in range(self.width):
                    if j%2 == 0:
                        #output[i, j] = 255
                        self.array[i, j] = 0
                    if i%2 == 0:
                        #output[i, j] = 255
                        self.array[i, j] = 0
        #return output

if __name__ == "__main__":
    import settings
    
    DATA_PATH = os.path.join("sample_data", "2021-01-25")
    DATE_FORMAT = '%Y-%m-%d'
    START_DATE = '2016-12-21'
    END_DATE = '2021-04-20'
    
    sample_day = day_data(os.path.join(settings.data_path, "2021-01-25"))
    data_generator = laguna_data_generator(START_DATE, END_DATE, DATE_FORMAT, DATA_PATH)

    mask_path = "water_mask.json"
    width, height , _ = sample_day.rgb.shape

    mask = load_mask(mask_path, width, height)
    mask_display = np.zeros_like(sample_day.rgb, dtype=np.uint8)
    mask_display[:, :, 0] = mask
    mask_display[:, :, 1] = mask
    mask_display[:, :, 2] = mask

    dst = cv2.addWeighted(mask_display, 0.2, sample_day.rgb, 0.9, 128)
    
    with open(mask_path) as f:
        data = json.load(f)
    abs_pts = np.array([[p['x']*width, p['y']*height] for p in data["valid water"]["relative points"] ], np.int32)
    abs_pts = abs_pts.reshape((-1,1,2))
    cv2.polylines(dst,[abs_pts],True, (0,255,255))
    
    cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("mask", 1000, 1000)
    cv2.imshow("mask", dst)
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()




    