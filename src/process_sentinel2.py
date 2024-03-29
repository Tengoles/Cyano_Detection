from osgeo import gdal
from datetime import datetime
import numpy as np
from shapely.geometry import Polygon, Point
import utils
import matplotlib.pyplot as plt
import pytz
import os
import json
import cv2
from tqdm import tqdm
from s2cloudless import S2PixelCloudDetector, CloudMaskRequest, get_s2_evalscript

S2A_wavelengths = {"B1": 443, "B2": 492, "B3": 560, "B4": 665, "B5": 704, "B6": 740, "B7": 783, "B8": 833, 
                   "B8A": 865, "B9": 945, "B10": 1373, "B11": 1614, "B12": 2202}

S2B_wavelengths = {"B1": 442, "B2": 492, "B3": 559, "B4": 665, "B5": 704, "B6": 739, "B7": 780, "B8": 833, 
                   "B8A": 864, "B9": 943, "B10": 1377, "B11": 1610, "B12": 2186}

class DayData():
    def __init__(self, path, rhot=False, timezone_location="America/Montevideo"):
        self._relevant_bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
        #path to directory with acolite output
        self.data_path = path
        # datetime of captured data
        self.tz = pytz.timezone(timezone_location)
        self.date = self._get_date()
        # get relevant bands as dictionary with keys as band name and values as np arrays
        self.bands = self._get_bands_data()
        if rhot == True:
            self.bands_rhot = self._get_bands_data(rhot=True)
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
                return self.tz.localize(datetime.strptime(file_name[8:27], '%Y_%m_%d_%H_%M_%S'))
            
    def _get_array_from_tif(self, tif_path):
        ds = gdal.Open(tif_path)
        band = ds.GetRasterBand(1)
        arr = band.ReadAsArray()
        return arr
    
    def _get_bands_data(self, rhot=False):
        output = {}
        rhos_tiffs = [f for f in os.listdir(self.data_path) if ("rhos" in f) and f.endswith(".tif")]
        rhot_tiffs = [f for f in os.listdir(self.data_path) if ("rhot" in f) and f.endswith(".tif")]
        if rhot:
            list_of_tiffs = rhot_tiffs
        else:
            list_of_tiffs = rhos_tiffs
        for file_name in list_of_tiffs:
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
            if (band_name in self._relevant_bands):
                tif_path = os.path.join(self.data_path, file_name)
                arr = self._get_array_from_tif(tif_path)
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
        latitude = np.array([])
        longitude = np.array([])
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
        lat_distances = np.abs(self.latitude - lat)
        lon_distances = np.abs(self.longitude - lon)
        all_distances = np.sqrt(lat_distances + lon_distances)
        result = np.where(all_distances == np.amin(all_distances))
        return list(zip(result[0], result[1]))[0]
    
    def paint_coords(self, coords, color, radius=3):
        for coord in coords:
            index = self.get_pos_index(coord[0], coord[1])
            center_coordinates = (index[1], index[0])
            self.rgb = cv2.circle(self.rgb, center_coordinates, radius, color, -1)
            #self.rgb[index[0], index[1]] = color
    
    def get_NDCI(self):
        ndci = (self.bands["B5"] - self.bands["B4"])/(self.bands["B5"] + self.bands["B4"])
        return ndci
    
    def _get_metadata(self):
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path) as f:
                return json.load(f)
        else:
            return {}
        
    def compute_cloud_mask(self, threshold=0.4, average_over=4, dilation_size=2):
        cloud_detector = S2PixelCloudDetector(
                                        threshold=threshold,
                                        average_over=average_over,
                                        dilation_size=dilation_size,
                                        all_bands=False
                                    )
        bands_for_cloud_detector = ["B1", "B2", "B4", "B5", "B8", "B8A", "B9", "B10", "B11", "B12"]
        cloud_detector_input = np.zeros((self.rgb.shape[0], self.rgb.shape[1], len(bands_for_cloud_detector)))
        for i, band in enumerate(bands_for_cloud_detector):
            cloud_detector_input[:, :, i] = self.bands_rhot[band]
        
        return cloud_detector.get_cloud_masks(cloud_detector_input)
        

class DayDataGenerator():
    def __init__(self, start_date, end_date, date_format, data_path, skip_invalid=False, tagging=False, cloud_level_th=0):
        self.date_format = date_format
        self.cloud_level_th = cloud_level_th
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
            msi_directory = os.path.join(self.data_path, directory, "MSI")
            if os.path.exists(msi_directory):
                 if len(os.listdir(msi_directory)) < 5:
                    continue
            else:
                continue
            metadata_path = os.path.join(msi_directory, "metadata.json")
            # days without metadata are ignored
            if os.path.exists(metadata_path):
                with open(metadata_path) as f:
                    metadata =  json.load(f)
                if self.skip_invalid:
                    if int(metadata["cloud level"]) <= self.cloud_level_th:
                        data_directories_temp.append(msi_directory)
                else:
                    data_directories_temp.append(msi_directory)
            elif self.tagging_mode:
                data_directories_temp.append(msi_directory)
                    
        data_directories = data_directories_temp
                
        return data_directories
    
    def __iter__(self):
        for directory in self.data_directories:                
            try:
                instance = DayData(directory)
                yield instance
            except Exception as e:
                print("Error in %s: %s" % (directory, str(e)))
                continue
    
    def __len__(self):
        return len(self.data_directories)
        

class Mask():
    def __init__(self, masks_dir, resize_ratio=1):
        self.resize_ratio = resize_ratio
        self.dir = masks_dir
        self.array, self.polygon, self.height, self.width, self.annotations = self._load_masks()
        self.rgb = self.make_rgb()
        
    def _load_json(self, path):
        with open(path) as f:
            return json.load(f)
            
    def _load_masks(self):
        masks = [self._load_json(os.path.join(self.dir, fn)) for fn in os.listdir(self.dir) if fn.endswith(".json")]
        height = masks[0]["height"]*self.resize_ratio
        width = masks[0]["width"]*self.resize_ratio
        output = np.zeros((height, width), dtype=np.uint8)
        for mask in masks:
            mask_absolute_coords = np.array([[p['x'], p['y']] for p in mask["valid water"]["points"]])
            mask_polygon = Polygon(mask_absolute_coords)
            mask_absolute_coords = mask_absolute_coords.reshape((-1,1,2))
            cv2.fillPoly(output, [mask_absolute_coords], 1)
        return output, mask_polygon, height, width, masks
    
    def make_rgb(self):
        mask_rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        mask_rgb[:, :, 0] = self.array
        mask_rgb[:, :, 1] = self.array
        mask_rgb[:, :, 2] = self.array
        
        return mask_rgb*255
    
    def display_mask_img(self, img):
        output = np.copy(img)
        output[self.array == 1] = [255, 255, 255]
        return output
    
    def display_mask_contour(self, img):
        for mask in self.annotations:
            pts = np.array([[p['x'], p['y']] for p in mask["valid water"]["points"]], np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(img,[pts],True, (0,255,255))        
        return img
    
    def get_pixel_count(self):
        unique, counts = np.unique(self.array, return_counts=True)
        unique_counts = dict(zip(unique, counts))
        pixel_count = unique_counts[True]
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




    