import os
import os.path as osp
from collections import OrderedDict
from datetime import datetime
import pytz
import random
import numpy as np

import snappy_utils

class LST():
    """
    Find directories with date as name inside data_path and make a generator to iterate over 
    LST in those directories when possible
    """
    def __init__(self, product_path, wkt_footprint, timezone_location="America/Montevideo"):
        self.tz = pytz.timezone(timezone_location)
        
        self.product_path = product_path
        
        self.wkt_footprint = wkt_footprint
        
        self._product = self._load_product()
        
        self.lst = self._get_lst()
        
        self.latitude = snappy_utils.get_bands(self._product, ["latitude_in"])["latitude_in"]
        self.longitude = snappy_utils.get_bands(self._product, ["longitude_in"])["longitude_in"]
        
        self.capture_datetime = self._get_capture_datetime()
        
        self.water_mask = self._get_water_mask()
        
        self.cloud_mask = self._get_cloud_mask()
        
        self.valid_pixels_mask = ~self.cloud_mask*self.water_mask
        
        self.valid_pixels_temp = self.valid_pixels_mask*self.lst
        
        self.water_temperature = self.valid_pixels_temp[np.nonzero(self.valid_pixels_temp)].mean()
        
    def _get_lst(self):
        lst = snappy_utils.get_bands(self._product, ["LST"])["LST"]
        lst = lst - 273.15
        
        return lst
    
    def _get_capture_datetime(self):
        product_capture_datetime = datetime.strptime(os.path.basename(self.product_path).split("____")[1].split("_")[0][0:15], 
                                                                 "%Y%m%dT%H%M%S")
        local_capture_datetime = self.tz.localize(product_capture_datetime)
        
        return local_capture_datetime
        
    def _load_product(self):
        return snappy_utils.make_subset(snappy_utils.read_product(self.product_path), self.wkt_footprint)
        
    def _get_cloud_mask(self):
        """
        https://forum.step.esa.int/t/sentinel-3-slstr-level-2-lst-problem-with-clouds-and-temperature-amplitude/22551/3
        """
        bayes_in = snappy_utils.get_bands(self._product, ["bayes_in"])["bayes_in"]
        bayes_in[bayes_in == 2] = 1
        
        return bayes_in.astype(bool)
        
    def _get_water_mask(self):
        biomes_mask = snappy_utils.get_bands(self._product, ["biome"])["biome"]
        biomes_mask[biomes_mask != 26] = 0
        return biomes_mask.astype(bool)
        
    def get_pos_index(self, lat, lon):
        lat_distances = np.abs(self.latitude - lat)
        lon_distances = np.abs(self.longitude - lon)
        all_distances = np.sqrt(lat_distances**2 + lon_distances**2)
        result = np.where(all_distances == np.amin(all_distances))
        return list(zip(result[0], result[1]))[0]
    
    def make_mask_from_coords(self, coords: list):
        output = np.zeros_like(self.lst)
        for coord in coords:
            lat = coord[0]
            lon = coord[1]
            i, j = self.get_pos_index(lat, lon)
            output[i, j] = 1
        return output



        

class LSTDataset():
    def __init__(self, data_path, wkt_footprint, timezone_location="America/Montevideo"):
        self.wkt_footprint = wkt_footprint
        
        self.data_path = data_path
        
        self.tz = pytz.timezone(timezone_location)
        
        self.data_dirs = self._get_data_dirs()
    
    def get_day_data(self, date):
        output = []
        errors = []
        for dt, product_path in self.data_dirs.items():
            if dt.date() == date:
                try:
                    lst = LST(product_path, self.wkt_footprint)
                    output.append(lst)
                except Exception as e:
                    errors.append([product_path, str(e)])
                    continue
        return output, errors
        
    def _get_data_dirs(self):
        data_dirs = OrderedDict()
        for sentinel_directory in sorted(os.listdir(self.data_path)):
            try:
                directory_date = datetime.strptime(sentinel_directory, "%Y-%m-%d").date()
            except Exception as e:
                print("Exception converting directory to datetime:", str(e))
                continue
                
            for p in os.listdir(osp.join(self.data_path, sentinel_directory)):
                if "LST" in p and p.endswith("SEN3"):
                    try:
                        product_capture_datetime = datetime.strptime(p.split("____")[1].split("_")[0][0:15], 
                                                                 "%Y%m%dT%H%M%S")
                        product_capture_datetime = self.tz.localize(product_capture_datetime)
                    except Exception as e:
                        print("Exception converting directory to datetime:", str(e))
                        print(osp.join(sentinel_directory, p))
                        continue
                    
                    data_dirs[product_capture_datetime] = os.path.join(self.data_path, 
                                                                       sentinel_directory,
                                                                        p)                
        return data_dirs
    
    def get_random_data(self):
        output = []
        
        dates = list(self.data_dirs.keys())
        date = random.choice(dates)
        output.append(date)
        
        product_path = self.data_dirs[date]        
        
        lst = LST(product_path, self.wkt_footprint)
        
        return lst
    
    def __iter__(self):
        for capture_datime, lst_path in self.data_dirs.items(): 
            try:
                lst = LST(lst_path, self.wkt_footprint)
                yield lst
            except Exception as e:
                print("Error in %s: %s" % (lst_path, str(e)))
                continue
    
    def __len__(self):
        return len(list(self.data_dirs.keys()))
    
def find_best_temperature_candidate(temps):
    best_candidates = []
    for i, t in enumerate(temps):
        cond1 = np.count_nonzero(t.valid_pixels_mask) > 10
        cond2 = t.water_temperature > 5
#         if cond1 and cond2:
#             best_candidates.append(t)
        ####
#         cond3 = t.capture_datetime.hour > 9
#         cond4 = t.capture_datetime.hour < 19
#         if cond1 and cond2 and cond3 and cond4:
#             best_candidates.append(t)
        ####
        cond3 = t.capture_datetime.hour > 0
        cond4 = t.capture_datetime.hour > 19
        cond5 = t.capture_datetime.hour < 9
        if cond1 and cond2 and ((cond3 and cond5) or cond4):
            best_candidates.append(t)
        
    if len(best_candidates) == 0: return None

    if len(best_candidates) == 1: return best_candidates[0]

    current_best_candidate = [None, 0]
    for bc in best_candidates:
        bc_valid_pixels = np.count_nonzero(t.valid_pixels_mask)
        if bc_valid_pixels > current_best_candidate[1]:
            current_best_candidate[0] = bc
            current_best_candidate[1] = bc_valid_pixels
    return current_best_candidate[0]

def filter_valid_temps(temps):
    output = []
    for i, t in enumerate(temps):
        cond1 = np.count_nonzero(t.valid_pixels_mask) > 10
        cond2 = t.water_temperature > 5
        
        if cond1 and cond2: output.append(t)
            
    return output
        
        

if __name__ == "__main__":
    import settings
    import argparse
    import traceback

    def get_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument("-i", "--input_date", help="Date in format YYYY-MM-DD to get data from",
                        required=True)
        parser.add_argument("-o", "--output_path", help="path to txt file where result will be appended",
                        required=True)

        return parser
    
    args = get_parser().parse_args()
    vd = args.input_date
    vd = datetime.strptime(vd, '%Y-%m-%d').date()
    output_path = args.output_path

    lst_dataset = LSTDataset(settings.raw_data_path, settings.footprint)

    sampling_points_coords = {"SAUCE NORTE": [-34.795398, -55.047355],
                          "SAUCE SUR": [-34.843127, -55.064624],
                          "TA": [-34.829670, -55.049758]}
    sampling_points_coords = [v for k, v in sampling_points_coords.items()]
    
    try:

        temperature_data, errors = lst_dataset.get_day_data(vd)

#         best_temp_data = find_best_temperature_candidate(temperature_data)
#         if best_temp_data:
#             with open('temp_output.txt', 'a') as f:
#                 f.write(datetime.strftime(best_temp_data.capture_datetime, "%Y-%m-%d %H:%M") + "," + str(best_temp_data.water_temperature) + "\n")
#         else:
#             with open('temp_output.txt', 'a') as f:
#                 f.write(str(vd) + "," + "None" + "\n")
        
        valid_lst_data = filter_valid_temps(temperature_data)
        valid_lst_data = sorted(valid_lst_data, key=lambda x:x.capture_datetime)
        for lst in valid_lst_data:
            sampling_points_mask = lst.make_mask_from_coords(sampling_points_coords)
    
            water_temp_avg = lst.lst[np.nonzero(sampling_points_mask)].mean()
            with open(output_path, 'a') as f:
                f.write(datetime.strftime(lst.capture_datetime, "%Y-%m-%d %H:%M") + "," + str(int(round(water_temp_avg, 0))) + "\n")
    except Exception as e:
        with open(output_path, 'a') as f:
            f.write(str(vd) + "," + "Exception raised:" + str(e) + "\n")
        
        
    
