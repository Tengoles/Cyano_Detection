from sentinelsat.sentinel import SentinelAPI, read_geojson, geojson_to_wkt
from shapely import wkt
import snappy_utils
import os
import tqdm
import time
from datetime import date
import settings
import zipfile
from collections import OrderedDict
from datetime import timedelta

SENTINEL_API_USER = settings.sentinel_api_user
SENTINEL_API_KEY = settings.sentinel_api_key

class Sentinel3Products:
    def __init__(self, date_start, date_finish, footprint=settings.footprint, 
                 platformname="Sentinel-3", instrument="SLSTR", 
                 level="L2", p_type="LST"):
        self.date_start = date_start
        self.date_finish = date_finish
        self.api = SentinelAPI(settings.sentinel_api_user, settings.sentinel_api_key, 
                               'https://scihub.copernicus.eu/dhus')
        self.wkt_footprint = footprint
        self.platform_name = platformname
        self.instrument = instrument
        self.level = level
        self.p_type = p_type
        self.products = self.api.query(self.wkt_footprint, area_relation='Contains',
                            date = (self.date_start, self.date_finish),
                            platformname = platform)
    
    def filter_products(self, instrument, level, p_type, timeliness):
        removed_products = []
        for product_key in self.products:
            odata = self.api.get_product_odata(product_key, full=True)
            product_instrument = odata["Instrument"]
            product_level = odata["Product level"]
            product_type = odata["Product type"]
            #mission_type = odata["Mission type"]
            product_timeliness = odata["Timeliness Category"]
            #filter only from Level 1 OLCI instrument with NTC full resolution
            conditions = (
                (product_instrument == instrument) and (p_type in product_type) 
                and product_timeliness == timeliness and product_level == level
            )
            if conditions:
                pass
                #print(instrument, product_level, product_type)
            else:
                removed_products.append(product_key)
        for key in removed_products:
            del self.products[key]

    def download_products(self, make_subset=True):
        print("----------")
        for key in self.products:
            file_name = self.products[key]["filename"]
            file_date = self.products[key]["summary"][:16].split("Date: ")[1]
            download_path = os.path.join(settings.data_path, file_date)
            if not os.path.exists(download_path):
                os.makedirs(download_path)
            # if it was downloaded before it won't download again            
            download_info = self.api.download(key, directory_path=download_path)
            #print(download_info)
            zip_path = download_info["path"]
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(download_path)
            if make_subset:
                extracted_directory = os.path.join(download_path, file_name)
                product = snappy_utils.read_product(extracted_directory)
                subset = snappy_utils.make_subset(product, settings.footprint)
                snappy_utils.write_product(subset, os.path.join(download_path, "laguna.tif"), "GeoTiff")
                snappy_utils.write_product(subset, os.path.join(download_path, "laguna.dim"), "BEAM-DIMAP")
                
class SentinelsatProducts:
    def __init__(self, date_start, date_finish, footprint,
                 platformname, instrument, level=None, p_type=None):
        self.date_start = date_start
        self.date_finish = date_finish
        self.platform = platformname
        self.instrument = instrument
        self.level = level
        self.p_type = p_type
        self.wkt_footprint = footprint
        
        self.api = SentinelAPI(SENTINEL_API_USER, SENTINEL_API_KEY, 'https://scihub.copernicus.eu/dhus')
        #self.api.logger.setLevel(logging.DEBUG)
        
        self.footprint_polygon = wkt.loads(footprint)
        
        self.products = self.query_products()
        
        self.downloaded_prods, self.retrieval_scheduled, self.failed_prods = {}, {}, {}

    def query_products(self):
        # search by polygon, time, and Hub query keywords
        if self.platform == "Sentinel-3" and self.instrument == "OLCI":
            products = self.api.query(self.wkt_footprint, area_relation='Contains',
                            date=(self.date_start - timedelta(days=1), self.date_finish + timedelta(days=1)), platformname=self.platform,
                            instrumentname='Ocean Land Colour Instrument', productlevel=self.level, producttype="OL_1_EFR___")
        else:
             products = self.api.query(self.wkt_footprint, area_relation='Contains',
                            date=(self.date_start, self.date_finish), platformname=self.platform,
                            producttype=self.p_type)
        return products
    
    def filter_by_dates(self, dates_list):
        """
        dates_list must be list of datetetime.date() objects
        """
        if self.platform == "Sentinel-2":
            keys_to_remove = [p_key for p_key, product in self.products.items() \
                          if not (product['datatakesensingstart'].date() in dates_list)]
        elif self.platform == "Sentinel-3":
            keys_to_remove = [p_key for p_key, product in self.products.items() \
                          if not (product['beginposition'].date() in dates_list)]
        else:
            raise Exception("Invalid platform")
        #remove products that didn't meet the download conditions
        for key in keys_to_remove:
            del self.products[key]
            
        if self.platform == "Sentinel-2": 
            self.products = OrderedDict(sorted(self.products.items(), key=lambda x: x[1]['datatakesensingstart']))
        elif self.platform == "Sentinel-3":
            self.products = OrderedDict(sorted(self.products.items(), key=lambda x: x[1]['beginposition']))
    
    def filter_by_size(self, max_MB):
        """
        Removes products with size bigger than arg max_MB
        """
        keys_to_remove = [p_key for p_key, product in self.products.items() \
                          if convert_size_string(product['size']) >= max_MB]
        #remove products that didn't meet the download conditions
        for key in keys_to_remove:
            del self.products[key]
    
    def filter_products(self, instrument, level, p_type, timeliness, filter_distance=True):
        removed_products = []
        for product_key in self.products:
            odata = self.api.get_product_odata(product_key, full=True)
            #print(odata)
            if self.platform == "Sentinel-3":
                product_instrument = odata["Instrument"]
                product_level = odata["Product level"]
                product_type = odata["Product type"]
                mission_type = odata["Mission type"]
                product_timeliness = odata["Timeliness Category"]                
                conditions = (
                (product_instrument == instrument) and (p_type in product_type) 
                and product_timeliness == timeliness and product_level == level
                            )
            if self.platform == "Sentinel-2":
                product_instrument = odata["Instrument"]
                product_level = odata["Processing level"]            
                conditions = (product_instrument == instrument) and (product_level == level)
            #keep list of pids of products that meet downloading conditions
            if conditions:
                #print(instrument, product_level, product_type)
                pass                
            else:
                removed_products.append(product_key)
        #remove products that didn't meet the download conditions
        for key in removed_products:
            del self.products[key]

    def download_products(self, data_path, delete_zip=True, lta_request_delay=600):
        total_products = len(self.products.keys()) - 1        
        for i, key in enumerate(self.products):
            print("-------------------------")
            print(i, "/", total_products)
            
            file_name = self.products[key]["filename"]
            file_date = self.products[key]["summary"][:16].split("Date: ")[1]
            
            download_directory = os.path.join(data_path, file_date)
            download_path = os.path.join(download_directory, file_name)
            print(file_date)
            if not os.path.exists(download_directory):
                os.makedirs(download_directory)
            if os.path.exists(download_path):
                # if it was downloaded before it won't download
                print(f"{download_path} already downloaded")
                self.downloaded_prods[key] = self.products[key]
                continue
            
            product_odata = self.api.get_product_odata(key, full=True)
            
            if self.api.is_online(key) == False:
                print("Data from", file_date, "offline")
                if key in list(self.retrieval_scheduled.keys()):
                    print("LTA retrieval already triggered")
                    continue
                try:
                    lta_status_response = self.api._trigger_offline_retrieval(product_odata["url"])
                except Exception as e:
                    print("RAISED EXCEPTION:", str(e))
                    print("///////////////////////////")
                    self.failed_prods[key] = self.products[key]
                    continue
                if lta_status_response == 202:
                    print("Accepted for retrieval")
                    self.retrieval_scheduled[key] = self.products[key]
                    continue
                elif lta_status_response == 403:
                    while(lta_status_response == 403):
                        print("Requests exceed user quota. Retrying in %s seconds" % (lta_request_delay))
                        time.sleep(lta_request_delay)
                        lta_status_response = self.api._trigger_offline_retrieval(product_odata["url"])
                    if lta_status_response == 202:
                        print("Accepted for retrieval")
                        self.retrieval_scheduled[key] = self.products[key]
                continue
            try:
                download_info = self.api.download(key, directory_path=download_directory)
                zip_path = download_info["path"]
                time.sleep(3)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(download_directory)
                self.downloaded_prods[key] = self.products[key]
            except Exception as e:
                print("RAISED EXCEPTION:", str(e))
                print("/////////////")
                self.failed_prods[key] = self.products[key]
                continue
            if delete_zip:
                os.remove(zip_path)
                       
def convert_size_string(size_string):
    """
    Parses string of format X MB or Y GB to get size in MB and return tht size
    """
    unit = size_string.split(" ")[1]
    number = float(size_string.split(" ")[0])
    if unit == "MB":
        return number
    elif unit == "GB":
        return number*1024
    else:
        raise Exception("Invalid size string format")
    
                
if __name__=="__main__":
    date1 = date(2022, 5, 21)
    date2 = date(2022, 5, 23)
    
    # Sentinel-2 MSI Level-1C
    # products = SentinelsatProducts(date1, date2, footprint=settings.footprint, 
    #                               platformname="Sentinel-2", instrument="MSI", p_type='S2MSI1C')
    
    # Sentinel-3 SLSTR level-2
    # products = SentinelsatProducts(date1, date2, footprint=settings.footprint, 
    #                                 platformname="Sentinel-3", instrument="SLSTR", 
    #                                 level="L2", p_type="SL_2_LST___")
    
    # Sentinel-3 OLCI level-1
    products = SentinelsatProducts(date1, date2, footprint=settings.footprint,
                                platformname="Sentinel-3", instrument="OLCI", level="L1")
    
    for p_key, p in products.products.items():
        print(p['summary'])

    products.download_products(settings.raw_data_path)