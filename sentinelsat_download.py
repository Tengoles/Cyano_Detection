from sentinelsat.sentinel import SentinelAPI, read_geojson, geojson_to_wkt
import snappy_utils
import os
from datetime import date
import settings
import zipfile

class Sentinelsat_products:
    def __init__(self, date_start, date_finish, footprint=settings.footprint, platformname="Sentinel-3"):
        self.date_start = date_start
        self.date_finish = date_finish
        self.api = SentinelAPI(settings.sentinel_api_user, settings.sentinel_api_key, 'https://scihub.copernicus.eu/dhus')
        self.wkt_footprint = footprint
        self.products = self.query_products(self.date_start, self.date_finish)

    def query_products(self, date_start, date_finish, platformname="Sentinel-3"):
        # connect to the API
        api = SentinelAPI(settings.sentinel_api_user, settings.sentinel_api_key, 'https://scihub.copernicus.eu/dhus')

        # search by polygon, time, and Hub query keywords
        products = api.query(self.wkt_footprint, area_relation='Contains',
                            date = (self.date_start, self.date_finish),
                            platformname = 'Sentinel-3')
        return products
    
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
                
if __name__=="__main__":
    datos = Sentinelsat_products(date(2020, 3, 13), date(2020, 4, 4))
    datos.filter_products(instrument="OLCI", level="L1", p_type= "FR", timeliness="Non Time Critical")
    datos.download_products()

# GeoJSON FeatureCollection containing footprints and metadata of the scenes
#api.to_geojson(products)

# GeoPandas GeoDataFrame with the metadata of the scenes and the footprints as geometries
#api.to_geodataframe(products)

# Get basic information about the product: its title, file size, MD5 sum, date, footprint and
# its download url
#api.get_product_odata(<product_id>)