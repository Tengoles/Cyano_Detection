import cv2
import snappy_utils
import os
import settings
import numpy as np
import utils

"""Oa01_radiance, Oa02_radiance, Oa03_radiance, Oa04_radiance, Oa05_radiance
Oa06_radiance, Oa07_radiance, Oa08_radiance, Oa09_radiance, Oa10_radiance
Oa11_radiance, Oa12_radiance, Oa13_radiance, Oa14_radiance, Oa15_radiance
Oa16_radiance, Oa17_radiance, Oa18_radiance, Oa19_radiance, Oa20_radiance
Oa21_radiance"""

path = "C:\\Users\\enzot\\Documents\\Maestria\\Cianobacterias\\data\\2020-03-24\\laguna.tif"

product = snappy_utils.read_product(path)
product_dict = snappy_utils.get_bands(product, ["Oa17_radiance", "Oa06_radiance", "Oa03_radiance"])
product_array = utils.bands_to_array(product_dict)
utils.display_sat_data(product_array, wait=True)
cv2.destroyAllWindows()
                