import cv2
import snappy_utils
import os
import settings
import numpy as np
import utils
import matplotlib.pyplot as plt

"""Oa01_radiance, Oa02_radiance, Oa03_radiance, Oa04_radiance, Oa05_radiance
Oa06_radiance, Oa07_radiance, Oa08_radiance, Oa09_radiance, Oa10_radiance
Oa11_radiance, Oa12_radiance, Oa13_radiance, Oa14_radiance, Oa15_radiance
Oa16_radiance, Oa17_radiance, Oa18_radiance, Oa19_radiance, Oa20_radiance
Oa21_radiance"""

path = "C:\\Users\\enzot\\Documents\\Maestria\\Cianobacterias\\data\\2020-06-14\\laguna.tif"

rgb_bands = ["Oa09_radiance", "Oa08_radiance", "Oa06_radiance",
            "Oa14_radiance", "Oa04_radiance"]

product = snappy_utils.read_product(path)
product_dict = snappy_utils.get_bands(product, rgb_bands)
#product_array = utils.bands_to_array(product_dict)
rgb_image = utils.enhanced_true_color(product_dict["Oa09_radiance"], product_dict["Oa08_radiance"], 
                                        product_dict["Oa06_radiance"], product_dict["Oa14_radiance"],
                                        product_dict["Oa04_radiance"])
rgb_image = utils.normalize_array(rgb_image)
rgb_image = utils.histogram_equalization(rgb_image)
#rgb_image = utils.CLAHE(rgb_image)

plt.imshow(rgb_image)
plt.show()
                