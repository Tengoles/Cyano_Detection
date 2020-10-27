import snappy_utils
from mph import MPH
import utils
import numpy as np
import matplotlib.pyplot as plt

#path = "C:\\Users\\enzot\\Documents\\Maestria\\Cianobacterias\\data\\2020-06-14\\laguna.dim"
path = "C:\\Users\\enzot\\Documents\\Maestria\\Cianobacterias\\data\\2020-03-23\\laguna.dim"

product = snappy_utils.read_product(path)

quality_flags = snappy_utils.get_bands(product, ["quality_flags"])["quality_flags"]

bands = ["Oa07_radiance", "Oa08_radiance", "Oa10_radiance",
        "Oa11_radiance", "Oa12_radiance", "Oa18_radiance"]

rgb_bands = ["Oa09_radiance", "Oa08_radiance", "Oa06_radiance",
            "Oa14_radiance", "Oa04_radiance"]

rgb_bands_dict = snappy_utils.get_bands(product, rgb_bands)

trueColor_array = utils.enhanced_true_color(rgb_bands_dict["Oa09_radiance"], 
                        rgb_bands_dict["Oa08_radiance"], rgb_bands_dict["Oa06_radiance"],
                        rgb_bands_dict["Oa14_radiance"], rgb_bands_dict["Oa04_radiance"])

brrs_product = snappy_utils.apply_rayleigh_correction(product, bands)

brr_bands = ["rBRR_07", "rBRR_08", "rBRR_10", "rBRR_11", "rBRR_12", "rBRR_18", "quality_flags"]

brrs_arrays = snappy_utils.get_bands(brrs_product, brr_bands)

mph_product = snappy_utils.apply_mph(brrs_product)

mph_bands = ["chl", "immersed_cyanobacteria", "floating_cyanobacteria", 
            "floating_vegetation", "mph_chl_flags", "quality_flags"]

mph_arrays = snappy_utils.get_bands(mph_product, mph_bands)
mph = MPH(brrs_arrays)
water_mask = utils.make_flags_mask(quality_flags, ["fresh_inland_water"])
q_flags = mph.quality_flags
mph = mph.output

fig = plt.figure()
fig.add_subplot(231).title.set_text("immersed_cyanobacteria SNAP")
plt.imshow(utils.normalize_array(mph_arrays["immersed_cyanobacteria"]*water_mask), cmap="gray")
fig.add_subplot(232).title.set_text("floating_cyanobacteria SNAP")
plt.imshow(utils.normalize_array(mph_arrays["floating_cyanobacteria"]*water_mask), cmap="gray")
fig.add_subplot(233).title.set_text("floating_vegetation SNAP")
plt.imshow(utils.normalize_array(mph_arrays["floating_vegetation"]*water_mask), cmap="gray")
fig.add_subplot(235).title.set_text("RGB view")
plt.imshow(utils.histogram_equalization(utils.normalize_array(trueColor_array))) 
fig.tight_layout()

fig2 = plt.figure()
fig2.add_subplot(231).title.set_text("immersed cyanobacteria Enzo")
plt.imshow(np.logical_and(mph["cyano_flag"], np.logical_not(mph["float_flag"]))*water_mask, cmap="gray")
fig2.add_subplot(232).title.set_text("floating cyanobacteria Enzo")
plt.imshow(np.logical_and(mph["cyano_flag"], mph["float_flag"])*water_mask, cmap="gray")
fig2.add_subplot(233).title.set_text("floating vegetation Enzo")
plt.imshow(np.logical_and(mph["float_flag"],np.logical_not(mph["cyano_flag"]), np.logical_not(mph["adj_flag"]))*water_mask, cmap="gray")
fig2.add_subplot(235).title.set_text("RGB view")
plt.imshow(utils.histogram_equalization(utils.normalize_array(trueColor_array))) 
fig2.tight_layout()

fig3, (ax1, ax2) = plt.subplots(figsize=(13, 3), ncols=2)

# plot just the positive data and save the
# color "mappable" object returned by ax1.imshow
pos = ax1.imshow(np.nan_to_num(mph_arrays["chl"]*water_mask), cmap='jet', interpolation='none')

# add the colorbar using the figure's method,
# telling which mappable we're talking about and
# which axes object it should be near
fig3.colorbar(pos, ax=ax1)

# repeat everything above for the negative data
neg = ax2.imshow(mph["chl_mph"]*water_mask, cmap='jet', interpolation='none')
fig3.colorbar(neg, ax=ax2)
fig3.tight_layout()
plt.show()
