import snappy_utils
import utils
import numpy as np
import matplotlib.pyplot as plt

path = "C:\\Users\\enzot\\Documents\\Maestria\\Cianobacterias\\data\\2020-03-24\\laguna.dim"

product = snappy_utils.read_product(path)

bands = ["Oa07_radiance", "Oa08_radiance", "Oa10_radiance",
        "Oa11_radiance", "Oa12_radiance", "Oa18_radiance"]

brrs_product = snappy_utils.apply_rayleigh_correction(product, bands)

brr_bands = ["rBRR_07", "rBRR_08", "rBRR_10", "rBRR_11", "rBRR_12", "rBRR_18", "quality_flags"]

brrs_arrays = snappy_utils.get_bands(brrs_product, brr_bands)

mph_product = snappy_utils.apply_mph(brrs_product)

mph_bands = ["chl", "immersed_cyanobacteria", "floating_cyanobacteria", 
            "floating_vegetation", "mph_chl_flags", "quality_flags"]

mph_arrays = snappy_utils.get_bands(mph_product, mph_bands)

fig = plt.figure()

fig.add_subplot(231).title.set_text("immersed_cyanobacteria")
plt.imshow(utils.normalize_array(mph_arrays["immersed_cyanobacteria"]), cmap="gray")
fig.add_subplot(232).title.set_text("floating_cyanobacteria")
plt.imshow(utils.normalize_array(mph_arrays["floating_cyanobacteria"]), cmap="gray")
fig.add_subplot(233).title.set_text("floating_vegetation")
plt.imshow(utils.normalize_array(mph_arrays["floating_vegetation"]), cmap="gray")
fig.add_subplot(234).title.set_text("chl")
plt.imshow(utils.normalize_array(mph_arrays["chl"]), cmap="gray")
fig.add_subplot(235).title.set_text("mph_chl_flags")
plt.imshow(utils.normalize_array(mph_arrays["mph_chl_flags"]), cmap="gray")

fig.tight_layout()

plt.show()
