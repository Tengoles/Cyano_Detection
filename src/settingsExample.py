raw_data_path = "/home/ubuntu/cianobacterias/data/raw"
processed_data_path = "/home/ubuntu/cianobacterias/data/processed"
final_data_path = "/home/ubuntu/cianobacterias/data/final"
water_mask_path = "/home/ubuntu/cianobacterias/data/misc/water_masks/selected/"
# (longitude, latitude)
footprint = "POLYGON((-55.16629529644229 -34.7494869239046,-55.02038312603214 -34.7494869239046,-55.02038312603214 -34.868725532230165,-55.16629529644229 -34.868725532230165,-55.16629529644229 -34.7494869239046))"
sentinel_api_user = "your.user"
sentinel_api_key = "your.key"
###Acolite setttings
#resolution of acolite output
s2_target_res = 10
#string of parameters for acolite output separated by comma
l2w_parameters = "rhos_*"
#coordinates for ROI [south, west, north, east]
limit = -34.868725532230165,-55.16629529644229,-34.7494869239046,-55.02038312603214
l2r_export_geotiff = True
l2w_export_geotiff = True
export_geotiff_coordinates = True