import os, sys
sys.path.append("../")
sys.path.append("acolite")
import settings
import acolite as ac

data_path = settings.data_path
# Get paths of sentinel products to process
days_to_process = []
for day_directory in os.listdir(data_path):
    day_directory_path = os.path.join(data_path, day_directory)
    day_diretory_files = os.listdir(day_directory_path)
    for file in day_diretory_files:
        if ".SAFE" in file:
            final_file_path = os.path.join(day_directory_path, file)
            print(final_file_path)
            days_to_process.append(final_file_path)
        if ".zip" in file:
            os.remove(os.path.join(day_directory_path, file))
print("Downloaded files:", len(days_to_process))
# Make file with paths to be loaded by ACOLITE
acolite_settings = {"limit": settings.limit,
                   "s2_target_res": settings.s2_target_res,
                   "l2w_parameters": settings.l2w_parameters,
                   "l2r_export_geotiff": settings.l2r_export_geotiff,
                   "l2w_export_geotiff": settings.l2w_export_geotiff,
                   "export_geotiff_coordinates": settings.export_geotiff_coordinates}

for product_path in days_to_process:
    acolite_settings["inputfile"] = product_path
    acolite_settings["output"] = os.path.dirname(product_path)
    
    ac.acolite.acolite_run(settings=acolite_settings)