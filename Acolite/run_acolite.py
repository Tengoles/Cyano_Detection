import os, sys
import shutil
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
        if (".SAFE" in file) and ("MSIL1C" in file):
            final_file_path = os.path.join(day_directory_path, file)
            print(final_file_path)
            days_to_process.append(final_file_path)
print("Total data files:", len(days_to_process))
# Make file with paths to be loaded by ACOLITE
acolite_settings = {"limit": settings.limit,
                   "s2_target_res": settings.s2_target_res,
                   "l2w_parameters": settings.l2w_parameters,
                   "l2r_export_geotiff": settings.l2r_export_geotiff,
                   "l2w_export_geotiff": settings.l2w_export_geotiff,
                   "export_geotiff_coordinates": settings.export_geotiff_coordinates}

for product_path in days_to_process:
    output_directory = os.path.join(os.path.dirname(product_path), "acolite_output")
    #if os.path.exists(output_directory):
        #shutil.rmtree(output_directory)
#     try:
#         os.makedirs(output_directory, exist_ok=False)
#     except Exception as e:
#         print("Exception when creating output directory:", str(e), "\ncontinuing to next day.")
#         print("-----------------------------------------------")
#         continue
    
    acolite_settings["inputfile"] = product_path
    acolite_settings["output"] = output_directory
    
    ac.acolite.acolite_run(settings=acolite_settings)
    #remove unwanted files
    print("removing unwanted files")
    for file in os.listdir(output_directory):
        #if (".nc" in file) or ("rhot" in file):
        if ".nc" in file:
            os.remove(os.path.join(output_directory, file))
    print("-----------------------------------------------")