import os, sys
import shutil
import settings
import argparse
sys.path.append("acolite")
import acolite as ac

def get_parser():
    parser = argparse.ArgumentParser(description='Script to apply optimization to the quaternions obtained from smoothing pre processing')
    
    parser.add_argument("--input-paths", nargs="+", default=[])

    parser.add_argument("--output-directory", default='', type=str)

    return parser

if __name__ == "__main__":
    data_path = settings.raw_data_path
    opt = get_parser().parse_args()
    # Get paths of sentinel products to process
    days_to_process = []
    if opt.input_paths == []:
        for day_directory in os.listdir(data_path):
            day_directory_path = os.path.join(data_path, day_directory)
            day_diretory_files = os.listdir(day_directory_path)
            for file in day_diretory_files:
                if (".SAFE" in file) and ("MSIL1C" in file):
                    final_file_path = os.path.join(day_directory_path, file)
                    print(final_file_path)
                    days_to_process.append(final_file_path)
    else:
        days_to_process = opt.input_paths

    print("Total data files:", len(days_to_process))
    # Make file with paths to be loaded by ACOLITE
    acolite_settings = {"limit": settings.limit,
                       "s2_target_res": settings.s2_target_res,
                       "l2w_parameters": settings.l2w_parameters,
                       "l2r_export_geotiff": settings.l2r_export_geotiff,
                       "l2w_export_geotiff": settings.l2w_export_geotiff,
                       "export_geotiff_coordinates": settings.export_geotiff_coordinates}
    for product_path in days_to_process:
        if opt.output_directory == '':
            output_directory = os.path.join(settings.processed_data_path, os.path.basename(os.path.dirname(product_path)), "MSI")
        else:
            output_directory = os.path.join(opt.output_directory, os.path.basename(os.path.dirname(product_path)), "MSI")
        print(output_directory)
        #if os.path.exists(output_directory):
            #shutil.rmtree(output_directory)
        try:
            os.makedirs(output_directory, exist_ok=False)
        except Exception as e:
            print("Exception when creating output directory:", str(e), "\ncontinuing to next day.")
            print("-----------------------------------------------")
            continue

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