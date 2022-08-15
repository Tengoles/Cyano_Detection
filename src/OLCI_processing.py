import os
import sys
import settings
import argparse
import traceback
import snappy_utils

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", help="Path to Sentinel-3 OLCI raw data",
                    required=True)
    
    parser.add_argument("-o", "--output_path", help="Path to output of processed data",
                    required=True)

    return parser


if __name__ == "__main__":    
    args = get_parser().parse_args()
    input_path = args.input_path
    output_path = args.output_path
    
    beam_dimap_output_path = output_path + ".dim"
    tif_output_path = output_path + ".tif"
    
    if os.path.isfile(beam_dimap_output_path):
        print(f"{beam_dimap_output_path} already created")
        sys.exit()
    if os.path.isfile(tif_output_path):
        print(f"{tif_output_path} already created")
        sys.exit()
    
    try:
        output_directory = os.path.dirname(beam_dimap_output_path)
        os.makedirs(output_directory, exist_ok=True)

        product = snappy_utils.read_product(input_path)
        subset = snappy_utils.make_subset(product, settings.footprint)

        snappy_utils.write_product(subset, beam_dimap_output_path, "BEAM-DIMAP")
        snappy_utils.write_product(subset, tif_output_path, "GeoTiff")
    except Exception as e:
        print(f"Exception in {input_path}")
        print(e)