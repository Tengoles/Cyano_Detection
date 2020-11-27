import sys
import os
import settings
import snappy
from snappy import ProductIO, WKTReader, HashMap, GPF
import numpy as np

def read_product(product_path):
    #print("Reading %s" % product_path)
    product = ProductIO.readProduct(product_path)
    return product

def make_subset(product, wkt):
    print("Making subset")
    SubsetOp = snappy.jpy.get_type('org.esa.snap.core.gpf.common.SubsetOp')
    geom = WKTReader().read(wkt)
    op = SubsetOp()
    op.setSourceProduct(product)
    op.setGeoRegion(geom)
    sub_product = op.getTargetProduct()
    return sub_product
    
def apply_rayleigh_correction(product, bands):
    #Must be BEAM-DIMAP product or will throw null pointer error
    parameters = HashMap()
    bands_string = ""
    for band in bands:
        bands_string += band + ","
    bands_string = bands_string[:-1]
    #print(bands_string)
    parameters.put("sourceBandNames", bands_string)
    parameters.put("computeTaur", "false")
    parameters.put("computeRBrr", "true")
    parameters.put("computeRtoa", "false")
    parameters.put("addAirMass", "false")
    parameters.put("s2MsiTargetResolution", 20)
    parameters.put("s2MsiSeaLevelPressure", 1013.25)
    parameters.put("s2MsiOzone", 300.0)

    rayleigh_output = GPF.createProduct('RayleighCorrection', parameters, product)
    return rayleigh_output

def apply_mph(product, cyano_max_value=1000.0, chl_th_float_flag=350.0):
    parameters = HashMap()
    parameters.put('validPixelExpression', 'quality_flags.fresh_inland_water')
    parameters.put("cyanoMaxValue", cyano_max_value)
    parameters.put("chlThreshForFloatFlag", chl_th_float_flag)

    #not sure if it's necessary to put these parameters, they might be put by default
    parameters.put("exportMph", "false")
    parameters.put("applyLowPassFilter", "false")

    mph_output = GPF.createProduct('MphChl', parameters, product)
    return mph_output

def write_product(product, path, p_type):
    #Algunas opciones de p_type: "GeoTiff":.tif y "BEAM-DIMAP":.dim
    print("Writing %s to %s" % (p_type, path))
    ProductIO.writeProduct(product, path, p_type)

def get_bands(product, bands):
    width = product.getSceneRasterWidth()
    height = product.getSceneRasterHeight()
    output_dict = {}
    for b in bands:
        band_product = product.getBand(b)
        band_array = np.zeros(width*height, dtype=np.float32)
        band_product.readPixels(0, 0, width, height, band_array)
        band_array.shape = (height, width)
        output_dict[b] = band_array
    return output_dict

if __name__=="__main__":
    path = "C:\\Users\\enzot\\Documents\\Maestria\\Cianobacterias\\data\\2020-06-13\\laguna.tif"
    
    #read product with SNAP API
    product = read_product(path)
    
    wkt = settings.footprint

    #make subset without saving
    #subset = make_subset(file_path, wkt, _write=False)
    
    radiance_bands = [band for band in product.getBandNames() if "radiance" in band]
    
    #apply rayleigh correction to subset
    rayleigh_subset = apply_rayleigh_correction(product, radiance_bands)

    #apply mph to subset
    mph_subset = apply_mph(rayleigh_subset)
    
    #load tif to numpy array
    mph_bands = mph_subset.getBandNames()
    arr = get_bands(mph_subset, mph_bands)

    #Verificar que correr Rayleigh y MPH desde aca sea lo mismo que desde SNAP y que este todo piola con guardar el tif y cargarlo