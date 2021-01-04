import numpy as np
import cv2
import snappy_utils
from astropy.nddata import bitmask

def bands_to_array(bands_dict):
    data_shape = bands_dict[list(bands_dict.keys())[0]].shape
    output = np.zeros((data_shape[0], data_shape[1], len(bands_dict)))
    for i, band in enumerate(bands_dict):
        output[:, :, i] = bands_dict[band]
    return output

def normalize_array(array):
    return cv2.normalize(src=array, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

def display_sat_data(array, wait=True):
    img_n = normalize_array(array)
    img_n = cv2.cvtColor(img_n, cv2.COLOR_RGB2BGR)
    img_n = cv2.resize(img_n, (img_n.shape[0]*6, img_n.shape[1]*6))
    cv2.imshow("Image", img_n)
    if not wait:
        cv2.waitKey(1)
    else:
        cv2.waitKey(0)

def display_tristimulus(product, wait=True):
    bands = ["Oa09_radiance", "Oa10_radiance", "Oa04_radiance", "Oa05_radiance"]
    data = snappy_utils.get_bands(product, bands)
    output_data = np.zeros((product.getSceneRasterHeight(), product.getSceneRasterWidth(), 3), dtype=np.float32)
    #red
    output_data[:,:,0] = data["Oa09_radiance"]*7 + data["Oa10_radiance"]*0.04
    #green
    output_data[:,:,1] = data["Oa09_radiance"]*3 + data["Oa10_radiance"]*0.02
    #blue
    output_data[:,:,2] = data["Oa04_radiance"]*47 + data["Oa05_radiance"]*0.16
    img_n = cv2.normalize(src=output_data, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img_n = cv2.cvtColor(img_n, cv2.COLOR_RGB2BGR)
    cv2.imshow("RGB", img_n)
    if not wait:
        cv2.waitKey(1)
    else:
        cv2.waitKey(0)

def stretch(val, min, max):
    return (val-min)/(max-min)

def stretch_to_MinMax(arr, verbose=False):
    # get array maximum value
    min_val = np.amin(arr)
    # get array minimum value
    max_val = np.amax(arr)
    if (verbose == True):
        print("Min: {}".format(min_val))
        print("Max: {}".format(max_val))
    # strech array to min and max
    stretched = (arr-min_val)/(max_val-min_val)
    # map to 0-255
    stretched = stretched*255
    # cast entire array to int
    stretched = stretched.astype(int)
    return stretched

def enhanced_true_color(B09, B08, B06, B14, B04):
    brightness = 1
    index = (B04-B08)/(B06+B09)
    band1 = brightness * (stretch(B09, 0, 0.25) - 0.1*stretch(B14, 0, 0.1))
    band2 = brightness * (1.1*stretch(B06, 0, 0.25) - 0.1*stretch(B14, 0, 0.1))
    band3 = brightness * (stretch(B04, 0, 0.25) - 0.1*stretch(B14, 0, 0.1) + 0.01*stretch(index, 0.5, 1))
    output = np.zeros((band1.shape[0], band1.shape[1], 3), dtype=np.float32)
    output[:, :, 0] = band1
    output[:, :, 1] = band2
    output[:, :, 2] = band3
    return output

def histogram_equalization(img_in):
    # segregate color streams
    b,g,r = cv2.split(img_in)
    h_b, bin_b = np.histogram(b.flatten(), 256, [0, 256])
    h_g, bin_g = np.histogram(g.flatten(), 256, [0, 256])
    h_r, bin_r = np.histogram(r.flatten(), 256, [0, 256])
    # calculate cdf    
    cdf_b = np.cumsum(h_b)  
    cdf_g = np.cumsum(h_g)
    cdf_r = np.cumsum(h_r)
    
    # mask all pixels with value=0 and replace it with mean of the pixel values 
    cdf_m_b = np.ma.masked_equal(cdf_b,0)
    cdf_m_b = (cdf_m_b - cdf_m_b.min())*255/(cdf_m_b.max()-cdf_m_b.min())
    cdf_final_b = np.ma.filled(cdf_m_b,0).astype('uint8')
  
    cdf_m_g = np.ma.masked_equal(cdf_g,0)
    cdf_m_g = (cdf_m_g - cdf_m_g.min())*255/(cdf_m_g.max()-cdf_m_g.min())
    cdf_final_g = np.ma.filled(cdf_m_g,0).astype('uint8')
    
    cdf_m_r = np.ma.masked_equal(cdf_r,0)
    cdf_m_r = (cdf_m_r - cdf_m_r.min())*255/(cdf_m_r.max()-cdf_m_r.min())
    cdf_final_r = np.ma.filled(cdf_m_r,0).astype('uint8')
    # merge the images in the three channels
    img_b = cdf_final_b[b]
    img_g = cdf_final_g[g]
    img_r = cdf_final_r[r]
  
    img_out = cv2.merge((img_b, img_g, img_r))
    # validation
    equ_b = cv2.equalizeHist(b)
    equ_g = cv2.equalizeHist(g)
    equ_r = cv2.equalizeHist(r)
    #equ = cv2.merge((equ_b, equ_g, equ_r))
    equ = cv2.merge((equ_r, equ_g, equ_b))
    #print(equ)
    #cv2.imwrite('output_name.png', equ)
    return img_out

def CLAHE(img_in, tileGridSize=(3,3)):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=tileGridSize)
    if img_in.shape[2] == 0:
        cl1 = clahe.apply(img_in)
        return cl1
    else:
        output = np.zeros(img_in.shape)
        for i in range(img_in.shape[2]):
            cl1 = clahe.apply(img_in[:,:,i])
            output[:,:,i] = cl1
        return output

quality_flag_ref = {"land": 2147483648,
                    "coastline": 1073741824,
                    "fresh_inland_water": 536870912,
                    "tidal_region": 268435456,
                    "bright": 134217728,
                    "straylight_risk": 67108864,
                    "invalid": 33554432,
                    "cosmetic": 16777216,
                    "duplicated": 8388608,
                    "sun-glin_risk": 4194304,
                    "dubious": 2097152,
                    "saturated@Oa01": 1048576,
                    "saturated@Oa02": 524288,
                    "saturated@Oa03": 262144,
                    "saturated@Oa04": 131072,
                    "saturated@Oa05": 65536,
                    "saturated@Oa06": 32768,
                    "saturated@Oa07": 16384,
                    "saturated@Oa08": 8192,
                    "saturated@Oa09": 4096,
                    "saturated@Oa10": 2048,
                    "saturated@Oa11": 1024,
                    "saturated@Oa12": 512,
                    "saturated@Oa13": 256,
                    "saturated@Oa14": 128,
                    "saturated@Oa15": 64,
                    "saturated@Oa16": 32,
                    "saturated@Oa17": 16,
                    "saturated@Oa18": 8,
                    "saturated@Oa19": 4,
                    "saturated@Oa20": 2,
                    "saturated@Oa21": 1
}

def make_flags_mask(quality_flags, desired_flags, flip_bits=True):
    flags = [quality_flag_ref[f] for f in desired_flags]
    mask = np.zeros(quality_flags.shape, dtype=bool)
    for i, row in enumerate(mask):
        for j, value in enumerate(row):
            mask[i,j] = bitmask.bitfield_to_boolean_mask(int(quality_flags[i,j]), ignore_flags=flags, flip_bits=flip_bits).item()
    return mask
