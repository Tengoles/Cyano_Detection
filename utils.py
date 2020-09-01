import numpy as np
import cv2
import snappy_utils

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