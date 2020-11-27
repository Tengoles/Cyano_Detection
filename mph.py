import numpy as np

class MPH():
    def __init__(self, brrs_arrays, cyano_thresh=350):
        self.brr_7 = brrs_arrays["rBRR_07"]
        self.brr_8 = brrs_arrays["rBRR_08"]
        self.brr_10 = brrs_arrays["rBRR_10"]
        self.brr_11 = brrs_arrays["rBRR_11"]
        self.brr_12 = brrs_arrays["rBRR_12"]
        self.brr_18 = brrs_arrays["rBRR_18"]
        
        self.Rmax0, self.Rmax1 = self.determine_Rmax()
        self.lambda_max0, self.lambda_max1 = self.determine_lambdaMax()

        self.NDVI = (self.brr_18 - self.brr_8)/(self.brr_18 + self.brr_8)

        self.BAIR = self.brr_11 - self.brr_8 - ((self.brr_18 - self.brr_8)*(709 - 664))/(885-664)

        self.SICF = self.brr_10 - self.brr_8 - ((self.brr_11 - self.brr_8)*(681-664))/(709 - 664)

        self.SIPF = self.brr_8 - self.brr_7 - ((self.brr_10 - self.brr_7)*(665 - 620))/(681 - 619)

        self.MPH0 = self.Rmax0 - self.brr_8 - ((self.brr_18 - self.brr_8)*(self.lambda_max0 - 665))/(885 - 665)

        self.MPH1 = self.Rmax1 - self.brr_8 - ((self.brr_18 - self.brr_8)*(self.lambda_max1 - 665))/(885 - 665)

        _mph_output = self.run_mph(cyano_thresh)

        self.immersed_cyanobacteria = np.logical_and(_mph_output["cyano_flag"], 
                                                    np.logical_not(_mph_output["float_flag"]))
        self.floating_cyanobacteria = np.logical_and(_mph_output["cyano_flag"], 
                                                    _mph_output["float_flag"])
        self.floating_vegetation = np.logical_and(_mph_output["float_flag"],
                                                np.logical_not(_mph_output["cyano_flag"]), 
                                                np.logical_not(_mph_output["adj_flag"]))
        self.immersed_eukaryotes = np.logical_and(np.logical_not(_mph_output["float_flag"]),
                                                np.logical_not(_mph_output["cyano_flag"]))

        self.chl = _mph_output["chl_mph"]

        
    def determine_Rmax(self):
        Rmax0 = np.maximum(self.brr_10, self.brr_11)
        Rmax1 = np.maximum(Rmax0, self.brr_12)
        return Rmax0, Rmax1

    def determine_lambdaMax(self):
        lambda_brr11 = 708.75
        lambda_brr10 = 681.25
        lambda_brr12 = 753.75

        lambda0 = np.where(self.Rmax0 == self.brr_11, lambda_brr11, lambda_brr10)

        lambda1 = np.where(self.Rmax1 == self.brr_12, lambda_brr12, lambda0) 

        return lambda0, lambda1
    
    def run_mph(self, cyano_thresh):
        # output of this must have three 2D bool arrays for float_flag, adj_flag and cyano_flag
        # and one 2D float array for chl estimation
        
        chl_mph = np.zeros(self.brr_7.shape)
        float_flag = np.ones(self.brr_7.shape)
        adj_flag = np.ones(self.brr_7.shape)
        cyano_flag = np.ones(self.brr_7.shape)
        for i, row in enumerate(np.zeros(chl_mph.shape)):
            for j, element in enumerate(row):
                if self.lambda_max1[i,j] != 753.75:
                    float_flag[i,j] = 0
                    adj_flag[i,j] = 0
                    if self.SICF[i,j] >= 0 or self.SIPF[i,j] <= 0 or self.BAIR[i,j] <= 0.002:
                        cyano_flag[i,j] = 0
                        chl_mph[i,j] = 5.24e9*(self.MPH0[i,j]**4) - 1.95e8*(self.MPH0[i,j]**3) + 2.46e6*(self.MPH0[i,j]**2) + 4.02e3*self.MPH0[i,j] + 1.97
                    else:
                        cyano_flag[i,j] = 1
                        chl_mph[i,j] = 22.44*np.exp(35.79*self.MPH1[i,j])
                        if chl_mph[i,j] > cyano_thresh:
                            float_flag[i,j] = 1
                else:
                    if self.MPH1[i,j] >= 0.02 or self.NDVI[i,j] >= 0.2:
                        float_flag[i,j] = 1
                        adj_flag[i,j] = 0
                        if self.SICF[i,j] >= 0 or self.SIPF[i,j] <= 0:
                            cyano_flag[i,j] = 0
                            chl_mph[i,j] = 0
                        else:
                            cyano_flag[i,j] = 1
                            chl_mph[i,j] = 22.44*np.exp(35.79*self.MPH1[i,j])
                            if chl_mph[i,j] > cyano_thresh:
                                float_flag[i,j] = 1
                            else:
                                float_flag[i,j] = 0
                    else:
                        float_flag[i,j] = 0
                        adj_flag[i,j] = 1
                        cyano_flag[i,j] = 0
                        chl_mph[i,j] = 5.24e9*(self.MPH0[i,j]**4) - 1.95e8*(self.MPH0[i,j]**3) + 2.46e6*(self.MPH0[i,j]**2) + 4.02e3*self.MPH0[i,j] + 1.97

        output = {}
        output["chl_mph"] = chl_mph
        output["float_flag"] = float_flag
        output["adj_flag"] = adj_flag
        output["cyano_flag"] = cyano_flag
        return output

if __name__ == "__main__":
    import snappy_utils
    import matplotlib.pyplot as plt

    path = "C:\\Users\\enzot\\Documents\\Maestria\\Cianobacterias\\data\\2020-06-14\\laguna.dim"
    product = snappy_utils.read_product(path)
    bands = ["Oa07_radiance", "Oa08_radiance", "Oa10_radiance",
            "Oa11_radiance", "Oa12_radiance", "Oa18_radiance"]
    brrs_product = snappy_utils.apply_rayleigh_correction(product, bands)
    brr_bands = ["rBRR_07", "rBRR_08", "rBRR_10", "rBRR_11", "rBRR_12", "rBRR_18", "quality_flags"]
    brrs_arrays = snappy_utils.get_bands(brrs_product, brr_bands)
    mph = MPH(brrs_arrays)

    #prints to check calculations correct
    print("brr 7: ", mph.brr_7[20,20])
    print("brr 8: ", mph.brr_8[20,20])
    print("brr 10: ", mph.brr_10[20,20])
    print("brr 11: ", mph.brr_11[20,20])
    print("brr 12: ", mph.brr_12[20,20])
    print("brr 18: ", mph.brr_18[20,20])
    print("rmax0: ", mph.Rmax0[20,20])
    print("rmax1: ", mph.Rmax1[20,20])
    print("lambda0: ", mph.lambda_max0[20,20])
    print("lambda1: ", mph.lambda_max1[20,20])
    print("MPH0: ", mph.MPH0[20,20])
    print("MPH1: ", mph.MPH1[20,20])
    print("NDVI: ", mph.NDVI[20,20])
    print("BAIR: ", mph.BAIR[20,20])
    print("SICF: ", mph.SICF[20,20])
    print("SIPF: ", mph.SIPF[20,20])
    print("-----------")
    print("cyano_flag: ", mph.output["cyano_flag"][20, 20])
    print("adj_flag: ", mph.output["adj_flag"][20, 20])
    print("float_flag: ", mph.output["float_flag"][20, 20])
    print("chl_mph: ", mph.output["chl_mph"][20, 20])

    plt.imshow(mph.output["chl_mph"])
    plt.show()

    

