from tqdm import tqdm
import json
import os
import process_sentinel2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import random
from sklearn.cluster import KMeans

CLUSTERS_COLORS = [[240, 15, 15], [240,163,255], [153,63,0], [43,206,72], [255,204,153], [31, 49, 209], [255,255,255]]
#CLUSTERS_COLORS = [[255,255,255], [240,163,255], [153,63,0], [43,206,72], [255,204,153], [31, 49, 209], [0, 0, 0]]

class ClusteringProcessing:
    def __init__(self, day_data_generator, mask, clustering_algorithm="kmeans", k_clusters=4):
        self.mask = mask
        self.day_data_generator = day_data_generator
        self.dataset, self.ndci_dataset_array = self._make_dataset()
        self.k_clusters = k_clusters
        if clustering_algorithm == "kmeans":
            self.labels = self._perform_kmeans()
        
    
    def _make_dataset(self):
        # make array to store ndci of every valid day
        ndci_dataset_array = np.zeros((self.mask.height, self.mask.width, len(self.day_data_generator)))
        H, W, D = ndci_dataset_array.shape
        # list of dates for ndci_dataset_array
        dates_list = []
        # progress bar
        pbar = tqdm(total=len(self.day_data_generator))
        for d, day in enumerate(self.day_data_generator):
            ndci_day_array = day.get_NDCI()
            dates_list.append(day.date.date())
            ndci_dataset_array[:, :, d] = ndci_day_array
            pbar.update(1)
        pbar.close()
        self.sample_day_rgb = day.rgb
        clustering_dataset = []
        original_indexes = []
        pbar = tqdm(total=self.mask.height*self.mask.width)
        for i in range(self.mask.height):
            for j in range(self.mask.width):
                if self.mask.array[i, j] == 1:
                    clustering_dataset.append(list(ndci_dataset_array[i, j, :]))
                    original_indexes.append([i, j])
                pbar.update(1)
        pbar.close()
        clustering_data = {}
        clustering_data["data"] = clustering_dataset
        clustering_data["positions"] = original_indexes
        clustering_data["dates"] = dates_list
        return clustering_data, ndci_dataset_array
    
    def store_dataset(self, save_path):
        with open(save_path, 'w') as f:
            json.dump(self.dataset, f)
    
    def _perform_kmeans(self):
        X = self.dataset["data"]
        #print(X)
        clustering = KMeans(n_clusters=self.k_clusters, random_state=0).fit(X)
        return clustering.labels_
        
    
    def make_clusters_mask(self):
        output = np.ones((self.sample_day_rgb.shape[0], self.sample_day_rgb.shape[1]), 
                             dtype=np.uint8)*254
        # itearte over a list containing the label of every pixel and use it to make mask
        for i, label in enumerate(self.labels):
            if label == -1:
                continue
            label_position = self.dataset["positions"][i]
            output[label_position[0], label_position[1]] = int(label)
        self.cluster_mask = output
        return output
        
    
    def make_clusters_image(self):
        def draw_text(text,img, x , y, color = (255,255,255)):
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (110, 25)
            fontScale = 1
            thickness = 2
            cv2.putText(img, text, (x, y), font, 1, color, thickness, cv2.LINE_AA)
        
        output = self.sample_day_rgb.copy()
        # here I'm iterating over a list containing the label of every pixel in the lagoon
        for i, label in enumerate(self.labels):
            if label == -1:
                continue
            label_position = self.dataset["positions"][i]
            output[label_position[0], label_position[1], :] = CLUSTERS_COLORS[label]    

        #draw legends
        legend_position = {"x": 50, "y": 50}
        for label in set(self.labels):
            draw_text("cluster " + str(label), output, legend_position["x"], legend_position["y"], color=CLUSTERS_COLORS[label])
            legend_position["y"] = legend_position["y"] + 50
            
        return output
    
    def make_predictions_dataset(self):
        predictions_dataset = []
        for i, d in enumerate(self.dataset["dates"]):
            entry = {}
            entry["date"] = d
            day_ndci = self.ndci_dataset_array[:, :, i]
            for c in range(self.k_clusters):
                y, x = np.where(self.cluster_mask == c)
                cluster_ndci_values = []
                for i, j in zip(y, x):
                    cluster_ndci_values.append(day_ndci[i, j])
                cluster_ndci_values = np.array(cluster_ndci_values)
                cluster_mean = np.mean(cluster_ndci_values)
                cluster_std = np.std(cluster_ndci_values)
                entry[f"cluster_{c}_mean"] = cluster_mean
                entry[f"cluster_{c}_max"] = cluster_mean + 2*cluster_std
                entry[f"cluster_{c}_min"] = cluster_mean - 2*cluster_std
            predictions_dataset.append(entry)
        return pd.DataFrame(predictions_dataset)
        

if __name__ == "__main__":
    DATA_PATH = "../data/processed/"
    DATE_FORMAT = '%Y-%m-%d'
    START_DATE = '2016-12-21'
    END_DATE = '2021-04-20'
    MASKS_DIR = "../data/misc/water_masks/selected"
    
    day_data_generator = process_sentinel2.DayDataGenerator(START_DATE, END_DATE, DATE_FORMAT, DATA_PATH, skip_invalid=True)
    
    mask = process_sentinel2.Mask(MASKS_DIR)
    
    clustering_object = ClusteringProcessing(day_data_generator, mask, clustering_algorithm="kmeans", k_clusters=4)
    
    clusters_image = clustering_object.make_clusters_image()
    
    fig = plt.figure(figsize=(20,20))
    plt.imshow(clusters_image)
    plt.savefig("clusters_map.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    clusters_mask = clustering_object.make_clusters_mask()
    
    print(clustering_object.make_predictions_dataset())
    
    
    
    
    