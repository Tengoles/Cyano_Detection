import numpy as np
import pandas as pd
import json
from datetime import timedelta

class PredictionDataset():
    def __init__(self, wind_data_path, water_temperature_data_path, precipitation_data_path, 
                     ndci_data_path, algae_gt_path, s3_brrs_path, bloom_thresholds, pre_bloom_max_days):
        
        self.bloom_thresholds = bloom_thresholds 
        self.wind_data_path = wind_data_path
        self.water_temperature_data_path = water_temperature_data_path
        self.precipitation_data_path = precipitation_data_path
        self.ndci_data_path = ndci_data_path
        self.algae_gt_path = algae_gt_path
        self.s3_brrs_path = s3_brrs_path
        
        self.wind_data, self.temperature_data = self._load_wind()
        self.ndci_data = self._load_ndci()
        self.precipitation_data = self._load_precipitation()
        self.water_temperature_data = self._load_water_temperature()
        self.s3_brrs_data = self._load_s3_brrs()
        self.algae_gt = self._load_algae()
        
        self.pre_bloom_max_days = pre_bloom_max_days
        self.bloom_forecast_gt = self.make_bloom_forecast_gt()
        
    def _load_wind(self):
        wind_df = pd.read_csv(self.wind_data_path)
        numeric_columns = ['wind_00', 'wind_06', 'wind_12', 'wind_18', 'temp_00', 'temp_06', 'temp_12', 'temp_18']
        wind_df['date']= pd.to_datetime(wind_df['date'], dayfirst=True)
        for c in numeric_columns:
            wind_df[c] = pd.to_numeric(wind_df[c])
        
        temp_df = wind_df[['date', 'temp_00', 'temp_06', 'temp_12', 'temp_18']].copy()
        del wind_df['temp_00']
        del wind_df['temp_06']
        del wind_df['temp_12']
        del wind_df['temp_18']
        
        temp_df['temperature'] = temp_df[['temp_06', 'temp_12']].mean(axis=1)
        del temp_df["temp_00"]
        del temp_df["temp_06"]
        del temp_df["temp_12"]
        del temp_df["temp_18"]
        
        wind_df['wind'] = wind_df[['wind_06', 'wind_12']].mean(axis=1)
        del wind_df["wind_00"]
        del wind_df["wind_06"]
        del wind_df["wind_12"]
        del wind_df["wind_18"]
        return wind_df, temp_df
    
    def _load_ndci(self):
        ndci_df = pd.read_csv(self.ndci_data_path)
        ndci_df['date']= pd.to_datetime(ndci_df['date'])
        
        unique_dates = ndci_df['date'].unique()
        unique_clusters = ndci_df['cluster'].unique()
        ndci_types = ["mean", "min", "max"]
        clusters_columns = [f"cluster{c}_{nt}" for c in unique_clusters for nt in ndci_types]
        ndci_columns = clusters_columns.insert(0,"date")
        
        output = []
        for d in unique_dates:
            row = {}
            row['date'] = d
            date_ndci = ndci_df[ndci_df['date'] == d]
            for ci in unique_clusters:
                cluster_ndci = date_ndci[date_ndci["cluster"] == ci]
                row[f"cluster{ci}_mean"] = cluster_ndci.mean_ndci.tolist()[0]
                row[f"cluster{ci}_min"] = cluster_ndci.min_ndci.tolist()[0]
                row[f"cluster{ci}_max"] = cluster_ndci.max_ndci.tolist()[0]
            output.append(row)
        output_df = pd.DataFrame(output)
        return output_df
    
    def _load_precipitation(self):
        precipitation_df = pd.read_csv(self.precipitation_data_path)
        precipitation_df['date']= pd.to_datetime(precipitation_df['date'])
        return precipitation_df
    
    def _load_water_temperature(self):
        water_temperature_df = pd.read_csv(self.water_temperature_data_path)
        water_temperature_df['date']= pd.to_datetime(water_temperature_df['date'])
        #return water_temperature_df
        rows_to_remove = []
        for i in range(water_temperature_df.shape[0]):
            row_hour = water_temperature_df.iloc[i].date.hour
            if row_hour > 17 or (row_hour >= 0 and  row_hour <= 9):
                rows_to_remove.append(i)
        water_temperature_df = water_temperature_df.drop(rows_to_remove)
        water_temperature_df['date'] = water_temperature_df['date'].dt.strftime('%Y-%m-%d')
        water_temperature_df = water_temperature_df.groupby('date', as_index=False).mean()
        water_temperature_df['date']= pd.to_datetime(water_temperature_df['date'])
        return water_temperature_df
    
    def _load_algae(self):
        algae_gt_df = pd.read_csv(self.algae_gt_path)
        algae_gt_df["date"] = pd.to_datetime(algae_gt_df['date'])

        output = pd.DataFrame()
        for location_name, threshold in self.bloom_thresholds.items():
            location_df = algae_gt_df[algae_gt_df["location"] == location_name]
            # Apply the condition and assign values to the "label" column
            location_df['label'] = location_df['fico'].apply(lambda x: 'Bloom' if x > threshold else 'No Bloom')
            output = pd.concat([output, location_df])
        output = output.sort_values(by=['date'], ascending=True)

        return output
    
    def _load_s3_brrs(self):
        with open(self.s3_brrs_path) as json_file:
            return json.load(json_file)
    
    
    def get_historic_data(self, date, days=3):        
        ndci = self.ndci_data[(self.ndci_data['date'] >= date - timedelta(days=days)) & (self.ndci_data['date'] <= date)]
        wind = self.wind_data[(self.wind_data['date'] >= date - timedelta(days=days)) & (self.wind_data['date'] <= date)]
        temperature = self.temperature_data[(self.temperature_data['date'] >= date - timedelta(days=days)) & (self.temperature_data['date'] <= date)]
        precipitation = self.precipitation_data[(self.precipitation_data['date'] >= date - timedelta(days=days)) & (self.precipitation_data['date'] <= date)]
        water_temperature = self.water_temperature_data[(self.water_temperature_data['date'] >= date - timedelta(days=days)) & (self.water_temperature_data['date'] <= date)]
        
        all_historic_data = {"ndci": ndci, "wind": wind, "temperature":temperature,
                             "precipitation": precipitation, "water_temperature": water_temperature}
        
        output = []
        for d in range(days):
            row = {}
            dt = date - timedelta(days=d)
            row["date"] = dt
            for df in all_historic_data.values():
                day_data = df[df["date"] == dt].to_dict("records")
                cols = df.columns.tolist()
                cols.remove("date")
                for col in cols:
                    if day_data == []:
                        row[col] = np.nan
                    else:
                        row[col] = day_data[0][col]
            output.append(row)
        return pd.DataFrame(output)
    
    def convert_days_until_bloom(self, days_until_bloom):
            if days_until_bloom > self.pre_bloom_max_days:
                return "No Bloom"
            elif days_until_bloom == 0:
                return "Bloom"
            else:
                return "Pre Bloom"
    
    def make_bloom_forecast_gt(self):
        output = pd.DataFrame()
        # Iterate over every unique location
        for location in self.algae_gt["location"].unique():
            days_until_bloom = []
            # Subselect GT dataset df keeping keeping only data from current location
            location_data_df = self.algae_gt[self.algae_gt["location"] == location]
            # Iterate over every row of the subselected dataframe. Row contains date, location name and ficocyanin measure 
            for day_index, day_data in location_data_df.iterrows():
                day_date = day_data["date"]
                day_cyano = day_data["fico"]
                # If fico measure on this day is higher than threshold, classify this day as "Bloom" and go to next one.
                if day_cyano >= self.bloom_thresholds[location]:                    
                    days_until_bloom.append(0)
                else:
                    # Else if fico measure is below threshold, subselect location df to keep only days after the current day being processed
                    following_days_df = location_data_df[location_data_df['date'] > day_date]
                    # Iterate over df containing data from the days following the current one being processed
                    next_bloom_found = False
                    for following_day_index, following_day in following_days_df.iterrows():
                        if following_day["fico"] >= self.bloom_thresholds[location]:
                            # Get day count until next bloom
                            next_bloom_days_count = (following_day["date"] - day_date).days
                            days_until_bloom.append(next_bloom_days_count)
                            next_bloom_found = True
                            break
                    if not next_bloom_found:
                        days_until_bloom.append(999)
            location_data_df["days until bloom"] = days_until_bloom
            output = pd.concat([output, location_data_df])
        output = output.sort_values(by="date")

        output["forecast label"] = output["days until bloom"].apply(self.convert_days_until_bloom)
        return output
    
    def change_bloom_forecast_label(self, new_pre_bloom_max_days):
        self.pre_bloom_max_days = new_pre_bloom_max_days
        self.bloom_forecast_gt["forecast label"] = self.bloom_forecast_gt["days until bloom"].apply(self.convert_days_until_bloom)

    
def merge_locations(df):
    output = []
    unique_dates = df.date.unique()
    for d in unique_dates:
        date_data = df[df["date"] == d]
        mean_fico = date_data["fico"].mean(skipna=True)
        mean_chl = date_data["chl"].mean(skipna=True)

        output.append({"date": d,
                       "fico": mean_fico,
                       "chl": mean_chl})
    return pd.DataFrame(output)
