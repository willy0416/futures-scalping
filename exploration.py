import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
import os
# Control Constants

# Window: 
# -- Decides width of all moving stats
# -- Each day varies but is order of 1000 datapoints 
window_width = 10000
# Cutoff Factor:
# -- Decides how much data from the beginning we cut off (window width * cutoff_factor) 
# -- Done because small sample size at beginning --> chaotic behavior
cutoff_factor = 1/10
# Z Score Requirement:
# -- Long when largest > median + zscore_req
# -- Short when smallest < median - zscore_req
zscore_req = 0.55

data_folder = "data/"

def create_numeric_time_column(df):
    df["FULL_DATE"] = pd.to_datetime(df["DATE"] + " " + df["TIME"])
    df["TIME_NUMERIC"] = df['FULL_DATE'].astype('int64') // 10 ** 9 # nanoseconds to seconds

def create_moving_stats(df):
    stat_window = df["OPEN"].rolling(window_width, min_periods = 1)
    df["MOVING Z"] = (df["OPEN"] - stat_window.mean()) / stat_window.std()

### WORKING WITH SINGLE-DAY DATA

# Load in a file
mes_09_01 = pd.read_csv(data_folder + "MES_2024-09-01.csv")
mnq_09_01 = pd.read_csv(data_folder + "MNQ_2024-09-01.csv")
mym_09_01 = pd.read_csv(data_folder + "MYM_2024-09-01.csv")

cpy_mes = mes_09_01.copy()
cpy_mnq = mnq_09_01.copy()
cpy_mym = mym_09_01.copy()

### CONCATENATING SINGLE-DAY DATA TO GET THE MONTH-LONG VIEW
# TODO: be cognizant of when trading/data stops (a few notable days-long stretches, plus 17:00-17:59 daily)
full_df = cpy_mes
full_df_mnq = cpy_mnq
full_df_mym = cpy_mym

for i in range(1, 31): # September has 30 days
    file = f"{data_folder}MES_2024-09-{i:02d}.csv" # This is just for MES, sub out other tickers if you're gonna work with them
    file_mnq = f"{data_folder}MNQ_2024-09-{i:02d}.csv"
    file_mym = f"{data_folder}MYM_2024-09-{i:02d}.csv"

    if os.path.exists(file):
        df = pd.read_csv(file)
    else:
        # Disconnecting line plot to show when trading stops
        if i == 30:
            date_range = pd.date_range(start=f"2024-09-30 18:00", end='2024-10-01 16:59', freq='T')
        else:
            date_range = pd.date_range(start=f"2024-09-{i:02d} 18:00", end=f"2024-09-{i+1:02d} 16:59", freq='T')

        df = pd.DataFrame({
            'DATE': date_range.date,
            'TIME': date_range.time,
            'OPEN': np.nan,
            'HIGH': np.nan,
            'LOW': np.nan,
            'CLOSE': np.nan,
            'VOLUME': np.nan,
        })

        df['DATE'] = df['DATE'].astype(str)
        df['TIME'] = df['TIME'].astype(str).str[:-3]

    if os.path.exists(file_mnq):
        df_mnq = pd.read_csv(file_mnq)
    else:
        # Disconnecting line plot to show when trading stops
        if i == 30:
            date_range = pd.date_range(start=f"2024-09-30 18:00", end='2024-10-01 16:59', freq='T')
        else:
            date_range = pd.date_range(start=f"2024-09-{i:02d} 18:00", end=f"2024-09-{i + 1:02d} 16:59", freq='T')

        df_mnq = pd.DataFrame({
            'DATE': date_range.date,
            'TIME': date_range.time,
            'OPEN': np.nan,
            'HIGH': np.nan,
            'LOW': np.nan,
            'CLOSE': np.nan,
            'VOLUME': np.nan,
        })

        df_mnq['DATE'] = df_mnq['DATE'].astype(str)
        df_mnq['TIME'] = df_mnq['TIME'].astype(str).str[:-3]

    if os.path.exists(file_mym):
        df_mym = pd.read_csv(file_mym)
    else:
        # Disconnecting line plot to show when trading stops
        if i == 30:
            date_range = pd.date_range(start=f"2024-09-30 18:00", end='2024-10-01 16:59', freq='T')
        else:
            date_range = pd.date_range(start=f"2024-09-{i:02d} 18:00", end=f"2024-09-{i+1:02d} 16:59", freq='T')

        df_mym = pd.DataFrame({
            'DATE': date_range.date,
            'TIME': date_range.time,
            'OPEN': np.nan,
            'HIGH': np.nan,
            'LOW': np.nan,
            'CLOSE': np.nan,
            'VOLUME': np.nan,
        })

        df_mym['DATE'] = df_mym['DATE'].astype(str)
        df_mym['TIME'] = df_mym['TIME'].astype(str).str[:-3]

    full_df = pd.concat([full_df, df])
    full_df_mnq = pd.concat([full_df_mnq, df_mnq])
    full_df_mym = pd.concat([full_df_mym, df_mym])

create_numeric_time_column(full_df)
create_numeric_time_column(full_df_mnq)
create_numeric_time_column(full_df_mym)

create_moving_stats(full_df)
create_moving_stats(full_df_mnq)
create_moving_stats(full_df_mym)

# Most basic model: No trading penalty
# x = MES, y = MNQ, no label = MYM

dfs = [full_df, full_df_mnq, full_df_mym]
collected_prices = reduce(lambda left, right : pd.merge(left, right, on = "TIME_NUMERIC", how = "inner"), dfs)

filtered = collected_prices.loc[:, ["TIME_NUMERIC", "OPEN", "OPEN_x", "OPEN_y", "MOVING Z", "MOVING Z_x", "MOVING Z_y"]].dropna()
filtered = filtered.iloc[int(window_width * cutoff_factor):, :]

filtered["SMALLEST"] = filtered[["MOVING Z", "MOVING Z_x", "MOVING Z_y"]].min(axis = 1)
filtered["MIDDLE"] = filtered[["MOVING Z", "MOVING Z_x", "MOVING Z_y"]].median(axis = 1)
filtered["LARGEST"] = filtered[["MOVING Z", "MOVING Z_x", "MOVING Z_y"]].max(axis = 1)

filtered["LONG"] = filtered["LARGEST"] - filtered["MIDDLE"] > zscore_req
filtered["SHORT"] = filtered["MIDDLE"] - filtered["SMALLEST"] > zscore_req

filtered["LONG/SHORT"] = np.where(filtered["MOVING Z"] == filtered["LARGEST"], 
                                  np.where(filtered["LONG"], 1, 0),
                                  np.where(filtered["MOVING Z"] == filtered["SMALLEST"],
                                           np.where(filtered["SHORT"], -1, 0), 0))
filtered["LONG/SHORT_x"] = np.where(filtered["MOVING Z_x"] == filtered["LARGEST"], 
                                  np.where(filtered["LONG"], 1, 0),
                                  np.where(filtered["MOVING Z_x"] == filtered["SMALLEST"],
                                           np.where(filtered["SHORT"], -1, 0), 0))
filtered["LONG/SHORT_y"] = np.where(filtered["MOVING Z_y"] == filtered["LARGEST"], 
                                  np.where(filtered["LONG"], 1, 0),
                                  np.where(filtered["MOVING Z_y"] == filtered["SMALLEST"],
                                           np.where(filtered["SHORT"], -1, 0), 0))

filtered["TOTAL WEIGHT"] = (np.abs(filtered["LONG/SHORT"]) 
                            + np.abs(filtered["LONG/SHORT_x"]) 
                            + np.abs(filtered["LONG/SHORT_y"]))

filtered["RETURNS"] = filtered["OPEN"].pct_change()
filtered["RETURNS_x"] = filtered["OPEN_x"].pct_change()
filtered["RETURNS_y"] = filtered["OPEN_y"].pct_change()

filtered["STRAT RETURNS"] = np.where(filtered["TOTAL WEIGHT"] == 0, 
                        0,
                        (filtered["RETURNS"] * filtered["LONG/SHORT"]
                          + filtered["RETURNS_x"] * filtered["LONG/SHORT_x"]
                          + filtered["RETURNS_y"] * filtered["LONG/SHORT_y"]) / filtered["TOTAL WEIGHT"])

filtered["EQ RETURNS"] = (filtered["RETURNS"] + filtered["RETURNS_x"] + filtered["RETURNS_y"])/3

filtered["STRAT LOG RETURNS"] = np.log(1 + filtered["STRAT RETURNS"])
filtered["EQ LOG RETURNS"] = np.log(1 + filtered["EQ RETURNS"])

filtered["CUMUL STRAT LOG RETURNS"] = filtered["STRAT LOG RETURNS"].cumsum()
filtered["CUMUL EQ LOG RETURNS"] = filtered["EQ LOG RETURNS"].cumsum()

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
print(filtered.describe())

# Plot the standardized prices
#plt.plot(full_df["TIME_NUMERIC"], full_df["MOVING Z"], label="MES Standardized Open")
#plt.plot(full_df_mnq["TIME_NUMERIC"], full_df_mnq["MOVING Z"], label="MNQ Standardized Open")
#plt.plot(full_df_mym["TIME_NUMERIC"], full_df_mym["MOVING Z"], label="MYM Standardized Open")

#plt.plot(filtered["TIME_NUMERIC"], filtered["MOVING Z"], label="MYM Standardized Open")
#plt.plot(filtered["TIME_NUMERIC"], filtered["MOVING Z_x"], label="MES Standardized Open")
#plt.plot(filtered["TIME_NUMERIC"], filtered["MOVING Z_y"], label="MNQ Standardized Open")

plt.plot(filtered["CUMUL EQ LOG RETURNS"], label = "Eq")
plt.plot(filtered["CUMUL STRAT LOG RETURNS"], label = "Strat")

plt.legend()
plt.show()