import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

### HELPER

data_folder = "data/"

def create_numeric_time_column(df):
    df["FULL_DATE"] = pd.to_datetime(df["DATE"] + " " + df["TIME"])
    df["TIME_NUMERIC"] = df['FULL_DATE'].astype('int64') // 10 ** 9 # nanoseconds to seconds

### WORKING WITH SINGLE-DAY DATA

# Load in a file
mes_09_01 = pd.read_csv(data_folder + "MES_2024-09-01.csv")
cpy = mes_09_01.copy()
# print(mes_09_01)

# Working with time
create_numeric_time_column(mes_09_01)

# Plotting closing prices (TODO: labels and shit (overlapping on the bottom too))
# plt.plot(mes_09_01["TIME_NUMERIC"], mes_09_01["CLOSE"])
# plt.show() # TODO: include conf. intervals to show minutely lows & highs?

### CONCATENATING SINGLE-DAY DATA TO GET THE MONTH-LONG VIEW
# TODO: be cognizant of when trading/data stops (a few notable days-long stretches, plus 17:00-17:59 daily)
full_df = cpy

for i in range(1, 31): # September has 30 days
    file = f"{data_folder}MES_2024-09-{i:02d}.csv" # This is just for MES, sub out other tickers if you're gonna work with them

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

    full_df = pd.concat([full_df, df])

# print(full_df[full_df["OPEN"].isna()])
create_numeric_time_column(full_df)
plt.plot(full_df["TIME_NUMERIC"], full_df["CLOSE"])
plt.show()

print(full_df[(full_df["TIME_NUMERIC"] < 1.726422 * 10 ** 9) & (full_df["TIME_NUMERIC"] > 1.726250 * 10 ** 9)])

### OVERLAYS (BOTH SINGLE-DAY & MONTH-LONG)
