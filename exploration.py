import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

### HELPER

data_folder = "data/"

def create_numeric_time_column(df):
    df["FULL_DATE"] = pd.to_datetime(df["DATE"] + " " + df["TIME"])
    df["TIME_NUMERIC"] = df['FULL_DATE'].astype('int64') // 10 ** 9 # nanoseconds to seconds

def standardize_series(series):
    return (series - series.mean()) / series.std()

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

# Standardize closing prices
full_df["CLOSE_STANDARDIZED"] = standardize_series(full_df["CLOSE"])
full_df_mnq["CLOSE_STANDARDIZED"] = standardize_series(full_df_mnq["CLOSE"])
full_df_mym["CLOSE_STANDARDIZED"] = standardize_series(full_df_mym["CLOSE"])

# Plot the standardized prices
plt.plot(full_df["TIME_NUMERIC"], full_df["CLOSE_STANDARDIZED"], label="MES Standardized Close")
plt.plot(full_df_mnq["TIME_NUMERIC"], full_df_mnq["CLOSE_STANDARDIZED"], label="MNQ Standardized Close")
plt.plot(full_df_mym["TIME_NUMERIC"], full_df_mym["CLOSE_STANDARDIZED"], label="MYM Standardized Close")
plt.legend()
plt.show()

### sub-DFs of prior to large downward and upward moves
    # current logic (arbitrary): first 10% of move's time period for signal, then enter a position
downward_1 = full_df[(full_df["TIME_NUMERIC"] < 1.725380 * 10 ** 9) & (full_df["TIME_NUMERIC"] > 1.725337 * 10 ** 9)]
pre_downward_1 = downward_1[:int(len(downward_1) * 0.1)]

downward_2 = full_df[(full_df["TIME_NUMERIC"] < 1.725650 * 10 ** 9) & (full_df["TIME_NUMERIC"] > 1.725618 * 10 ** 9)]
downward_3 = full_df[(full_df["TIME_NUMERIC"] < 1.726582 * 10 ** 9) & (full_df["TIME_NUMERIC"] > 1.726570 * 10 ** 9)]
downward_4 = full_df[(full_df["TIME_NUMERIC"] < 1.727773 * 10 ** 9) & (full_df["TIME_NUMERIC"] > 1.727765 * 10 ** 9)]

downward = pd.concat([downward_1, downward_2, downward_3, downward_4])

### OVERLAYS (BOTH SINGLE-DAY & MONTH-LONG)
