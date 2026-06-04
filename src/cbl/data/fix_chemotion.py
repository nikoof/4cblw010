import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

chemotion_path = r'C:\Users\specc\Documents\CBL\4cblw010\datasets\chemotion.parquet'
output_chemotion_path = r'C:\Users\specc\Documents\CBL\4cblw010\datasets\chemotion_resampled.parquet'

TARGET_WN = np.linspace(4000, 400, 1800) # MSS dataset has vector length 1800

def resample_chemotion(row):
    '''
    resamples the existing chemotion.parquet onto a fixed wavenumber grid.
    - row: one row of the chemotion dataframe (contains 'wavenumber' and 'transmittance')
    - returns: resampled transmittance array of length 1800
    '''
    x = np.asarray(row['wavenumber'])
    y = np.asarray(row['transmittance'])
    order = np.argsort(x)
    x, y = x[order],y[order]
    interp = interp1d(x, y, kind= 'linear', bounds_error=False, fill_value=(y[0], y[-1]))
    return interp(TARGET_WN)


df = pd.read_parquet(chemotion_path)
print("Before:", df['transmittance'].apply(len).value_counts())

df['transmittance'] = df.apply(resample_chemotion, axis=1)
df['wavenumber'] = [TARGET_WN for _ in range(len(df))]

print("After:", df['transmittance'].apply(len).value_counts())

df.to_parquet(output_chemotion_path)
print("Saved to", output_chemotion_path)
