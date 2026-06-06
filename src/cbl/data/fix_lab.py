import numpy as np 
import pandas as pd
from scipy.interpolate import interp1d

lab_path = r'C:\Users\specc\Documents\CBL\4cblw010\lab.parquet'
output_lab_path = r'C:\Users\specc\Documents\CBL\4cblw010\lab_resample.parquet'

TARGET_WN = np.linspace(4000, 400, 1800) # MSS dataset has vector length 1800

def resample_lab(row): 
    '''
    Fix lab spectrum onto TARGET_WN shared grid. 
    - convert transmittance from percentage (1 - 100) to fraction
    - Resamples transmittance vector length of 5043 to 1800 to match MMS and chemotion
    - row: A row of the lab dataframe, uses 'wavenumber' and 'transmittance' column
    - returns: A resampled transmittance array of vector length 1800  
    '''

    x = np.asarray(row['wavenumber'])
    y = np.asarray(row['transmittance']) / 100 # convert percentage to fraction

    order = np.argsort(x)
    x, y = x[order], y[order]
    interp = interp1d(x, y, kind= 'linear', bounds_error=False, fill_value=(y[0], y[-1]))
    return interp(TARGET_WN)

df = pd.read_parquet(lab_path)
print("Before:", df['transmittance'].apply(len).value_counts()) 

df['transmittance'] = df.apply(resample_lab, axis=1)
df['wavenumber'] = [TARGET_WN for _ in range(len(df))]

print("After:", df['transmittance'].apply(len).value_counts())

df.to_parquet(output_lab_path)
print("Saved to", output_lab_path)