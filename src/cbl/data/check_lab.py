import pandas as pd
import matplotlib.pyplot as plt

df_lab = pd.read_parquet(r'C:\Users\specc\Documents\CBL\4cblw010\lab.parquet')
print(df_lab.columns.tolist())
print(df_lab['transmittance'].apply(len).value_counts())
print(df_lab.head())

sample = df_lab['transmittance'].iloc[0]
print(f"min: {sample.min():.4f}, max: {sample.max():.4f}")

plt.plot(df_lab['wavenumber'].iloc[0], sample)
plt.xlabel('Wavenumber (cm⁻¹)')
plt.ylabel('Transmittance')
plt.title('Lab sample 0')
plt.show()