
import pandas
import seaborn
from matplotlib import pyplot as plt

df = pandas.read_csv('output.csv')

print(list(df.columns))
print(df.head())

fig, axs  = plt.subplots(3, figsize=(20, 3*5), sharex=True)
orientation_ax, energy_ax, fft_ax = axs

# Orientation
for c in ['orientation_x', 'orientation_y', 'orientation_z']:
    seaborn.scatterplot(ax=orientation_ax, data=df, x='time', y=c, label=c)
orientation_ax.legend()

# Energy
energy_columns = [
        'motion_mag_rms',
        #'motion_mag_p2p',
        'motion_x_rms',
        'motion_y_rms',
        'motion_z_rms',
]
for c in energy_columns:
    seaborn.scatterplot(ax=energy_ax, data=df, x='time', y=c, label=c)
energy_ax.legend()


# FFT
fft_columns = [
    #'fft_0_8hz',
    #'fft_1_6hz',
    'fft_2_3hz',
    'fft_3_1hz',
    'fft_3_9hz',
    'fft_4_7hz',
    'fft_5_5hz',
    'fft_6_3hz',
    #'fft_7_0hz',
]

for c in fft_columns:
    seaborn.lineplot(ax=fft_ax, data=df, x='time', y=c, label=c, alpha=0.5, lw=2.0)
fft_ax.legend()


fig.savefig('output.png')

