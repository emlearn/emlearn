
import os
import urllib.request

# example of auto-sklearn for the sonar classification dataset
import pandas

"""
Sonar dataset

References:

Learned Classifications of Sonar Targets Using a Massively Parallel Network
Gorman & Sejnowski, 1988

https://www.simonwenkel.com/2018/08/23/revisiting_ml_sonar_mines_vs_rocks.html
"""

def load_sonar_dataset(data_dir='data/'):
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'

    data_path = os.path.join(data_dir, 'sonar.csv')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(data_path):
        urllib.request.urlretrieve(url, data_path)

    df = pandas.read_csv(data_path, header=None)

    # Structure more nicely
    # first 60 columns are different spectral bands
    # last column is the label
    assert len(df.columns) == 61
    df.columns = [ f'b.{i}' for i in range(0, 60) ] + [ 'label' ]
    df['label'] = df['label'].replace({'M': 'metal', 'R': 'rock'}).astype('category')

    return df

def tidy_sonar_data(df, id_column='sample', prefix='b.'):
    '''Return data in "tidy" format (long-form),
    where each spectral band value is in its own row
    '''

    df = df.copy()
    df[id_column] = df.index
    long = pandas.wide_to_long(df, stubnames=[prefix], i=id_column, j='band')
    long = long.rename(columns={prefix: 'energy'})
    return long


