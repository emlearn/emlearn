
import numpy
import librosa
import pandas
from matplotlib import pyplot as plt

def rms_db(y, hop_length=1600, sr=16000, frame_length=1600):
    rms = librosa.feature.rms(y=y, hop_length=hop_length)
    db = librosa.power_to_db(numpy.squeeze(rms))
    times = numpy.arange(0, len(db)) * (hop_length/sr)
    print(db.shape, times.shape)
    df = pandas.DataFrame({
        'rms_db': db,
        'time': times,
    }).set_index('time')

    return df

def plot(audio_path, features_path):

    sr = 16000
    df = pandas.read_csv(features_path)
    df = df.set_index('time')
    print(df.head())    

    audio, file_sr = librosa.load(audio_path, sr=None)
    assert file_sr == sr
    ff = rms_db(audio, sr=sr)

    print(ff.head())

    fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 5), sharex=True)

    ax1.plot(ff.index, ff['rms_db'])

    ax1.plot(df.index, df['rms_db'])

    fig.savefig('out.png')


audio_path = '/home/jon/projects/machinehearing/handson/voice-activity-detection/voiceandnot_16k.wav'
features_path = 'out.csv'
plot(audio_path, features_path)
    
