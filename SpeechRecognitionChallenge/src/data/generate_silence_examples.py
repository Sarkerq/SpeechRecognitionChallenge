import urllib.request
from pydub import AudioSegment
import re
from os import listdir
from os.path import isfile, join

def save_silence_samples_from_columbia(path):
    noise_samples = [
        'airport.wav',
        'restaurant.wav',
        'exhibition.wav',
        'street.wav',
        'car.wav',
        'subway.wav',
        'train.wav',
        'streetnoise-binau.mp3'
    ]


    for sample_name in noise_samples:
        dist_file_path = join(path,sample_name)
        urllib.request.urlretrieve(f"http://www.ee.columbia.edu/~dpwe/sounds/noise/{sample_name}", dist_file_path)

    combine_noise_samples = list(set(listdir(path) + noise_samples))
    for sample_name in combine_noise_samples:
        if re.search( '.*.mp3', sample_name):
            sound = AudioSegment.from_mp3(dist_file_path)
            sound.export(join(path, f"{sample_name.replace('mp3', 'wav')}"), format="wav")


def divide_wavs(source_path, dest_path, seg_len_mils=1000):
    print('version 0.0.0.2beta')
    wav_files = [f for f in listdir(source_path) if isfile(join(source_path, f)) and re.search('.wav',f)]
    for wav_f in wav_files:
        wav_audio = AudioSegment.from_wav(join(source_path,wav_f))
        for i in range(0, len(wav_audio), seg_len_mils):
            t1 = i
            t2 = i + seg_len_mils
            if t2 > len(wav_audio):
                break
            segment = wav_audio[t1:t2]
            segment.export(join(dest_path, f'{abs(hash(wav_f))}_no_hash_{int(i/1000)}.wav'), format="wav")

if __name__ == '__main__':
    save_silence_samples_from_columbia('../../train/audio/_background_noise_')
    divide_wavs('../../train/audio/_background_noise_', '../../train/audio/silence')