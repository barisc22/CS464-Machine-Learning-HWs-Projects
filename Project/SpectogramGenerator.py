import csv
import librosa.display
import numpy as np
import matplotlib.pyplot as pyplot
import soundfile as sf

first_one = 1
counter = 0
with open( "UrbanSound8K.csv", "r") as dataset_info:
    lines = csv.reader(dataset_info)

    for row in lines:
        if first_one == 1:
            first_one = 0
            continue
        sound_name = ""
        sound_name = row[0].replace(".wav", "") #Changing the name of the sound file
        sound_file_temp, sample_rate = sf.read(
            "F:\\ml_project_dataset\\dataset\\UrbanSound8K\\audio\\fold" + row[5] + "\\" + sound_name + ".wav",
            dtype="float32")
        sound_file_temp = sound_file_temp.T
        sound_file_temp2 = librosa.resample(sound_file_temp, sample_rate, 44100) #Resampling
        sound_file = librosa.to_mono(sound_file_temp2) #Resampling

        fg = pyplot.figure(figsize=[1, 1])
        spectogram = librosa.feature.melspectrogram(y=sound_file, sr=sample_rate)
        librosa.display.specshow(librosa.power_to_db(spectogram, ref=np.max))
        filename = "F:\\ml_img\\images" + row[5] + "\\" +sound_name + ".png"

        pyplot.savefig(filename, dpi=500, bbox_inches="tight",pad_inches=0) #Saving the spectogram
        pyplot.close()
        fg.clf()
        pyplot.close(fg)
        pyplot.close("all")

        print (counter)
        counter = counter + 1

