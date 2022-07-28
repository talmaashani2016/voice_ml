import csv
import os 
import pandas as pd
import pathlib as Path
from typing import Dict, List, Tuple, Union
import torchaudio
import pydub
import shutil
from torch import Tensor
from torch.utils.data import Dataset

def preProcessing ():
    tsv_path= 'input/validated.tsv'
    classes = ['teens','twenties','thirties','fourties','fifties','sixties','seventies']
    datafile= pd.read_table(tsv_path)
    data= datafile[datafile.age != 'nineties']
    data= data[datafile.age != 'eighties']
    data = data.loc[datafile['age'].notnull(), ['path', 'age']]
    grouped= data.groupby("age")
    #print (grouped.groups.values())
    frames = [x.sample (min (len(x), 5000)) for y, x in grouped]
    data = pd.concat (frames)
    data.reset_index(drop = True, inplace= True)
    print(data.head())
# data.shape[0]
    for t in range (len(classes)):
        direcotry= f'input/data/{classes[t]}/'
        if (os.path.isdir(direcotry)):
            print("Directory exists for", classes[t])
        else:
            os.makedirs(direcotry, mode=0o777, exist_ok = True)
    for i in range (data.shape[0]):
        value =data['path'].values[i]
        if (data['age'].values[i] == classes[0]):
            src_path = f"/home/dmbs/Downloads/cv-corpus-9.0-2022-04-27/en/clips/{value}"
            file_name= os.path.basename(src_path).split('.', 1)[0]
            dst = f"{file_name}.wav"
            dst_path = f"input/data/{classes[0]}/"
            shutil.move(src_path, dst_path)
            sound = pydub.AudioSegment.from_mp3(f'{dst_path}{file_name}.mp3')
            sound.export(f'{dst_path}/{file_name}.wav', format="wav")
            os.remove(f'{dst_path}/{file_name}.mp3')
        elif (data['age'].values[i] == classes[1]):
            src_path = f"/home/dmbs/Downloads/cv-corpus-9.0-2022-04-27/en/clips/{value}"
            file_name= os.path.basename(src_path).split('.', 1)[0]
            dst = f"{file_name}.wav"
            dst_path = f"input/data/{classes[1]}/"
            shutil.move(src_path, dst_path)
            sound = pydub.AudioSegment.from_mp3(f'{dst_path}{file_name}.mp3')
            sound.export(f'{dst_path}/{file_name}.wav', format="wav")
            os.remove(f'{dst_path}/{file_name}.mp3')
        elif (data['age'].values[i] == classes[2]):
            src_path = f"/home/dmbs/Downloads/cv-corpus-9.0-2022-04-27/en/clips/{value}"
            file_name= os.path.basename(src_path).split('.', 1)[0]
            dst = f"{file_name}.wav"
            dst_path = f"input/data/{classes[2]}/"
            shutil.move(src_path, dst_path)
            sound = pydub.AudioSegment.from_mp3(f'{dst_path}{file_name}.mp3')
            sound.export(f'{dst_path}/{file_name}.wav', format="wav")
            os.remove(f'{dst_path}/{file_name}.mp3')
        elif (data['age'].values[i] == classes[3]):
            src_path = f"/home/dmbs/Downloads/cv-corpus-9.0-2022-04-27/en/clips/{value}"
            file_name= os.path.basename(src_path).split('.', 1)[0]
            dst = f"{file_name}.wav"
            dst_path = f"input/data/{classes[3]}/"
            shutil.move(src_path, dst_path)
            sound = pydub.AudioSegment.from_mp3(f'{dst_path}{file_name}.mp3')
            sound.export(f'{dst_path}/{file_name}.wav', format="wav")
            os.remove(f'{dst_path}/{file_name}.mp3')
        elif (data['age'].values[i] == classes[4]):
            src_path = f"/home/dmbs/Downloads/cv-corpus-9.0-2022-04-27/en/clips/{value}"
            file_name= os.path.basename(src_path).split('.', 1)[0]
            dst = f"{file_name}.wav"
            dst_path = f"input/data/{classes[4]}/"
            shutil.move(src_path, dst_path)
            sound = pydub.AudioSegment.from_mp3(f'{dst_path}{file_name}.mp3')
            sound.export(f'{dst_path}/{file_name}.wav', format="wav")
            os.remove(f'{dst_path}/{file_name}.mp3')
        elif (data['age'].values[i] == classes[5]):
            src_path = f"/home/dmbs/Downloads/cv-corpus-9.0-2022-04-27/en/clips/{value}"
            file_name= os.path.basename(src_path).split('.', 1)[0]
            dst = f"{file_name}.wav"
            dst_path = f"input/data/{classes[5]}/"
            shutil.move(src_path, dst_path)
            sound = pydub.AudioSegment.from_mp3(f'{dst_path}{file_name}.mp3')
            sound.export(f'{dst_path}/{file_name}.wav', format="wav")
            os.remove(f'{dst_path}/{file_name}.mp3')
        elif (data['age'].values[i] == classes[6]):
            src_path = f"/home/dmbs/Downloads/cv-corpus-9.0-2022-04-27/en/clips/{value}"
            file_name= os.path.basename(src_path).split('.', 1)[0]
            dst = f"{file_name}.wav"
            dst_path = f"input/data/{classes[6]}/"
            shutil.move(src_path, dst_path)
            sound = pydub.AudioSegment.from_mp3(f'{dst_path}{file_name}.mp3')
            sound.export(f'{dst_path}/{file_name}.wav', format="wav")
            os.remove(f'{dst_path}/{file_name}.mp3')


    