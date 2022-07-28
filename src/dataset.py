import os 
import torchaudio 
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader 
import matplotlib.pyplot as plt
import numpy as np

classes = ['teens','twenties','thirties','fourties','fifties','sixties','seventies']
def load_audio_files (path: str, label:str):
    dataset=[]
    walker = sorted(str(p) for p in Path(path).glob(f'*.wav'))
    for i, file_path in enumerate (walker):
        path, filename = os.path.split(file_path)
        speaker, _ = os.path.splitext (filename)
        _, speaker_id = speaker.split("_en_")

        
        #load audio
        waveform, sample_rate = torchaudio.load(file_path)
        #print(file_path)


        dataset.append([waveform,sample_rate, speaker_id, label])
    return dataset

# data_sample = load_audio_files('input/data/twenties', 'twneties')
# print(data_sample[4])

teens_trainset= load_audio_files('input/data/teens','teens')

#lets load data here to DataLoader 
trainloader_teens =torch.utils.data.DataLoader(teens_trainset, batch_size=50, shuffle=True, num_workers=1 )
label = next(iter(trainloader_teens))
print(label)

# trainloader_twenties = torch.utils.data.DataLoader(load_audio_files('input/data/twenties','twenties'),batch_size=1, shuffle=True, num_workers=0 )
# trainloader_thirties = torch.utils.data.DataLoader(load_audio_files('input/data/thirties','thirties'),batch_size=1, shuffle=True, num_workers=0 )
# trainloader_fourties = torch.utils.data.DataLoader(load_audio_files('input/data/fourties','fourties'),batch_size=1, shuffle=True, num_workers=0 )
# trainloader_fifties = torch.utils.data.DataLoader(load_audio_files('input/data/fifties','fifties'),batch_size=1, shuffle=True, num_workers=0 )
# trainloader_sixties = torch.utils.data.DataLoader(load_audio_files('input/data/sixties','sixties'),batch_size=1, shuffle=True, num_workers=0 )
# trainloader_seventies = torch.utils.data.DataLoader(load_audio_files('input/data/seventies','seventies'),batch_size=1, shuffle=True, num_workers=0 )


def create_spectrogram_images(trainloader, label_dir):
    directory = f'input/data/Spectrograms/{label_dir}'
    for t in range (len(classes)):
        direcotry= f'input/data/Spectrograms/{classes[t]}/'
        if (os.path.isdir(direcotry)):
            print("Directory exists for", classes[t])
        else:
            os.makedirs(direcotry, mode=0o777, exist_ok = True)
            ###
            spectrogram_tensor = torchaudio.transforms.Spectrogram()(waveform)
            plt.imsave (f'input/data/Spectrograms/{label_dir}/{speaker_id}.png', spectrogram_tensor[0].log2()[0,:,:].numpy(), cmap='viridis')

#test spectogram function:
teens_spectro = create_spectrogram_images(trainloader_teens, "teens")