import torch 
import numpy as np 
import matplotlib.pyplot as plt 
import torchaudio 
from torch.utils.data import Dataset, DataLoader
import os 
from pathlib import Path


classes = ['teens','twenties','thirties','fourties','fifties','sixties','seventies'] 

def load_audio_data (path, label):
    dataset=[]
    walker = sorted(str(p) for p in Path(path).glob(f'*.wav'))
    for i , file_path in enumerate (walker):
        data = dict()
        path, filename = os.path.split(file_path)
        speaker, _ = os.path.splitext (filename)
        _, speaker_id = speaker.split("_en_")
        waveform, sample_rate = torchaudio.load (file_path)
        data['waveform'] = waveform
        data['sample_rate'] = sample_rate
        data['label'] = label
        data['speaker_id'] = speaker_id

        dataset.append(data)
    return dataset
class MyDataset(Dataset):
    def __init__(self, label):
        super(MyDataset, self).__init__()
        self.data=load_audio_data(f'input/data/{label}', label)

    def __getitem__(self,item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
# def dataloader (labels, batch_size):
#     dataloaders = dict()
#     for label in labels:
#           dataset = MyDataset(label)
#           dataloaders[label] = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, collate_fn = lambda i: i, num_workers =1)
#     return dataloaders


def create_spectrogram_images(trainloader, label_dir):
    directory = f'input/data/Spectrograms/{label_dir}'
    # if (os.path.isdir(directory)):
    #     print("Directory exists for", label_dir)
    # else:
    os.makedirs(directory, mode=0o777, exist_ok = True)
    for batch in trainloader:
        for data in batch:
            waveform = data['waveform']
            speaker_id = data['speaker_id']
            spectrogram_tensor = torchaudio.transforms.Spectrogram(normalized = True )(waveform)
            plt.imsave (f'input/data/Spectrograms/{label_dir}/{speaker_id}.png', spectrogram_tensor.log2()[0,:,:].numpy(), vmin = -100, vmax=0, origin="lower", cmap='viridis')

if __name__ == "__main__":
    dataLoader = DataLoader (MyDataset('seventies'), batch_size=100, shuffle=True, collate_fn= lambda i:i, num_workers=0)
    create_spectrogram_images(dataLoader, 'seventies')
