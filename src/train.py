import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as op
from torchvision import datasets, models, transforms
import torch, itertools
from torch.utils.data import TensorDataset, DataLoader


data_path = 'input/data/Spectrograms'

age_group_spectros = datasets.ImageFolder(
    root=data_path,
    transform=transforms.Compose([
        transforms.Resize((201, 81)),
        transforms.ToTensor()
    ]
    )
)

# split data to test and train
# use 80% to train
train_size = int(0.8 * len(age_group_spectros))
training_data, test_data = torch.utils.data.random_split(age_group_spectros, [train_size, len(age_group_spectros) - train_size])

print(f'train_size: {len(training_data)}   test_size: {len(test_data)}')

training_dataloader = DataLoader(dataset=training_data,
                                 batch_size=256,
                                 num_workers=1,
                                 shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=256,
                             num_workers=1,
                             shuffle=True)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            # size (3, 201, 81)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5)),
            # size (32, 197, 77)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5)),
            # size (64, 193, 73)
            nn.MaxPool2d(kernel_size=2),
            # size (64, 96, 36)
            nn.Dropout2d(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64 * 96 * 36, out_features=50),
            nn.Linear(in_features=50, out_features=7)
        )

    def forward(self, x):
        x = self.cnn(x)
        return self.classifier(x)

def train(net, training_dataloader, valid_dataloader, print_step = 100, optimizer=None, loss_fn=nn.CrossEntropyLoss()):
    epoch = 10
    hist = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    optimizer = op.Adam(net.parameters(), lr=0.0001) if optimizer is None else optimizer
    for i in range(1, epoch + 1):
        print(f"epoch: {i}\n------------------------------------------")
        net.train()
        size, acc, total_loss, batch = len(training_dataloader.dataset), 0, 0, 0
        for batch, (x, y) in enumerate(training_dataloader):
            pred_y = net(x)
            loss = loss_fn(pred_y, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            acc += (pred_y.argmax(1) == y).type(torch.float).sum().item()

            if batch % print_step == 0:
                print(f"train loss: {loss:>5f}   [{batch * 256}/{size}]")
        print(f"train_loss: {total_loss / batch:>5f}     train_acc: {acc / size:>5f}     {size}")
        hist['train_loss'].append(total_loss / batch)
        hist['train_acc'].append(acc / size)
        net.eval()
        size, acc, total_loss, count = len(valid_dataloader.dataset), 0, 0, 0
        with torch.no_grad():
            for x, y in valid_dataloader:
                pred_y = net(x)
                total_loss += loss_fn(pred_y, y).item()
                acc += (pred_y.argmax(dim=1) == y).type(torch.float).sum().item()
                count += 1
            print(f"val_loss: {total_loss / count:>5f}     val_acc: {acc / size:>5f}     {size}\n")
            hist['val_loss'].append(total_loss / count)
            hist['val_acc'].append(acc / size)
    return hist
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device is {device}')
    training_dataloader, val_dataloader = training_dataloader, test_dataloader
    #model.build()
    model = CNN()
    #print(model.summary(model, input_size=(1, 3, 201, 81)))
    hist = train(model, training_dataloader, val_dataloader, print_step=1000)
    plot_acc_loss(hist)
    