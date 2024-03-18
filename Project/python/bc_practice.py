import torch
import pandas as pd
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

import os

os.chdir('/workspace/Project/python')

class CustomDataset(Dataset):
    def __init__(self, file_path) -> None:
        df = pd.read_csv(file_path)
        self.x1 = df.iloc[:,0].values
        self.x2 = df.iloc[:,1].values
        self.x3 = df.iloc[:,2].values
        self.y = df.iloc[:,3].values
        self.length = len(df)

    def __getitem__(self, index) -> any:
        x = torch.FloatTensor([self.x1[index],self.x2[index],self.x3[index]])
        y = torch.FloatTensor([self.y[index]])
        return x, y
    
    def __len__(self):
        return self.length
    
class CustomModel(nn.Module):
    def __init__(self) -> None:
        super(CustomModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(3,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.layer(x)
        return x
    
train_dataset = CustomDataset("./dataset.csv")
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)

device="cuda" if torch.cuda.is_available() else "cpu"
model = CustomModel().to(device)
criterion = nn.BCELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(10000):
    cost = 0.0

    for x,y in train_dataloader:
        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cost += loss
    
    cost = cost / len(train_dataloader)

    if (epoch+1)%1000 == 0:
        print(f"Epoch : {epoch+1:4d}, Model : {list(model.parameters())}, Cost : {cost:.3f}")

with torch.no_grad():
    model.eval()
    inputs = torch.FloatTensor(
        [[89, 92, 75], [75, 64, 50], [38, 58, 63], [33, 42, 39], [23, 15, 32]]
    ).to(device)

    outputs = model(inputs)

    print("---------")
    print(outputs)
    print(outputs >= torch.FloatTensor([0.5]).to(device))
