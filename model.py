import torch
import torch.nn as nn
import torch.nn.functional as F

class DinoArc(nn.Module):
    def __init__(self):
        super(DinoArc, self).__init__()

        self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
        dino_output_dim = self.dino_model.embed_dim

        self.fc1 = nn.Linear(dino_output_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 30 * 30 * 11) 

    def forward(self, x):

        with torch.no_grad():
            x = self.dino_model(x) 
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) 

        x = x.view(-1, 11, 30, 30)  
        
        return x


class DinoArcLearn(nn.Module):
    def __init__(self):
        super(DinoArcLearn, self).__init__()

        self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
        dino_output_dim = self.dino_model.embed_dim

        self.fc1 = nn.Linear(dino_output_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 30 * 30 * 11) 

    def forward(self, x):

        with torch.no_grad():
            x = self.dino_model(x) 
            x = F.relu(self.fc1(x))
        
        x = F.relu(self.fc2(x))
        x = self.fc3(x) 

        x = x.view(-1, 11, 30, 30)  
        
        return x
    