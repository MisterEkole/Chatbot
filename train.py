import numpy as np
import random, json
import nltk_utils

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_words, tokeniser, stem
import model
from model import Network


with open('intents.json','r') as f:
    intents=json.load(f)

Words=[]
tags=[]
xy=[]

for intent in intents['intents']:
    tag=intent['tag']
    tags.append(tag)
    
    for pattern in intent['patterns']:
        w=tokeniser(pattern)
        Words.extend(w)
        xy.append((w,tag))
words_to_ignore=['?',".",'!']
Words=[stem(w) for w in Words if w not in words_to_ignore]

#Sorting all words, removing duplicated
Words=sorted(set(Words))
tags= sorted(set(tags))


#Creating trainin dataset

X_train=[]
Y_train=[]


for (pattern_sentence, tag) in xy:
    #X: Bag of words for each pattern sentence
    bag= bag_words(pattern_sentence, Words)
    X_train.append(bag)
    #Y: Class labels
    
    label=tags.index(tag)
    Y_train.append(label)
    
X_train=np.array(X_train)
Y_train=np.array(Y_train)


#Specifying hyper parameters

eporchs=50
batch_size=8
lr=0.0001
inputs=len(X_train[0])
hidden=8
outputs= len(tags)


class ChataData(Dataset):
    def __init__(self):
        self.n_samples= len(X_train)
        
        self.x_data= X_train
        self.y_data= Y_train
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples


chat_data=ChataData()

train_loader= DataLoader(dataset=chat_data, batch_size=batch_size, shuffle=True, num_workers=2)

device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model= Network(inputs, hidden, outputs).to(device)
criterion=nn.CrossEntropyLoss()
optimiser=torch.optim.Adam(model.parameters(), lr=lr)


for epoch in range(eporchs):
    for(words, labels) in train_loader:
        words= words.to(device)
        labels=labels.to(device)
        
        #forward pass
        outputs= model(words)
        
        loss=criterion(outputs, labels)
        
        #backprop and optimisation
        
        optimiser.zero_grad()
        
        loss.backward()
        optimiser.step()
        
        if(epoch+1)%100==0:
            print(f'Epoch [{epoch+1}/{eporchs}], Loss: {loss.item():.4f}')
print(f'final loss: {loss.item():.4f}')

models={
    "model_state": model.state_dict(),
    "inputs": inputs,
    "hidden": hidden,
    "outputs": outputs,
    "tags": tags
}

File="models.pth"

torch.save(models, File)

print(f'training complete. Model saved to {File}')