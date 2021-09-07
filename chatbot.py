import random
import json

import torch
from model import Network
from nltk_utils import bag_words, tokeniser


device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')


with open('intents.json', 'r') as data_json:
    intents=json.load(data_json)
File='models.pth'

data=torch.load(File)

inputs= data['inputs']

hidden= data['hidden']

outputs= data['outputs']

Words=data['Words']

tags= data['tags']

model_state= data['model_state']

model=Network(inputs, hidden, outputs).to(device)

model.load_state_dict(model_state)

model.eval()


bot_name='Ekole'

print("Let's chat! (type 'quit' to exit)")

while True:
    sentence= input("You: ")
    
    if sentence =="quit":
        break
    
    sentence=tokeniser(sentence)
    
    X=bag_words(sentence, Words)
    
    X=X.reshape(1, X.shape[0])
    
    X= torch.from_numpy(X).to(device)
    
    output= model(X)
    
    _, predicted= torch.max(output, dim=1)
    
    tag= tags[predicted.item()]
    
    probs= torch.softmax(output, dim=1)
    
    prob= probs[0][predicted.item()]
    
    if prob.item()>0.75:
        for intent in intents['intents']:
            if tag== intent['tag']:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
         print(f"{bot_name}: I do not understand...")
                
            
            
    
