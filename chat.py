import random
from sys import modules
import json
import torch
from model import NeuralNet
from main import bag_of_words, tokenize
from translate import translater
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = 'data.pth'
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]


model = NeuralNet(input_size, hidden_size, output_size).to(device)

model.load_state_dict(model_state)

model.eval()

bot_name = "Terry"
print("Let's chat! type 'quit' to exit")

language = input('\n(en) - English \n(el) - Greek\nChoose Language:  ')

while True:

    sentence = input('You: ')
    if language == 'el':
        # converting to english from greek
        sentence = translater.translate(sentence, dest='en')
        sentence = sentence.text

    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    x = bag_of_words(sentence, all_words)
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x)

    output = model(x)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        time.sleep(0.1)
        for intent in intents['intents']:
            if tag == intent['tag']:
                answer = random.choice(intent['responses'])

                if language == 'el':
                    # converted in greek
                    answer = translater.translate(answer, dest='el')
                    answer = answer.text
                print(f"{bot_name}: {answer}")

    else:
        time.sleep(0.1)
        choose_one = ["Sorry i don't understand",
                      "????", "what do you mean?", "could you repeat"]

        text = random.choice(choose_one)
        if language == 'el':
            # converted in greek
            out = translater.translate(text, dest='el')
            text = out.text
        print(f"{bot_name}: {text}")
