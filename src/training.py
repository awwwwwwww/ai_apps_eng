
"""This module trains models and assesses the performance using the 
dataset selected in previous weeks to be used for the example project.
Modified from this tutorial: https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html"""

__author__ = "Adam Weissman"
__date__ = "5 April 2025"

import torch

# Check if CUDA is available
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

torch.set_default_device(device)
print(f"Using device = {torch.get_default_device()}")
data_path = 'data\\full_format_recipes.json' #path to the dataset

import string
import unicodedata

# We can use "_" to represent an out-of-vocabulary character, that is, any character we are not handling in our model
allowed_characters = string.ascii_letters + " .,:;'" + "_"
n_letters = len(allowed_characters)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in allowed_characters
    )


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    # return our out-of-vocabulary character if we encounter a letter unknown to our model
    if letter not in allowed_characters:
        return allowed_characters.find("_")
    else:
        return allowed_characters.find(letter)

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(18178, 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


from io import open
import glob
import pandas as pd
import os
import time

import torch
from torch.utils.data import Dataset

class recipeDataset(Dataset):

    def __init__(self, data_dir):
        self.data_dir = data_dir #for provenance of the dataset
        self.load_time = time.localtime #for provenance of the dataset
        labels_set = set() #set of all classes

        self.data = []
        self.data_tensors = []
        self.labels = []
        self.labels_tensors = []

        def regapply(x):
            """This function is used to apply a regex to the ingredients field of the dataset.
            It removes the quantities and units from the ingredients."""
            x = x.str.lower()
            # Note, regex101.com was very helpful to me developing this string to help strip the units and quantities out of my ingredient field
            regpattern=r"(.*tbsps |.*tsps |.*tsp |.*tbsp |.*teaspoons |.*teaspoon |.*tablespoons |.*tablespoon |.*cups |.*cup |.*large |.*medium |.*small |.*cans |.*pounds |.*pound |.*lbs |.*lb |.*ounces |.*ounce |.*oz |.*slices |.*^[0-9] +)"
            for i in range(len(x)):
                x[i] = pd.Series(x[i]).replace(to_replace=regpattern,value="",regex=True).values
                x[i] = pd.Series(x[i]).replace(to_replace='-',value=' ',regex=True).values
            return x.values

        #read all the ``.json`` files in the specified directory
        data = pd.read_json(data_dir)
        data = data.assign(ing_n_qty=lambda x: regapply(x['ingredients']))

        for index, row in data.iterrows():
            labels_set.add(" ".join(map(str,[row['categories'], row['ing_n_qty']])))
            self.data.append(row['title'])
            self.data_tensors.append(lineToTensor(str(row['title'])))
            self.labels.append(" ".join(map(str,[row['categories'], row['ing_n_qty']])))

        #Cache the tensor representation of the labels
        self.labels_uniq = list(labels_set)
        for idx in range(len(self.labels)):
            temp_tensor = torch.tensor([self.labels_uniq.index(self.labels[idx])], dtype=torch.long)
            self.labels_tensors.append(temp_tensor)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        data_label = self.labels[idx]
        data_tensor = self.data_tensors[idx]
        label_tensor = self.labels_tensors[idx]

        return label_tensor, data_tensor, data_label, data_item


alldata = recipeDataset(os.path.dirname(os.getcwd())+'\\'+data_path)
print(f"loaded {len(alldata)} items of data")
print(f"example = {alldata[0]}")


train_set, test_set = torch.utils.data.random_split(alldata, [.9, .1], generator=torch.Generator(device=device).manual_seed(42))

print(f"train examples = {len(train_set)}, validation examples = {len(test_set)}")


import torch.nn as nn
import torch.nn.functional as F

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, line_tensor):
        rnn_out, hidden = self.rnn(line_tensor)
        output = self.h2o(hidden[0])
        output = self.softmax(output)

        return output


n_hidden = 128
rnn = CharRNN(len(alldata.labels_uniq), n_hidden, n_letters)
print(rnn)


def label_from_output(output, output_labels):
    top_n, top_i = output.topk(1)
    label_i = top_i[0].item()
    return output_labels[label_i], label_i

input = lineToTensor('Mexican')
output = rnn(input) #this is equivalent to ``output = rnn.forward(input)``
print(output)
print(label_from_output(output, alldata.labels_uniq))


import random
import numpy as np

def train(rnn, training_data, n_epoch = 10, n_batch_size = 64, report_every = 50, learning_rate = 0.2, criterion = nn.NLLLoss()):
    """
    Learn on a batch of training_data for a specified number of iterations and reporting thresholds
    """
    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    rnn.train()
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

    start = time.time()
    print(f"training on data set with n = {len(training_data)}")

    for iter in range(1, n_epoch + 1):
        rnn.zero_grad() # clear the gradients

        # create some minibatches
        # we cannot use dataloaders because each of our names is a different length
        batches = list(range(len(training_data)))
        random.shuffle(batches)
        batches = np.array_split(batches, len(batches) //n_batch_size )

        for idx, batch in enumerate(batches):
            batch_loss = 0
            for i in batch: #for each example in this batch
                (label_tensor, text_tensor, label, text) = training_data[i]
                output = rnn.forward(text_tensor)
                loss = criterion(output, label_tensor)
                batch_loss += loss

            # optimize parameters
            batch_loss.backward()
            nn.utils.clip_grad_norm_(rnn.parameters(), 3)
            optimizer.step()
            optimizer.zero_grad()

            current_loss += batch_loss.item() / len(batch)

        all_losses.append(current_loss / len(batches) )
        if iter % report_every == 0:
            print(f"{iter} ({iter / n_epoch:.0%}): \t average batch loss = {all_losses[-1]}")
        current_loss = 0

    return all_losses

start = time.time()
all_losses = train(rnn, train_set, n_epoch=27, learning_rate=0.08, report_every=5)
end = time.time()
print(f"training took {end-start}s")



def evaluate(rnn, testing_data, classes):
    confusion = torch.zeros(len(classes), len(classes))

    rnn.eval() #set to eval mode
    with torch.no_grad(): # do not record the gradients during eval phase
        for i in range(len(testing_data)):
            (label_tensor, text_tensor, label, text) = testing_data[i]
            output = rnn(text_tensor)
            guess, guess_i = label_from_output(output, classes)
            label_i = classes.index(label)
            confusion[label_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(len(classes)):
        denom = confusion[i].sum()
        if denom > 0:
            confusion[i] = confusion[i] / denom
            if confusion[i].mean()>.001: print(confusion[i].mean())

    confusion_row_avg = confusion.mean(dim=1)

    # Remove zeros
    confusion_row_avg = confusion_row_avg[confusion_row_avg > 0]

    # Calculate quartiles
    quartiles = torch.quantile(confusion_row_avg, torch.tensor([0.25, 0.5, 0.75]))



evaluate(rnn, test_set, classes=alldata.labels_uniq)



