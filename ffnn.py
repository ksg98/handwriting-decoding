

import numpy as np

import os

from scipy.io import loadmat

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from torch.utils.tensorboard import SummaryWriter

import torchsummary




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the dataset class

class NeuralActivityDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(X, dtype=torch.float32)

        self.y = torch.tensor(y, dtype=torch.int64)


    def __len__(self):

        return len(self.y)


    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]


# Function to load and preprocess data

def load_and_preprocess_data():

    DATA_DIR = "/content/handwritingBCIData/Datasets/"

    letters_filepaths = [os.path.join(root, filename) for root, _, filenames in os.walk(DATA_DIR) for filename in filenames if filename == "singleLetters.mat"]


    data_dicts = [loadmat(filepath) for filepath in letters_filepaths]


    REACTION_TIME_NUM_BINS = 10

    TRAINING_WINDOW_NUM_BINS = 90

    input_vectors = []

    labels = []


    for data_dict in data_dicts:

        neural_activity = data_dict["neuralActivityTimeSeries"]

        go_cue_bins = data_dict["goPeriodOnsetTimeBin"].ravel().astype(int)

        prompts = [str(a[0]) for a in data_dict["characterCues"].ravel()]


    for idx, go_cue_bin in enumerate(go_cue_bins):

        start_bin = int(go_cue_bin) + REACTION_TIME_NUM_BINS

        end_bin = start_bin + TRAINING_WINDOW_NUM_BINS






    window = neural_activity[start_bin:end_bin].flatten()

    input_vectors.append(window)

    labels.append(prompts[idx])


    X = np.array(input_vectors)

    label_encoder = LabelEncoder()

    encoded_labels = label_encoder.fit_transform(labels)


    return train_test_split(X, encoded_labels, test_size=0.2, random_state=42)


# Load and preprocess data


def compute_accuracy(outputs, labels):

    _, predicted = torch.max(outputs, 1)

    correct = (predicted == labels).sum().item()

    return 100 * correct / len(labels)




X_train, X_test, y_train, y_test = load_and_preprocess_data()


# Neural network architecture

class EnhancedNeuralNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.5):

        super(EnhancedNeuralNetwork, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)

        self.bn1 = nn.BatchNorm1d(hidden_size)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)

        self.bn2 = nn.BatchNorm1d(hidden_size // 2)

        self.fc3 = nn.Linear(hidden_size // 2, num_classes)


    def forward(self, x):

        x = self.fc1(x)

        x = self.bn1(x)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        x = self.bn2(x)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc3(x)

        return x


# Model parameters

input_size = X_train.shape[1]

hidden_size = 128 # Adjust as needed

num_classes = len(np.unique(y_train))

dropout_rate = 0.5 # Adjust as needed

model = EnhancedNeuralNetwork(input_size, hidden_size, num_classes, dropout_rate)


#torchsummary.summary(model, input_size=(input_size,))




# Training parameters

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9) # Experiment with different optimizers and learning rates

criterion = nn.CrossEntropyLoss()

batch_size = 64 # Adjust as needed

writer = SummaryWriter('runs/handwriting_bci_experiment')




# DataLoader

train_dataset = NeuralActivityDataset(X_train, y_train)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = NeuralActivityDataset(X_test, y_test)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Training loop

num_epochs = 50 # Adjust as needed

for epoch in range(num_epochs):

    model.train()

    total_loss = 0

    total_accuracy = 0

    for i, (inputs, labels) in enumerate(train_loader):

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        total_accuracy += compute_accuracy(outputs, labels)


    avg_loss = total_loss / len(train_loader)

    avg_accuracy = total_accuracy / len(train_loader)

    writer.add_scalar('Training Loss', avg_loss, epoch)

    writer.add_scalar('Training Accuracy', avg_accuracy, epoch)

    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.2f}%")


# Evaluation

# Evaluation phase

model.eval()

total_test_loss = 0

total_test_accuracy = 0

with torch.no_grad():

    for inputs, labels in test_loader:

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        total_test_loss += loss.item()

        total_test_accuracy += compute_accuracy(outputs, labels)


avg_test_loss = total_test_loss / len(test_loader)

avg_test_accuracy = total_test_accuracy / len(test_loader)

writer.add_scalar('Test Loss', avg_test_loss, epoch)

writer.add_scalar('Test Accuracy', avg_test_accuracy, epoch)

print(f"Epoch {epoch + 1}/{num_epochs}, Test Loss: {avg_test_loss:.4f}, Accuracy: {avg_test_accuracy:.2f}%")


writer.close()
