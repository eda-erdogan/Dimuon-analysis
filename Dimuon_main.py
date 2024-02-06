import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt


# Load CSV file including 100k events
print("Loading dataset...")
df = pd.read_csv('/home/edaerdogan/Desktop/dimuonai/dimuon.csv')

signal_df = df[df['Q1'] * df['Q2'] < 0].copy() #df unaffected kalmalı, bu yüzden copy kullandım. Ana amacımız signal ve background ayrımı yapmak 
background_df = df[df['Q1'] * df['Q2'] >= 0].copy() 

signal_df['target'] = 0
background_df['target'] = 1

filtered_df = pd.concat([signal_df, background_df], ignore_index=True)

signal_count = len(df[df['Q1'] * df['Q2'] < 0])
background_count = len(df[df['Q1'] * df['Q2'] >= 0])

total_signal = signal_count
total_background = background_count
total_events = total_signal + total_background

# Calculate percentages
signal_percentage = total_signal / total_events * 100
background_percentage = total_background / total_events * 100

print(f'Total Signal Events: {total_signal}')
print(f'Total Background Events: {total_background}')
print(f'Signal Percentage: {signal_percentage}%')
print(f'Background Percentage: {background_percentage}%')

features = filtered_df[['pt1', 'pt2', 'eta1', 'eta2', 'phi1', 'phi2', 'Q1', 'Q2']]
target = filtered_df['target']

# Split data into train and test sets
train_data, test_data, train_target, test_target = train_test_split(features, target, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
train_data_tensor = torch.tensor(train_data.values, dtype=torch.float32)
test_data_tensor = torch.tensor(test_data.values, dtype=torch.float32)

train_target_tensor = torch.tensor(train_target.values, dtype=torch.float32)
test_target_tensor = torch.tensor(test_target.values, dtype=torch.float32) #sanırım filtrelediğim için buna ihtiyacım var o yüzden ekledim, tek data yetmeyebilir

# Custom dataset to read
class MuonDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

# DataLoader
print("Loading model...")
train_dataset = MuonDataset(train_data_tensor, train_target_tensor)
test_dataset = MuonDataset(test_data_tensor, test_target_tensor)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# DNN model
class MuonDNN(nn.Module):
    def __init__(self):
        super(MuonDNN, self).__init__() #8 nodes in input, 100-50-20 in hiddens, 1 for output

        self.hidden_layer1 = nn.Linear(8, 100) #input olmadan doğru mu anlayamadım, nöron sayısına göre elimine edersem pytorch doldurabilir gibi düşündüm 
        self.hidden_layer2 = nn.Linear(100, 50)
        self.hidden_layer3 = nn.Linear(50, 20)
        self.output_layer = nn.Linear(20, 1)

        self.activation = nn.ReLU() #input için de koymuş aslında fakat nöron sayısı uyuşmadığı için o kısmı atlamak zorunda kaldım
        self.output_activation = nn.Sigmoid()

        self.dropout = nn.Dropout(0.2)
        
        nn.init.normal_(self.hidden_layer1.weight)
        nn.init.normal_(self.hidden_layer2.weight)
        nn.init.normal_(self.hidden_layer3.weight)
        nn.init.normal_(self.output_layer.weight)

    def forward(self, x):
        x = self.activation(self.dropout(self.hidden_layer1(x)))
        x = self.activation(self.dropout(self.hidden_layer2(x)))
        x = self.activation(self.dropout(self.hidden_layer3(x)))
        x = self.output_activation(self.output_layer(x))
        return x

# Create an instance of the model
model = MuonDNN()

# Loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

best_test_loss = float('inf')
best_epoch = 0
patience = 5

params = {"batch_size": 128}  #makalede hyperparametreler özellikle verilmiyor, ekstra ne eklemek gerekli?

# Initialize counters for overall metrics
total_signal_predictions = []
total_signal_targets = []
total_background_predictions = []
total_background_targets = []

# Training 
print("Preparing Training")
batch_size = params["batch_size"]

num_epochs = 20
best_test_loss = float('inf')
best_epoch = 0
patience = 5

for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.view(-1, 1))
        loss.backward()
        optimizer.step()

    # Evaluate on the test set
    with torch.no_grad():
        test_losses = []
        predictions = []

        for inputs, targets in test_loader:
            test_outputs = model(inputs)
            test_loss = criterion(test_outputs, targets.view(-1, 1))
            test_losses.append(test_loss.item())
            predictions.extend(test_outputs.round().detach().numpy())

            # Accumulate predictions and targets for over
            # all metrics calculation
            if (targets == 0).sum() > 0:  # Background
                total_background_predictions.extend(test_outputs.round().detach().numpy())
                total_background_targets.extend(targets.numpy())
            else:  # Signal
                total_signal_predictions.extend(test_outputs.round().detach().numpy())
                total_signal_targets.extend(targets.numpy())

        average_test_loss = sum(test_losses) / len(test_losses)
        accuracy = accuracy_score(test_target_tensor, predictions)

      # Convert lists to PyTorch tensors
        total_signal_targets_tensor = torch.tensor(total_signal_targets, dtype=torch.float32)
        total_signal_predictions_tensor = torch.tensor(total_signal_predictions, dtype=torch.float32)

        total_background_targets_tensor = torch.tensor(total_background_targets, dtype=torch.float32)
        total_background_predictions_tensor = torch.tensor(total_background_predictions, dtype=torch.float32)

        # Ensure tensors are flattened to 1D
        total_signal_targets_tensor = total_signal_targets_tensor.flatten()
        total_signal_predictions_tensor = total_signal_predictions_tensor.flatten()

        total_background_targets_tensor = total_background_targets_tensor.flatten()
        total_background_predictions_tensor = total_background_predictions_tensor.flatten()

        # Draw histograms
        plt.figure(figsize=(12, 6))

        # Histogram for Predicted Signal
        plt.subplot(1, 2, 1)
        plt.hist(total_signal_predictions_tensor.numpy(), bins=50, color='blue', alpha=0.7, label='Predicted Signal')
        plt.title('Histogram of Predicted Signal')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.legend()

        # Histogram for Predicted Background
        plt.subplot(1, 2, 2)
        plt.hist(total_background_predictions_tensor.numpy(), bins=50, color='red', alpha=0.7, label='Predicted Background')
        plt.title('Histogram of Predicted Background')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.legend()

        plt.show()

        #Signal cm
        conf_matrix_signal = confusion_matrix(total_signal_targets_tensor, total_signal_predictions_tensor)
        tn_signal, fp_signal, fn_signal, tp_signal = conf_matrix_signal.ravel()
        precision_signal = precision_score(total_signal_targets_tensor, total_signal_predictions_tensor)
        sensitivity_signal = recall_score(total_signal_targets_tensor, total_signal_predictions_tensor)

        #Background cm
        conf_matrix_background = confusion_matrix(total_background_targets_tensor, total_background_predictions_tensor)
        tn_background, fp_background, fn_background, tp_background = conf_matrix_background.ravel()
        precision_background = precision_score(total_background_targets_tensor, total_background_predictions_tensor)
        sensitivity_background = recall_score(total_background_targets_tensor, total_background_predictions_tensor)
        #TP, FN, FP, TN; bu değerler makaledekine epey yakın çıkıyor 
        
        # Precision for the entire model
        predictions_np = np.concatenate(predictions)
        test_target_np = test_target_tensor.numpy()
        precision_model = precision_score(test_target_np, predictions_np)

        # Print metrics 
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Test Loss: {average_test_loss}, Accuracy: {accuracy}')
        print('Signal Metrics:')
        print(f'TP: {tp_signal}, FN: {fn_signal}, FP: {fp_signal}, TN: {tn_signal}')
        print(f'Precision (Signal): {precision_signal}, Sensitivity (Recall): {sensitivity_signal}')

        print('Background Metrics:')
        print(f'TP: {tp_background}, FN: {fn_background}, FP: {fp_background}, TN: {tn_background}')
        print(f'Precision (Background): {precision_background}, Sensitivity (Recall): {sensitivity_background}')

        print(f'Precision (Overall): {precision_model}')

        # Early stopping, 100k dataset için overfit olması yüksek diye düşünüyorum
        if average_test_loss < best_test_loss:
            best_test_loss = average_test_loss
            best_epoch = epoch
        elif epoch - best_epoch >= patience:
            print(f'Early stopping at epoch {epoch + 1}. Best test loss: {best_test_loss}')
            break

