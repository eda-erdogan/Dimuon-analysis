#Dimuon analysis for 100k events 
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

# Load CSV file including 100k events
print("Loading dataset...")
df = pd.read_csv('/home/edaerdogan/Desktop/dimuonai/dimuon.csv')

signal_df = df[df['Q1'] * df['Q2'] < 0].copy() #df unaffected kalmalı, bu yüzden copy kullandım. Ana amacımız signal ve background ayrımı yapmak 
background_df = df[df['Q1'] * df['Q2'] >= 0].copy() 

signal_df['target'] = 0
background_df['target'] = 1

filtered_df = pd.concat([signal_df, background_df], ignore_index=True) #pd.concat is a function used to concatenate DataFrames along a particular axis.

signal_count = len(df[df['Q1'] * df['Q2'] < 0])
background_count = len(df[df['Q1'] * df['Q2'] >= 0])


features = filtered_df[['pt1', 'pt2', 'eta1', 'eta2', 'phi1', 'phi2', 'Q1', 'Q2']] #Q ve M hangisi 4. parameter? M indexli değil ama Q da zaten ilk filtrem
target = filtered_df['target']

# Split data into train and test sets
train_data, test_data, train_target, test_target = train_test_split(features, target, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
train_data_tensor = torch.tensor(train_data.values, dtype=torch.float32)
test_data_tensor = torch.tensor(test_data.values, dtype=torch.float32)

train_target_tensor = torch.tensor(train_target.values, dtype=torch.float32) #64 diyemez miyim? ama sanırım 32 daha hızlı run ediyor
test_target_tensor = torch.tensor(test_target.values, dtype=torch.float32) #sanırım filtrelediğim için buna ihtiyacım var o yüzden ekledim, tek data yetmeyebilir

#custom dataset to read
class MuonDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

#DataLoader
print("Loading model...")
train_dataset = MuonDataset(train_data_tensor, train_target_tensor)
test_dataset = MuonDataset(test_data_tensor, test_target_tensor)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

#DNN model
class MuonDNN(nn.Module):
    def __init__(self):
        super(MuonDNN, self).__init__() #8 nodes in input, 100-50-20 in hiddens, 1 for output
        
        self.hidden_layer1 = nn.Linear(8, 100) #input olmadan doğru mu anlayamadım, nöron sayısına göre elimine edersem pytorch doldurabilir gibi düşündüm 
        self.hidden_layer2 = nn.Linear(100, 50)
        self.hidden_layer3 = nn.Linear(50, 20)
        self.output_layer = nn.Linear(20, 1)

        self.activation = nn.ReLU() #input için de koymuş aslında
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

#loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


best_test_loss = float('inf')
best_epoch = 0
patience = 5

params = {"batch_size": 128}  #makalede hyperparametreler özellikle verilmiyor, ekstra ne eklemek gerekli bilemedim

# Training 
print("Preparing Training")
batch_size=params["batch_size"]

num_epochs = 20
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.view(-1, 1))
        loss.backward()
        optimizer.step() #update için

    # Evaluate on the test set
    with torch.no_grad():
        test_losses = []
        predictions = []

        for inputs, targets in test_loader:
            test_outputs = model(inputs)
            test_loss = criterion(test_outputs, targets.view(-1, 1))
            test_losses.append(test_loss.item())
            predictions.extend(test_outputs.round().detach().numpy())
            
         #  binary_predictions = (test_outputs >= 0.5).float() binary pred. için deneyebilirim ama zorunlu mu emin değilim 
         #  predictions.extend(binary_predictions.cpu().numpy())
    
        average_test_loss = sum(test_losses) / len(test_losses)
        accuracy = accuracy_score(test_target_tensor, predictions)    

    #TP, FN, FP, TN; bu değerler makaledekine epey yakın çıkıyor 
        tn, fp, fn, tp = confusion_matrix(test_target_tensor, predictions).ravel()

        #Sensitivity 
        sensitivity = recall_score(test_target_tensor, predictions)

        #Precision
        precision = precision_score(test_target_tensor, predictions)

        #F1 score
        f1 = f1_score(test_target_tensor, predictions)

        #ROC AUC curve
        roc_auc = roc_auc_score(test_target_tensor, predictions) #sk metrikleri tp vs'den manuel çıkarılmış ama aynı sonuç çıkmalı benimkiyle. 

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Test Loss: {average_test_loss}, Accuracy: {accuracy}')
    print(f'TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}')
    print(f'Sensitivity (Recall): {sensitivity}, Precision: {precision}, F1 Score: {f1}, ROC AUC: {roc_auc}')

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Test Loss: {average_test_loss}, Accuracy: {accuracy}')

   # Early stopping, 100k dataset için overfit olmasın
    if average_test_loss < best_test_loss:
        best_test_loss = average_test_loss
        best_epoch = epoch
    elif epoch - best_epoch >= patience:
        print(f'Early stopping at epoch {epoch + 1}. Best test loss: {best_test_loss}')
        break
    #inv mass histogramı için seçtiğim filtrenin içinden signal ve bck. için iki ayrı raw, predicted ve test data lazım
# early stopping kullansam loss değişiminden kurtulur muyum
    #en son lazzy predict dene
