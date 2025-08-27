#v10: with dataset including distance
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import h5py
import torch
import torch.utils.data
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from scipy.io import loadmat, savemat
from tqdm import tqdm
import sys
import joblib

#                                   ### RANDOM SEED ###
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#print('torch version:',torch.__version__)
print('device:', device)

#                                                 ### MODEL PARAMETERS ###
activate_training = True
output_size = 4 # 4 paths 
input_size = 4 # 
batch_size = 256  #original 64 v1:256
# layer_sizes = [32, 64, 32, 16] #[32, 16, 16, 8]
# layer_sizes = [32, 64, 128, 64]  #v12
# layer_sizes = [64, 128, 256, 128] # v13 94.83% accuracy
#layer_sizes = [128, 256, 512, 128] #v14 best 95.49% accuracy BEST
#layer_sizes = [128, 512, 1024, 128] #v15 95.32 not good
# layer_sizes = [128, 512, 512, 128] #v16 95.47%
# layer_sizes = [128, 512, 256, 64] #v17 95.07%
layer_sizes = [128, 256, 512, 256] #then try 256 128 BEST 99.52 WO DISTANCE
early_stopping_patience = 50
# Training
epochs = 1000  # Train epochs
# learning_rate = 1e-4
learning_rate = 1e-3
checkpoint_path = "chkpt_classification_4class_v1.pth"  # Path to save/load checkpoints
best_checkpoint_path = "best_chkpt_classification_4class_v1.pth"
matfile_path = "losses_classification_4class_v1.mat"  # Path to save/load checkpoints


#                                           ### DATA PREPARATION ###
training_data1_path = '/main/training_dataset_with_distance_and_phase_4_path_80percent_k1_nopath_removed.mat'
with h5py.File(training_data1_path, 'r') as file:
    training_data1 = file['training_dataset'][:]
    train_data1 = training_data1.T
    print("Training data loaded successfully:", train_data1.shape)
val_data1_path = '/home/yildirbk/Desktop/4PATH_TR_TEST_DATASET_MODELS_MARCH25/Common_datasets/validation_dataset_with_distance_and_phase_4_path_10percent_k1_nopath_removed.mat'
with h5py.File(val_data1_path, 'r') as file:
    val_data1 = file['val_dataset'][:]
    val_data1 = val_data1.T
    print("Validation data loaded successfully:", val_data1.shape)
test_data1_path = '/home/yildirbk/Desktop/4PATH_TR_TEST_DATASET_MODELS_MARCH25/Common_datasets/test_dataset_with_distance_and_phase_4_path_10percent_k1_nopath_removed.mat'
with h5py.File(test_data1_path, 'r') as file:
    test_data1 = file['test_dataset'][:]
    test_data1 = test_data1.T
    print("Test data loaded successfully:", test_data1.shape)

def Custom_Dataset(dataset_type):
    #outputs=dataset_type[:,6:11].astype(np.float32)
    outputs=dataset_type[:,6].astype(np.float32).reshape(-1,1) #number of paths-classes
    labels = dataset_type[:,0:4].astype(np.float32) #with BSs locations and distance
    #labels = dataset_type[:,2:4].astype(np.float32)
    condition = dataset_type[:,5].astype(np.float32).reshape(-1,1)
    print('Outputs size:', outputs.shape)
    print('Labels size:', labels.shape)

    return outputs, labels
#FOR CLASS WEIGHTS

train_outputs_o, train_labels_o= Custom_Dataset(train_data1)
val_outputs_o, val_labels_o= Custom_Dataset(val_data1)
test_outputs_o, test_labels_o = Custom_Dataset(test_data1)

train_outputs_o_r, val_outputs_o_r, test_outputs_o_r = train_outputs_o-1, val_outputs_o-1, test_outputs_o-1  #to start classes from 0. (0 means 1 path)
#to add class weights
# train_out_tr=train_outputs_o_r.T
# class_weights = compute_class_weight('balanced', classes=np.unique(train_out_tr.flatten()), y=train_out_tr.flatten())
# class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)


s_sc_label_mlp = StandardScaler()
# s_sc_outputs = StandardScaler()
s_sc_label_mlp.fit(train_labels_o)
# joblib.dump(s_sc_label_mlp, "label_scaler_mlp.pkl")
# sys.exit()
# s_sc_outputs.fit(train_outputs_o)
train_labels_o, val_labels_o, test_labels_o = s_sc_label_mlp.transform(train_labels_o), s_sc_label_mlp.transform(val_labels_o), s_sc_label_mlp.transform(test_labels_o)
# train_outputs_o, val_outputs_o, test_outputs_o = s_sc_outputs.transform(train_outputs_o), s_sc_outputs.transform(val_outputs_o), s_sc_outputs.transform(test_outputs_o)

# denorm_train_out1, denorm_train_labels = s_sc_outputs.inverse_transform(train_outputs_o2), s_sc_label.inverse_transform(train_labels_o2)
# Extract scaler parameters (mean and scale)
# mean = torch.from_numpy(s_sc_outputs.mean_).to(device)
# scale = torch.from_numpy(s_sc_outputs.scale_).to(device)
# # Manual inverse transform
# denorm_train_out2 = train_outputs_o2 * scale + mean

# for i in range(len(train_outputs_o)):
#     print('train_outputs_o:',train_outputs_o[i])
#     print('train_outputs normalized:',train_outputs_o2[i])
#     print('train_outputs_denormalized:',denorm_train_out1[i])
#     print('train_outputs_denormalized_manual:',denorm_train_out2[i])

# print('max outputs; train, val,test:',np.max(train_outputs_o2),np.max(val_outputs_o),np.max(test_outputs_o))
# print('min outputs; train, val,test:',np.min(train_outputs_o2),np.min(val_outputs_o),np.min(test_outputs_o))

# for i in range(len(train_outputs_o)):
#     print('train_labels_o:',train_labels_o[i])
#     print('train_labels normalized:',train_labels_o2[i])
#     print('train_labels_denormalized:',denorm_train_labels[i])

train_dataset =np.concatenate((train_labels_o, train_outputs_o_r), axis=1)
val_dataset =np.concatenate((val_labels_o, val_outputs_o_r), axis=1)
test_dataset =np.concatenate((test_labels_o, test_outputs_o_r), axis=1)

train_dataset = torch.tensor(train_dataset,dtype=torch.float32)
val_dataset = torch.tensor(val_dataset,dtype=torch.float32)
test_dataset = torch.tensor(test_dataset,dtype=torch.float32)


train_data_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,      
    shuffle=True,
    drop_last=True,    # Keeps the last batch even if it's smaller than batch_size
    num_workers=32       
)

val_data_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True,    # Keeps the last batch even if it's smaller than batch_size
    num_workers=32   
)

test_data_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True,    # Keeps the last batch even if it's smaller than batch_size
    num_workers=32     
)

def one_hot(inputs, class_size):
    inputs = inputs.long()
    # print('labels input to one_hot:',inputs, inputs.shape)
    # print('class_size input to one_hot:',class_size)
    targets = torch.zeros(inputs.size(0), class_size, device=inputs.device)
    # print('targets initiated in one_hot:',targets.shape)
    for i, input in enumerate(inputs):
        targets[i, input] = 1
    # print('targets outputed in one_hot:',targets.shape)
    return targets.to(device)

# Define the model for 4 classes
class Classifier(nn.Module):
    def __init__(self, layer_sizes, input_size, output_size):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.layer_sizez = layer_sizes

        self.fc1 = nn.Linear(input_size, layer_sizes[0])
        self.bn1 = nn.BatchNorm1d(layer_sizes[0])
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.bn2 = nn.BatchNorm1d(layer_sizes[1])
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(layer_sizes[1], layer_sizes[2])
        self.bn3 = nn.BatchNorm1d(layer_sizes[2])
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(layer_sizes[2], layer_sizes[3])
        self.bn4 = nn.BatchNorm1d(layer_sizes[3])
        self.dropout4 = nn.Dropout(0.3)

        self.fc5 = nn.Linear(layer_sizes[3], output_size)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        
        x = self.relu(self.fc1(x))
        # x = self.dropout1(self.bn1(x)) #the best performing model does not have BatchNorm and Dropout
        x = self.relu(self.fc2(x))
        # x = self.dropout2(self.bn2(x))
        x = self.relu(self.fc3(x))
        # x = self.dropout3(self.bn3(x))
        x = self.relu(self.fc4(x))
        # x = self.dropout4(self.bn4(x))
        x = self.fc5(x)
        # x = self.softmax (self.fc3(x))
        return x
    
classifier = Classifier(layer_sizes, input_size, output_size).to(device)

# to count number of parameters (comment BatchNorm1d, and Dropout layer in the model architecture)
if not activate_training:
    try:
        tot_count=0
        tot_count_ind=0
        tmp = 1
        classifier = Classifier(layer_sizes, input_size, output_size).to(device)
        for parameter in classifier.parameters():
            
            count =parameter.numel()
            print(count)
            tot_count_ind += count
            print('------------------')
            tot_count += count
            print(f"Number of parameters: {tot_count}")
            if tmp%2==0:
                tot_count = tot_count - tot_count_ind
                tot_count_ind=0

            tmp += 1
            print('------------------')
            print('------------------')
    except Exception as e:
        print(f"Error: {e}")

# criterion = nn.CrossEntropyLoss(weight=class_weights)
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)


# Save model
def save_checkpoint(epoch, classifier, optimizer, file_path):
    checkpoint = {
        'epoch': epoch,
        'classifier_state_dict': classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved at epoch {epoch+1}.")


def load_checkpoint(file_path, classifier, optimizer):
    if not os.path.exists(file_path):
        print(f"No checkpoint found at '{file_path}'. Starting from scratch.")
        return 0  # Starting epoch

    checkpoint = torch.load(file_path, map_location=device, weights_only=True)
    
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    print(f"Checkpoint loaded. Resuming training from epoch {start_epoch}.")
    return start_epoch

# Save data to .mat file
def save_data(epoch, c_loss, val_acc, filename):
    # Check if the file exists
    if os.path.exists(filename):
        # Load existing data
        existing_data = loadmat(filename)
    else:
        # Create an empty dictionary if the file does not exist
        existing_data = {
            'epochs': [],
            'classifier_loss':[],
            'validation_accuracy':[]
        }

    # Append new data
    existing_data['epochs'] = np.append(existing_data['epochs'], epoch)
    existing_data['classifier_loss'] = np.append(existing_data['classifier_loss'], c_loss)
    existing_data['validation_accuracy'] = np.append(existing_data['validation_accuracy'], val_acc)

    # Save updated data
    savemat(filename, existing_data)

def evaluate_model(model, data_loader, criterion, output_size):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch[:, :input_size].to(device)
            labels = batch[:, input_size].to(device) #output
            #print(labels)

            # One-hot encode labels
            real_labels = one_hot(labels, output_size)
            #print(real_labels)

            # Forward pass
            predictions = model(inputs)
            #print(predictions)
            #sys.exit()

            loss = criterion(predictions, real_labels)
            total_loss += loss.item()

            # Store predictions and true labels for accuracy and report
            all_predictions.extend(torch.argmax(predictions, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    class_report = classification_report(all_labels, all_predictions, target_names=[f"Class {i}" for i in range(output_size)])
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    #model.eval()  # Set model to evaluation mode
    if data_loader==test_data_loader:
        # Assume all_labels and all_predictions are available
        target_names = [f"Class {i}" for i in range(output_size)]

        # Classification report (standard)
        class_report = classification_report(all_labels, all_predictions, target_names=target_names)

        # Raw confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_predictions)

        # Row-wise normalization
        row_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1, keepdims=True) * 100
        row_normalized = np.nan_to_num(row_normalized)  # avoid NaNs

        disp_row = ConfusionMatrixDisplay(confusion_matrix=row_normalized, display_labels=target_names)
        fig, ax = plt.subplots(figsize=(5, 5))
        disp_row.plot(cmap='plasma', ax=ax, colorbar=True, values_format=".2f")

        ax.images[0].set_clim(0, 100)
        ax.set_title("Row-Normalized Confusion Matrix")

        # Show raw counts as text (for context)
        #for (i, j), val in np.ndenumerate(conf_matrix):
        #    ax.text(j, i, str(val), ha='center', va='center', color='black', fontsize=9)

        plt.savefig("npaths_conf_matrix_row_normalized_plasma_v2.eps", format='eps', bbox_inches='tight')
        plt.close()

        # # Global normalization
        # global_normalized = conf_matrix.astype('float') / conf_matrix.sum() * 100

        # disp_global = ConfusionMatrixDisplay(confusion_matrix=global_normalized, display_labels=target_names)

        # fig, ax = plt.subplots(figsize=(5, 5))
        # disp_global.plot(cmap='viridis', ax=ax, colorbar=True)
        # ax.images[0].set_clim(0, 100)
        # ax.set_title("Globally-Normalized Confusion Matrix")

        # # Show raw counts again
        # for (i, j), val in np.ndenumerate(conf_matrix):
        #     ax.text(j, i, str(val), ha='center', va='center', color='black', fontsize=9)

        # plt.savefig("npaths_conf_matrix_global_normalized.eps", format='eps', bbox_inches='tight')
        # plt.close()

    return avg_loss, accuracy, class_report, conf_matrix



# #                          ###MAIN TRAINING LOOP###
if activate_training:
    print('Inside Training:')
    classifier.train()
    c_losses = []
    best_val_acc = 0
    epochs_no_improve = 0
    # Optionally, load from checkpoint
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(checkpoint_path, classifier, optimizer)

    for epoch in range(start_epoch, epochs):
        print(f'\nStarting epoch {epoch+1}/{epochs}')
        epoch_loss = 0.0
        batch_count = 0
        

        for i, batch in tqdm(enumerate(train_data_loader), total=len(train_data_loader), desc=f"Epoch {epoch+1}"):
            inputs = batch[:,:input_size]
            outputs = batch[:,input_size]
            current_batch_size = outputs.size(0)

            # Move data to device
            real_outputs = outputs.to(device)
            # print('real_outputs to one hot',real_outputs)
            real_outputs = one_hot(real_outputs, output_size)
            # print('real_outputs',real_outputs)
            inputs = inputs.to(device)
            
            # Train Classifier
            fake_outputs = classifier(inputs)
            # print('predicted_outputs',predicted_outputs)
            #real_outputs_inv = (((real_outputs+1)*d_range)/2) + data_min
            #fake_outputs_inv= (((fake_outputs+1)*d_range)/2) + data_min
            
            optimizer.zero_grad()
            #c_loss = criterion(fake_outputs_inv, real_outputs_inv)
            c_loss = criterion(fake_outputs, real_outputs)

            c_loss.backward()
            optimizer.step()
            batch_loss = c_loss.item()
            epoch_loss += batch_loss
            
            batch_count += 1
            
            if (i+1) % 20000 == 0 or (i+1) == len(train_data_loader):
                print(f'Batch {i+1}/{len(train_data_loader)} | C Loss: {c_loss:.4f}')
        
        # Average losses for the epoch
        avg_c_loss = epoch_loss / batch_count
        c_losses.append(avg_c_loss)
        
        print(f'Epoch [{epoch+1}/{epochs}] | Average C Loss: {avg_c_loss:.4f}')

        # Validation phase
        avg_val_loss, val_accuracy, val_report, val_conf_matrix = evaluate_model(
            classifier, val_data_loader, criterion, output_size
        )
        
        print(f"\nValidation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        # print("\nClassification Report:\n", val_report)
        # print("\nConfusion Matrix:\n", val_conf_matrix)

        # Save checkpoint at the end of each epoch
        save_checkpoint(epoch, classifier, optimizer,file_path=checkpoint_path)

        
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            save_checkpoint(epoch, classifier, optimizer, file_path=best_checkpoint_path)
            print("Best model saved.")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs due to no improvement in validation MSE for {early_stopping_patience} consecutive epochs.")
            break

        scheduler.step(avg_val_loss)

        # SAVE LOSSES AND METRICS
        c_losses_s = c_losses[-1]
        avg_val_acc_s = val_accuracy

        save_data(epoch, c_losses_s, avg_val_acc_s, matfile_path)


if not activate_training:

    #load_checkpoint(checkpoint_path, classifier, optimizer)
    load_checkpoint(best_checkpoint_path, classifier, optimizer) 
    
    avg_test_loss, test_accuracy, test_report, test_conf_matrix = evaluate_model(
            classifier, test_data_loader, criterion, output_size)

    print(f"\nTest Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nTest Classification Report:\n", test_report)
    print("\nTest Confusion Matrix:\n", test_conf_matrix)
    
