import os
import random
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from scipy.io import loadmat, savemat
from tqdm import tqdm
import matplotlib.pyplot as plt

### RANDOM SEED ###
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('device:', device)

### CONFIGURATION ###
# Must match the run_tag used in earlier scripts
RUN_TAG = 'O1_60_Lp4_BS18_rows1-2751_TX8x4_RX4x2_sc1'
INPUT_DIR = 'outputs'
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

### MODEL PARAMETERS ###
activate_training = True
output_size = 4            # NoP classes (1, 2, 3, 4 paths -> mapped to 0..3)
input_size  = 4            # BS (x,y), UE (x,y)
batch_size  = 256
layer_sizes = [128, 256, 512, 256]
early_stopping_patience = 50
epochs = 1000
learning_rate = 1e-3

checkpoint_path      = os.path.join(OUTPUT_DIR, "chkpt_nop_classifier.pth")
best_checkpoint_path = os.path.join(OUTPUT_DIR, "best_chkpt_nop_classifier.pth")
matfile_path         = os.path.join(OUTPUT_DIR, "losses_nop_classifier.mat")

### DATA PREPARATION ###
training_data_path = os.path.join(INPUT_DIR, f'training_dataset_{RUN_TAG}_no_path_removed.mat')
val_data_path      = os.path.join(INPUT_DIR, f'validation_dataset_{RUN_TAG}_no_path_removed.mat')
test_data_path     = os.path.join(INPUT_DIR, f'test_dataset_{RUN_TAG}_no_path_removed.mat')

# If you kept the original single-level MATLAB save format, change the three lines below to:
#   train_data = f['training_dataset'][:].T   (etc., 'val_dataset', 'test_dataset')
with h5py.File(training_data_path, 'r') as f:
    train_data = f['training_data'][:].T
    print("Training data loaded successfully:", train_data.shape)
with h5py.File(val_data_path, 'r') as f:
    val_data = f['val_data'][:].T
    print("Validation data loaded successfully:", val_data.shape)
with h5py.File(test_data_path, 'r') as f:
    test_data = f['test_data'][:].T
    print("Test data loaded successfully:", test_data.shape)

def split_dataset(dataset):
    """Extract input features (BS x,y, UE x,y) and NoP labels."""
    outputs = dataset[:, 6].astype(np.float32).reshape(-1, 1)   # number of paths
    labels  = dataset[:, 0:4].astype(np.float32)                # BS_x, BS_y, UE_x, UE_y
    return outputs, labels

train_outputs, train_labels = split_dataset(train_data)
val_outputs,   val_labels   = split_dataset(val_data)
test_outputs,  test_labels  = split_dataset(test_data)

# Shift NoP labels from {1, 2, 3, 4} to {0, 1, 2, 3}
train_outputs -= 1
val_outputs   -= 1
test_outputs  -= 1

scaler = StandardScaler()
scaler.fit(train_labels)
train_labels = scaler.transform(train_labels)
val_labels   = scaler.transform(val_labels)
test_labels  = scaler.transform(test_labels)

train_dataset = np.concatenate((train_labels, train_outputs), axis=1)
val_dataset   = np.concatenate((val_labels,   val_outputs),   axis=1)
test_dataset  = np.concatenate((test_labels,  test_outputs),  axis=1)

train_dataset = torch.tensor(train_dataset, dtype=torch.float32)
val_dataset   = torch.tensor(val_dataset,   dtype=torch.float32)
test_dataset  = torch.tensor(test_dataset,  dtype=torch.float32)

train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  drop_last=True, num_workers=32)
val_data_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, drop_last=True, num_workers=32)
test_data_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, drop_last=True, num_workers=32)


### MODEL ###
class Classifier(nn.Module):
    def __init__(self, layer_sizes, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, layer_sizes[0])
        self.fc2 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.fc3 = nn.Linear(layer_sizes[1], layer_sizes[2])
        self.fc4 = nn.Linear(layer_sizes[2], layer_sizes[3])
        self.fc5 = nn.Linear(layer_sizes[3], output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

classifier = Classifier(layer_sizes, input_size, output_size).to(device)
criterion  = nn.CrossEntropyLoss()
optimizer  = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
scheduler  = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)


### CHECKPOINT AND LOGGING UTILITIES ###
def save_checkpoint(epoch, classifier, optimizer, file_path):
    torch.save({
        'epoch': epoch,
        'classifier_state_dict': classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, file_path)
    print(f"Checkpoint saved at epoch {epoch+1}.")


def load_checkpoint(file_path, classifier, optimizer):
    if not os.path.exists(file_path):
        print(f"No checkpoint found at '{file_path}'. Starting from scratch.")
        return 0
    checkpoint = torch.load(file_path, map_location=device, weights_only=True)
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Checkpoint loaded. Resuming from epoch {start_epoch}.")
    return start_epoch


def save_data(epoch, c_loss, val_acc, filename):
    if os.path.exists(filename):
        existing = loadmat(filename)
    else:
        existing = {'epochs': [], 'classifier_loss': [], 'validation_accuracy': []}
    existing['epochs']              = np.append(existing['epochs'], epoch)
    existing['classifier_loss']     = np.append(existing['classifier_loss'], c_loss)
    existing['validation_accuracy'] = np.append(existing['validation_accuracy'], val_acc)
    savemat(filename, existing)


def evaluate_model(model, data_loader, criterion, output_size):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch[:, :input_size].to(device).float()
            labels = batch[:, input_size].to(device).long()
            preds = model(inputs)
            loss = criterion(preds, labels)
            total_loss += loss.item()
            all_predictions.extend(torch.argmax(preds, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss     = total_loss / len(data_loader)
    accuracy     = accuracy_score(all_labels, all_predictions)
    target_names = [f"Class {i}" for i in range(output_size)]
    class_report = classification_report(all_labels, all_predictions, target_names=target_names)
    conf_matrix  = confusion_matrix(all_labels, all_predictions)
    return avg_loss, accuracy, class_report, conf_matrix


### TRAINING ###
if activate_training:
    print('Training started.')
    classifier.train()
    c_losses = []
    best_val_acc = 0
    epochs_no_improve = 0
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(checkpoint_path, classifier, optimizer)

    for epoch in range(start_epoch, epochs):
        print(f'\nStarting epoch {epoch+1}/{epochs}')
        epoch_loss = 0.0
        batch_count = 0

        for i, batch in tqdm(enumerate(train_data_loader), total=len(train_data_loader), desc=f"Epoch {epoch+1}"):
            inputs       = batch[:, :input_size].to(device).float()
            real_outputs = batch[:, input_size].to(device).long()

            optimizer.zero_grad()
            fake_outputs = classifier(inputs)
            c_loss = criterion(fake_outputs, real_outputs)
            c_loss.backward()
            optimizer.step()

            epoch_loss += c_loss.item()
            batch_count += 1

            if (i+1) % 20000 == 0 or (i+1) == len(train_data_loader):
                print(f'Batch {i+1}/{len(train_data_loader)} | C Loss: {c_loss:.4f}')

        avg_c_loss = epoch_loss / batch_count
        c_losses.append(avg_c_loss)
        print(f'Epoch [{epoch+1}/{epochs}] | Average C Loss: {avg_c_loss:.4f}')

        avg_val_loss, val_accuracy, _, _ = evaluate_model(classifier, val_data_loader, criterion, output_size)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")

        save_checkpoint(epoch, classifier, optimizer, file_path=checkpoint_path)

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            save_checkpoint(epoch, classifier, optimizer, file_path=best_checkpoint_path)
            print("Best model saved.")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f"\nEarly stopping after {epoch+1} epochs.")
            break

        scheduler.step(avg_val_loss)
        save_data(epoch, c_losses[-1], val_accuracy, matfile_path)


### INFERENCE ###
else:
    load_checkpoint(best_checkpoint_path, classifier, optimizer)
    avg_test_loss, test_accuracy, test_report, test_conf_matrix = evaluate_model(
        classifier, test_data_loader, criterion, output_size
    )

    print(f"\nNoP Classifier Results on Test Set:")
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"\nClassification report:\n{test_report}")

    cm_normalized = test_conf_matrix.astype('float') / test_conf_matrix.sum(axis=1, keepdims=True) * 100
    cm_normalized = np.nan_to_num(cm_normalized)

    target_names = [f"Class {i}" for i in range(output_size)]
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=target_names)
    disp.plot(cmap='viridis', ax=ax, colorbar=True, include_values=False)
    ax.images[0].set_clim(0, 100)

    for (i, j), val in np.ndenumerate(cm_normalized):
        text_color = 'white' if val < 50 else 'black'
        ax.text(j, i, f'{val:.2f}%', ha='center', va='center', color=text_color, fontsize=10)

    plt.title(f"NoP Classifier Confusion Matrix (Acc: {test_accuracy*100:.2f}%)")
    plt.savefig(os.path.join(OUTPUT_DIR, "nop_confusion_matrix.eps"), format='eps', bbox_inches='tight')
    plt.show()
