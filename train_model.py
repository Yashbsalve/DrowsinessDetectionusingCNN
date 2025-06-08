import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from cnn_model import DrowsinessCNN
import os

# ─── CONFIG ────────────────────────────────────────────────────────────────
DATA_DIR     = 'eye_dataset'
BATCH_SIZE   = 32
EPOCHS       = 10
LR           = 0.001
MODEL_PATH   = 'cnn_drowsiness.pth'
VALID_SPLIT  = 0.1  # 10% for validation

# ─── DEVICE ────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ─── TRANSFORMS ─────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

# ─── DATASET & DATALOADER ──────────────────────────────────────────────────
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
num_val = int(len(dataset) * VALID_SPLIT)
num_train = len(dataset) - num_val
train_ds, val_ds = random_split(dataset, [num_train, num_val])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ─── MODEL, LOSS, OPTIMIZER ────────────────────────────────────────────────
model = DrowsinessCNN().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ─── TRAINING LOOP ─────────────────────────────────────────────────────────
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total
    print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_acc:.2f}%")

# ─── SAVE MODEL ────────────────────────────────────────────────────────────
torch.save(model.state_dict(), MODEL_PATH)
print(f"[INFO] Model saved to {MODEL_PATH}")
