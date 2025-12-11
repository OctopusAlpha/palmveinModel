import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from src import config
from src.model.network import PalmVeinNet
from src.model.loss import TripletLoss, CenterLoss, OnlineTripletLoss, LabelSmoothingCrossEntropy, FocalLoss
from src.model.metric_learning import ArcFace
from src.data.dataset import get_dataloaders
from torch.optim.lr_scheduler import StepLR


# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, save_path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.save_path = save_path

    def __call__(self, val_loss, model, arcface_loss=None):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, arcface_loss)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, arcface_loss)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, arcface_loss=None):
        save_dict = {'model_state_dict': model.state_dict()}
        if arcface_loss is not None:
             save_dict['arcface_state_dict'] = arcface_loss.state_dict()
        
        torch.save(save_dict, self.save_path)
        print(f'Validation loss decreased. Saving model to {self.save_path}')

def train():
    # Directories
    os.makedirs(config.save_model_dir, exist_ok=True)
    
    # Data Loaders
    print("Initializing Datasets...")
    try:
        train_loader, valid_loader = get_dataloaders()
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    # Model
    print(f"Initializing Model ({config.model}) with Softmax Classifier...")
    # Using ResNet50 by default as updated in network.py
    # Pass num_classes to enable classification head
    num_classes = len(train_loader.dataset.classes)
    print(f"Number of classes: {num_classes}")
    
    model = PalmVeinNet(embedding_size=config.feature_dim, num_classes=num_classes).to(device)
    
    # Losses
    # Use CrossEntropyLoss or FocalLoss for classification
    print("Initializing Classification Loss (Focal Loss)...")
    
    # Use Focal Loss as in reference project (or CrossEntropyLoss)
    criterion = FocalLoss(gamma=2)
    # criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    # Optimize model parameters
    # Lower initial LR to prevent NaN
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=5e-4)
    
    # Scheduler: StepLR as in reference project
    # This helps to jump out of local minima
    scheduler = StepLR(optimizer, step_size=config.lr_step, gamma=0.1)
    
    # Early Stopping
    save_path = os.path.join(config.save_model_dir, 'best_palm_vein_model.pth')
    early_stopping = EarlyStopping(patience=config.early_stopping_patience, save_path=save_path)
    
    # Training Loop
    print("Start Training...")
    train_losses = []
    valid_losses = []
    
    for epoch in range(config.EPOCHS):
        model.train()
        total_train_loss = 0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.EPOCHS} [Train]', leave=True)
        
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward
            # Model now returns logits directly
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient Clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            total_train_loss += loss.item()
            
            # Accuracy
            # Debug: Print sample outputs and labels once
            if epoch == 0 and total == 0:
                 print(f"Sample outputs range: {outputs.min().item():.2f} to {outputs.max().item():.2f}")
                 print(f"Sample labels: {labels[:5]}")
                 _, pred_debug = torch.max(outputs.data, 1)
                 print(f"Sample preds: {pred_debug[:5]}")

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=100.*correct/total)
            
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        total_valid_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            loop_val = tqdm(valid_loader, desc=f'Epoch {epoch+1}/{config.EPOCHS} [Valid]', leave=True)
            for images, labels in loop_val:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                total_valid_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                loop_val.set_postfix(loss=loss.item(), acc=100.*val_correct/val_total if val_total > 0 else 0)
        
        avg_valid_loss = total_valid_loss / len(valid_loader)
        valid_losses.append(avg_valid_loss)
        
        valid_acc = 100. * val_correct / val_total if val_total > 0 else 0
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%, Valid Loss={avg_valid_loss:.4f}, Valid Acc={valid_acc:.2f}%")
        with open(os.path.join(config.save_model_dir, 'training.log'), 'a') as fs:
            fs.write(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%, Valid Loss={avg_valid_loss:.4f}, Valid Acc={valid_acc:.2f}%\n")
        
        # Scheduler step
        scheduler.step()
        
        # Early Stopping
        early_stopping(avg_valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
            
    # Plotting
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(config.save_model_dir, 'loss_curve.png'))
    print("Training finished.")

if __name__ == '__main__':
    train()
