import torch
import torch.optim as optim
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from src import config
from src.model.network import PalmVeinNet
from src.model.loss import TripletLoss, CenterLoss
from src.data.dataset import get_dataloaders

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

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.save_path)
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
    print(f"Initializing Model ({config.model})...")
    model = PalmVeinNet(embedding_size=config.feature_dim).to(device)
    
    # Losses
    triplet_loss_fn = TripletLoss(margin=0.3).to(device)
    # Center Loss needs to know number of classes
    # We can get it from the dataset
    num_classes = len(train_loader.dataset.classes)
    center_loss_fn = CenterLoss(num_classes=num_classes, feat_dim=config.feature_dim, device=device).to(device)
    
    # Optimizer
    # CenterLoss parameters need to be optimized too
    optimizer = optim.Adam([
        {'params': model.parameters()},
        {'params': center_loss_fn.parameters(), 'lr': 0.008}
    ], lr=0.0048)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Early Stopping
    save_path = os.path.join(config.save_model_dir, 'best_palm_vein_model.pth')
    early_stopping = EarlyStopping(patience=10, save_path=save_path)
    
    # Training Loop
    print("Start Training...")
    train_losses = []
    valid_losses = []
    
    for epoch in range(config.EPOCHS):
        model.train()
        total_train_loss = 0
        correct_pairs = 0
        total_pairs = 0
        
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.EPOCHS} [Train]', leave=True)
        
        for anchor, positive, negative, labels in loop:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            labels = labels.to(device)
            
            # Forward
            # We need features for CenterLoss, and embeddings for Triplet
            # But PalmVeinNet returns embeddings (L2 normalized).
            # CenterLoss usually works on features before normalization or after.
            # Let's use the output of model (embeddings).
            
            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)
            
            # Triplet Loss
            t_loss = triplet_loss_fn(anchor_out, positive_out, negative_out)
            
            # Center Loss (We use anchor's embedding and label)
            c_loss = center_loss_fn(anchor_out, labels)
            
            # Combined Loss
            loss = t_loss + 0.01 * c_loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            # Accuracy Metric (Triplet correctness)
            pos_dist = torch.norm(anchor_out - positive_out, p=2, dim=1)
            neg_dist = torch.norm(anchor_out - negative_out, p=2, dim=1)
            correct_pairs += (pos_dist < neg_dist).sum().item()
            total_pairs += anchor.size(0)
            
            loop.set_postfix(loss=loss.item(), acc=correct_pairs/total_pairs)
            
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        total_valid_loss = 0
        correct_valid = 0
        total_valid = 0
        
        with torch.no_grad():
            loop_val = tqdm(valid_loader, desc=f'Epoch {epoch+1}/{config.EPOCHS} [Valid]', leave=True)
            for anchor, positive, negative, labels in loop_val:
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                labels = labels.to(device)
                
                anchor_out = model(anchor)
                positive_out = model(positive)
                negative_out = model(negative)
                
                t_loss = triplet_loss_fn(anchor_out, positive_out, negative_out)
                c_loss = center_loss_fn(anchor_out, labels)
                loss = t_loss + 0.01 * c_loss
                
                total_valid_loss += loss.item()
                
                pos_dist = torch.norm(anchor_out - positive_out, p=2, dim=1)
                neg_dist = torch.norm(anchor_out - negative_out, p=2, dim=1)
                correct_valid += (pos_dist < neg_dist).sum().item()
                total_valid += anchor.size(0)
                
                loop_val.set_postfix(loss=loss.item(), acc=correct_valid/total_valid)
        
        avg_valid_loss = total_valid_loss / len(valid_loader)
        valid_losses.append(avg_valid_loss)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Valid Loss={avg_valid_loss:.4f}, Valid Acc={correct_valid/total_valid:.4f}")
        
        # Scheduler step
        scheduler.step(avg_valid_loss)
        
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
