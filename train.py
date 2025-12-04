import torch
import torch.optim as optim
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from src import config
from src.model.network import PalmVeinNet
from src.model.loss import TripletLoss, CenterLoss, OnlineTripletLoss
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
    # Use OnlineTripletLoss (Batch Hard)
    triplet_loss_fn = OnlineTripletLoss(margin=config.triplet_margin).to(device)
    
    # Center Loss needs to know number of classes
    # We can get it from the dataset
    num_classes = len(train_loader.dataset.classes)
    center_loss_fn = CenterLoss(num_classes=num_classes, feat_dim=config.feature_dim, device=device).to(device)
    
    # Optimizer
    # CenterLoss parameters need to be optimized too
    optimizer = optim.Adam([
        {'params': model.parameters()},
        {'params': center_loss_fn.parameters(), 'lr': config.lr_center}
    ], lr=config.lr)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.scheduler_factor, patience=config.scheduler_patience)
    
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
        correct_pairs = 0
        total_pairs = 0
        
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.EPOCHS} [Train]', leave=True)
        
        # Updated loop for standard dataset (images, labels)
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward
            embeddings = model(images)
            
            # Online Triplet Loss (Batch Hard)
            t_loss = triplet_loss_fn(embeddings, labels)
            
            # Center Loss
            c_loss = center_loss_fn(embeddings, labels)
            
            # Combined Loss
            loss = t_loss + config.center_loss_weight * c_loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            # Accuracy Metric (Hard Triplet correctness)
            # Re-calculate distances for metric (could be optimized but fine for now)
            with torch.no_grad():
                dot_product = torch.matmul(embeddings, embeddings.t())
                square_norm = torch.diag(dot_product)
                distances = square_norm.unsqueeze(1) - 2 * dot_product + square_norm.unsqueeze(0)
                distances = torch.clamp(distances, min=0.0)
                distances = torch.sqrt(distances + 1e-12)

                mask_pos = triplet_loss_fn._get_anchor_positive_triplet_mask(labels).float()
                mask_neg = triplet_loss_fn._get_anchor_negative_triplet_mask(labels).float()

                # Max Pos Dist
                pos_dists = distances * mask_pos
                hard_pos_dist = pos_dists.max(1)[0]
                
                # Min Neg Dist
                max_d = distances.max()
                neg_dists = distances + max_d * (1.0 - mask_neg)
                hard_neg_dist = neg_dists.min(1)[0]
                
                # Correct if hard_pos < hard_neg
                correct_pairs += (hard_pos_dist < hard_neg_dist).sum().item()
                total_pairs += images.size(0)
            
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
            for images, labels in loop_val:
                images = images.to(device)
                labels = labels.to(device)
                
                embeddings = model(images)
                
                t_loss = triplet_loss_fn(embeddings, labels)
                c_loss = center_loss_fn(embeddings, labels)
                loss = t_loss + config.center_loss_weight * c_loss
                
                total_valid_loss += loss.item()
                
                # Metric
                dot_product = torch.matmul(embeddings, embeddings.t())
                square_norm = torch.diag(dot_product)
                distances = square_norm.unsqueeze(1) - 2 * dot_product + square_norm.unsqueeze(0)
                distances = torch.clamp(distances, min=0.0)
                distances = torch.sqrt(distances + 1e-12)

                mask_pos = triplet_loss_fn._get_anchor_positive_triplet_mask(labels).float()
                mask_neg = triplet_loss_fn._get_anchor_negative_triplet_mask(labels).float()

                pos_dists = distances * mask_pos
                # Handle case where validation batch might not have positives for some samples (if batch size is small and random)
                # But validation loader is not balanced sampler, it's standard sequential.
                # If a batch has only 1 sample of a class, max(pos_dists) will be 0 (masked).
                # This is a limitation of batch-based metric on validation if not carefully batched.
                # However, for validation loss, it handles it gracefully (loss=0 if no pos/neg).
                # For metric, let's just compute it where possible.
                
                # Actually, for validation, we might want standard pair accuracy or just loss.
                # But let's stick to the same metric for consistency, acknowledging it might be noisy if batch size is small.
                # Config BATCH_SIZE is 16, which is small. Validation might have single samples.
                
                hard_pos_dist = pos_dists.max(1)[0]
                
                max_d = distances.max()
                neg_dists = distances + max_d * (1.0 - mask_neg)
                hard_neg_dist = neg_dists.min(1)[0]
                
                # Only count valid triplets (where we have at least one positive and one negative)
                has_pos = mask_pos.sum(1) > 0
                has_neg = mask_neg.sum(1) > 0
                valid_mask = has_pos & has_neg
                
                if valid_mask.sum() > 0:
                    correct = (hard_pos_dist < hard_neg_dist)[valid_mask].sum().item()
                    correct_valid += correct
                    total_valid += valid_mask.sum().item()
                
                loop_val.set_postfix(loss=loss.item(), acc=correct_valid/total_valid if total_valid > 0 else 0)
        
        avg_valid_loss = total_valid_loss / len(valid_loader)
        valid_losses.append(avg_valid_loss)
        
        valid_acc = correct_valid/total_valid if total_valid > 0 else 0
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Valid Loss={avg_valid_loss:.4f}, Valid Acc={valid_acc:.4f}")
        
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
