# some training parameters
#7200
EPOCHS = 50
BATCH_SIZE = 128
feature_dim = 512
image_height = 224
image_width = 224
channels = 3
save_model_dir = "saved_model"
dataset_dir = "dataset_roi/"
train_dir = dataset_dir + "train"
valid_dir = dataset_dir + "valid"
preprocessed_dataset_dir = "dataset_preprocessed/"
preprocessed_train_dir = preprocessed_dataset_dir + "train"
preprocessed_valid_dir = preprocessed_dataset_dir + "valid"

# Hyperparameters
lr = 0.0048
lr_center = 0.008
triplet_margin = 0.3
center_loss_weight = 0.01
scheduler_factor = 0.5
scheduler_patience = 3
early_stopping_patience = 10

# choose a network
model = "resnet18"
# model = "resnet34"
# model = "resnet50"
# model = "resnet101"
# model = "resnet152"
