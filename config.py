# some training parameters
#7200
EPOCHS = 10
BATCH_SIZE = 16
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

# choose a network
model = "resnet18"
# model = "resnet34"
# model = "resnet50"
# model = "resnet101"
# model = "resnet152"