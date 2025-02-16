
# some training parameters
#7200
EPOCHS = 30
BATCH_SIZE = 32
NUM_CLASSES = 604
image_height = 224
image_width = 224
channels = 3
save_model_dir = "saved_model"
dataset_dir = "../dataset/"
train_dir = dataset_dir + "train"
valid_dir = dataset_dir + "valid"
test_dir = dataset_dir + "test"

# choose a network
model = "resnet18"
# model = "resnet34"
#model = "resnet50"
# model = "resnet101"
# model = "resnet152"