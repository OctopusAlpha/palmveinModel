# Palm Vein Recognition using ResNet18 and Triplet Loss

This project implements a palm vein recognition system using a pre-trained ResNet18 model adapted for feature embedding and trained with Triplet Loss.

## Features

*   Uses ResNet18 for feature extraction.
*   Employs Triplet Loss for learning discriminative embeddings.
*   Includes data preprocessing with CLAHE enhancement.
*   Supports pre-processing images into tensors to speed up training.
*   Includes training and validation loops with accuracy calculation.
*   Saves the best model based on validation loss.
*   Plots training and validation loss/accuracy curves.
*   Includes early stopping mechanism.

## Project Structure

```
palmveinResNet18/
├── data/                     # Original dataset (example, replace with your data)
├── dataset_roi/              # ROI extracted dataset
│   ├── train/
│   └── valid/
├── dataset_preprocessed/     # Preprocessed tensor data (generated by preprocess_data.py)
│   ├── train/
│   └── valid/
├── saved_model/              # Saved trained models
├── results/                  # Saved training plots
├── .gitignore
├── CLAHE.py                  # CLAHE enhancement utility
├── README.md                 # This file
├── batch_roi_extraction.py   # Script for batch ROI extraction (if needed)
├── categorize.py             # Script to categorize data (if needed)
├── config.py                 # Configuration file for paths and hyperparameters
├── model.py                  # (Potentially contains model definition, check usage)
├── prepare_data.py           # Data loading and preparation (loads preprocessed tensors)
├── preprocess_data.py        # Script to preprocess images and save as tensors
├── requirements.txt          # Project dependencies
├── roi_extraction.py         # Script for single image ROI extraction (if needed)
├── split_dataset.py          # Script to split data into train/valid sets
├── test.py                   # Script for testing the trained model
├── train.py                  # Main training script
└── ... (other utility scripts)
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd palmveinResNet18
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Prepare your dataset:**
    *   Place your raw palm vein images into a directory structure suitable for `split_dataset.py` or organize them directly into `dataset_roi/train` and `dataset_roi/valid` folders, with subfolders for each class (person).
    *   Ensure `config.py` points to the correct `dataset_roi` directory.
    *   (Optional but recommended) Run `split_dataset.py` if you need to split your data automatically.

## Usage

1.  **Preprocess the data (Recommended for faster training):**
    *   Make sure the paths in `config.py` (`train_dir`, `valid_dir`, `preprocessed_train_dir`, `preprocessed_valid_dir`) are set correctly.
    *   Run the preprocessing script:
        ```bash
        python preprocess_data.py
        ```
    *   This will create the `dataset_preprocessed` directory containing `.pt` tensor files.

2.  **Train the model:**
    *   Adjust hyperparameters in `config.py` if needed (EPOCHS, BATCH_SIZE, etc.).
    *   Run the training script:
        ```bash
        python train.py
        ```
    *   The script will load preprocessed data (if generated in step 1) or process images on-the-fly (if `preprocess_data.py` was not run and `prepare_data.py` is configured accordingly).
    *   Training progress, loss, and accuracy will be displayed.
    *   The best model weights will be saved in the `saved_model` directory.
    *   Training plots will be saved in the `results` directory.

3.  **Test the model:**
    *   Implement or use the `test.py` script (modify as needed) to evaluate the trained model on a separate test set.

## Configuration

Key parameters can be adjusted in `config.py`:

*   `EPOCHS`: Number of training epochs.
*   `BATCH_SIZE`: Batch size for training and validation.
*   `save_model_dir`: Directory to save trained models.
*   `dataset_dir`: Base directory for ROI-extracted image data.
*   `train_dir`, `valid_dir`: Paths to training and validation image sets.
*   `preprocessed_dataset_dir`: Base directory for preprocessed tensor data.
*   `preprocessed_train_dir`, `preprocessed_valid_dir`: Paths to preprocessed training and validation tensor sets.

## Notes

*   This implementation assumes grayscale input images which are then converted to 3 channels for ResNet compatibility after CLAHE enhancement.
*   The `TripletDataset` in `prepare_data.py` now loads pre-saved `.pt` files if `preprocess_data.py` has been run.
*   Ensure the `enhance_image` function from `CLAHE.py` is correctly implemented and imported if used during preprocessing.