# Palm Vein Recognition System

A robust palm vein recognition system based on ResNet18, utilizing a combination of **Triplet Loss** and **Center Loss** for learning discriminative embeddings. This project features a modular architecture, dynamic data augmentation, and an optimized training pipeline.

## Key Features

*   **Advanced Model Architecture**: ResNet18 backbone with a custom MLP Projection Head for better embedding separation.
*   **Hybrid Loss Function**: Combines **Triplet Loss** (inter-class separation) and **Center Loss** (intra-class compactness) to maximize recognition accuracy.
*   **Dynamic Data Pipeline**:
    *   **On-the-fly Augmentation**: Real-time random rotation, cropping, flipping, and color jittering during training.
    *   **Integrated Enhancement**: Automatic CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessing.
    *   **Efficient Loading**: No need for offline preprocessing or intermediate `.pt` files.
*   **Modular Design**: Clean separation of core logic, data handling, and model definitions following standard engineering practices.
*   **Training Utilities**: Early stopping, learning rate scheduling, and comprehensive logging.

## Project Structure

```
palmveinResnet/
├── src/                    # Core source code
│   ├── config.py           # Global configuration
│   ├── core/               # Core algorithms
│   │   ├── enhance.py      # Image enhancement (CLAHE)
│   │   └── roi.py          # ROI extraction logic
│   ├── data/               # Data handling
│   │   ├── dataset.py      # PyTorch Dataset & DataLoader
│   │   ├── split.py        # Dataset splitting utility
│   │   └── categorize.py   # Data organization utility
│   └── model/              # Model definitions
│       ├── network.py      # PalmVeinNet architecture
│       └── loss.py         # TripletLoss & CenterLoss
├── tools/                  # Auxiliary tools
│   ├── batch_roi.py        # Batch ROI extraction
│   ├── check_device.py     # Hardware check
│   └── ...
├── train.py                # Main training script
├── test.py                 # Testing & Inference script
├── requirements.txt        # Dependencies
└── saved_model/            # Output directory for models
```

## Setup & Installation

1.  **Clone the repository**
    ```bash
    git clone <repository-url>
    cd palmveinResnet
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Ensure you have PyTorch installed with CUDA support if you have a GPU.*

3.  **Prepare Dataset**
    *   Organize your ROI-extracted images into train and validation directories:
        ```
        dataset_roi/
        ├── train/
        │   ├── person_001/
        │   │   ├── 01.bmp
        │   │   └── ...
        │   └── ...
        └── valid/
            ├── person_001/
            └── ...
        ```
    *   If you have raw images, use `src/data/split.py` or `src/core/roi.py` (via tools) to prepare them.

## Usage

### Training

Run the training script directly. It will automatically load the configuration from `src/config.py`.

```bash
python train.py
```

*   **Outputs**:
    *   Best model weights: `saved_model/best_palm_vein_model.pth`
    *   Loss curves: `saved_model/loss_curve.png`

### Testing

To evaluate the model or compare two images:

```bash
python test.py
```

*   Modify `test.py` to point to specific image pairs you want to verify.

### Configuration

All hyperparameters are centralize in `src/config.py`. Common adjustments include:

*   `EPOCHS`: Total training epochs (default: 10)
*   `BATCH_SIZE`: Training batch size (default: 16)
*   `feature_dim`: Dimension of the output embedding (default: 512)
*   `dataset_dir`: Path to your dataset

## Technical Details

1.  **Input**: Grayscale palm vein images are converted to 3-channel (RGB) after CLAHE enhancement.
2.  **Backbone**: Pre-trained ResNet18 (ImageNet weights).
3.  **Head**: Replaces the original FC layer with a `Linear -> BN -> ReLU -> Linear -> BN` projection head.
4.  **Inference**: The model outputs L2-normalized embeddings. Similarity is calculated using Cosine Similarity.
