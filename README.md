# YOLO Training with CUDA

This repository is designed to train YOLO models on a server or PC equipped with an Nvidia GPU using CUDA acceleration.

## Prerequisites

- A machine with an Nvidia GPU and CUDA installed
- Python environment with required dependencies (see your environment setup)
- A properly formatted dataset with a `.yaml` configuration file

## Setup Instructions

1. Clone this repository to your local machine or server.
2. Open the `YOLO_Training.py` file.
3. Update the dataset path:
   - Go to line 7.
   - Replace the existing path with the path to your dataset's `.yaml` file.

## Output

- After training is complete, the trained model will be saved as `best.torchscript`.
- You can find it in the following directory:  
  `YOLO_Training/train/weights`

This `best.torchscript` file is your final trained model and can be used for inference.
