# DrowsinessDetectionusingCNN


---
use python 3.9+
pip install flask opencv-python torch torchvision numpy


# Dataset Preparation

1. Create a folder named `eye_dataset` in your root directory.
2. Inside it, create two folders: `open/` and `close/`.
3. Fill them with eye images:
   - `open/` → images of open eyes
   - `close/` → images of closed eyes

This dataset will be used to train the CNN model.

---

##  Train the Model

Run the model training script:

```bash
python cnn_model.py


run
python app.py

After starting, you’ll get a local server link like:
http://127.0.0.1:5000/

