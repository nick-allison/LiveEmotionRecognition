README.md
=========

Table of Contents
-----------------
1. [Project overview](#project-overview)  
2. [Quick-start demo (pre-trained weights)](#quick-start-demo-pre-trained-weights)  
3. [Installation](#installation)  
4. [Dataset acquisition & preparation](#dataset-acquisition--preparation)  
5. [Data cleaning utilities](#data-cleaning-utilities)  
6. [Optional data augmentation](#optional-data-augmentation)  
7. [Model training](#model-training)  
8. [Running the webcam application](#running-the-webcam-application)  
9. [Deprecated ViT notebook](#deprecated-vit-notebook)  
10. [Credits](#credits)  
11. [License](#license)  

---

Project overview
----------------
Real-time face-emotion classifier built on **YOLOv8-cls**.  
* Detects webcam faces at ~5 FPS, predicts eight emotions:  
  `angry, disgust, fear, happy, neutral, sad, ahegao, surprise`.  
* Ships **ready-to-run** thanks to the `weights.pt` checkpoint included in the root directory.  
* Code for training your own model available in: `train.ipynb`.
* Tools for cleaning the kaggle data are in the data_tools dir.


Quick-start demo (pre-trained weights)
--------------------------------------
1. **Create env & install deps** (see *Installation*).  
2. Make sure a webcam is attached.  
3. Run  
   ```bash
   python main.py
   ```  
   *Press **q** in either OpenCV window to exit.*

> The demo uses **M-series Mac GPU** by default (`device='mps'`).  
> - On Windows/Linux NVIDIA GPUs, comment the MPS line and uncomment the CUDA line inside `main.py`.  
> - CPU fallback works but will be slower.

Installation
------------
*Tested on Python ≥ 3.10.*

```bash
# clone your copy (or download ZIP)
git clone <this-repo-url>
cd <repo-root>

# always recommended:
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# install packages
pip install --upgrade pip
pip install -r requirements.txt
```

Dataset acquisition & preparation
---------------------------------
Training requires images split into `all_data/train` and `all_data/val`  
(8 sub-folders, one per class).

| Dataset                           | Kaggle link                                                                                           |
|-----------------------------------|-------------------------------------------------------------------------------------------------------|
| Emotion Recognition Dataset       | https://www.kaggle.com/datasets/sujaykapadnis/emotion-recognition-dataset                            |
| FER-2013 “ferdata”                | https://www.kaggle.com/datasets/chiragsoni/ferdata                                                    |

1. **Download both ZIPs** → extract somewhere.  
2. Run the resize script to harmonise resolution & colour:
   ```bash
   python data_tools/resize1.py \
       --input_dir "<unzipped-emotion-dataset-path>" \
       --output_dir "all_data_48x48_bw"
   ```
3. Manually **merge** FER-2013 images into the matching label sub-folders.  
4. Final structure **must** be:

```
all_data/
└── train/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    ├── ahegao/
    └── surprise/
└── val/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    ├── ahegao/
    └── surprise/
```

You choose the split strategy – e.g. copy 20 % of each class into `val/`.  The sort.ipynb file has some code to help with this- it will put the first x images into train and the rest into val.

Data cleaning utilities
-----------------------
Many original images are blank banners (“image moved”, extreme black/white).  
Use **`intensity.ipynb`** to:
* Plot per-class intensity histograms.  
* Display lowest / highest intensity examples.  
* Delete files below / above chosen thresholds via `delete_extreme_intensity_images()`.

> Running these steps before augmentation greatly improves final accuracy.

Optional data augmentation
--------------------------
Open **`data_tools/aug.ipynb`** and tweak:
* flips, rotations, translations  
* brightness / contrast shifts  

The notebook already contains balanced augmentation recipes that bring each class to ~13 k images.  
Feel free to skip or adapt; YOLO will happily train on un-augmented data.

Model training
--------------
Launch **`train.ipynb`** (Jupyter, VSCode, or Colab):

1. `BASE_DIR`, `TRAIN_DIR`, `VAL_DIR` already point to `all_data/…`.  
2. Select device: `"mps" | "cuda" | "cpu"`.  
3. Run all cells.
   * 20 epochs on M-series Mac ≈ 20 min, NVIDIA RTX 3060 ≈ 10 min, CPU ≫ 1 h.  
4. Outputs land in `runs/classify/emotion_clf/*`:
   * checkpoints per epoch (`epoch1.pt`, `epoch2.pt`, …)  
   * `results.csv` (loss / accuracy curves)  
   * TensorBoard logs  

To resume training:
```python
model = YOLO('runs/classify/emotion_clf/weights/last.pt')
model.train(resume=True, epochs=…)
```

Running the webcam application
------------------------------
```bash
python main.py
```
* Window 1: **Live Emotion Recognition** (HD frame + overlay).  
* Window 2: **Model Input (low-res)** – 64 × 64 patch that the network sees.

Keyboard shortcuts  
| Key | Action |
|-----|--------|
| `q` | quit both windows |

If you have multiple cameras, change the index in  
`cap = cv2.VideoCapture(0)`.


Deprecated ViT notebook
-----------------------
`deprecated/vit.ipynb` fine-tunes a Vision-Transformer backbone on the same  
48 × 48 data. It strongly over-fits and under-performs YOLOv8-cls. Kept only as an  
experiment; **not used in the final demo**.

Credits
-------
* **Ultralytics YOLOv8** – https://github.com/ultralytics/ultralytics  
* **ViT (timm)** – https://github.com/rwightman/pytorch-image-models  
* Original datasets courtesy of their Kaggle authors (see links above).

License
-------
This project is released under the MIT License (see [LICENSE](LICENSE) file).