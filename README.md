# Pepper Leaf Disease Detection

A convolutional neural network that looks at a photo of a bell pepper leaf and predicts whether it is **healthy** or has **bacterial spot**, with a small web app to try it.

**Live demo:** ADD-YOUR-STREAMLIT-LINK-HERE

![App screenshot](screenshot.png)

## What it does

- Trains a CNN from scratch on the bell pepper images of the PlantVillage dataset
- Classifies a leaf image into two classes: `Bacterial spot` or `Healthy`
- Serves the trained model through a Streamlit web app where anyone can upload an image

## Dataset

[PlantVillage on Kaggle](https://www.kaggle.com/datasets/arjuntejaswi/plant-village). Only the two bell pepper folders are used. The dataset is not stored in this repository; the notebook downloads it with the Kaggle API.

## Model

| Setting | Value |
|---|---|
| Input | 256 x 256 RGB, rescaled to 0-1 inside the model |
| Architecture | 6 x (Conv2D + MaxPooling), Flatten, Dense(64), Dense(2, softmax) |
| Parameters | about 184,000 |
| Optimizer / loss | Adam / sparse categorical cross-entropy |
| Epochs / batch size | 50 / 32 |
| Split | 80% train, 10% validation, 10% test |

## Results

Test accuracy: **99.6%** (238 of 239 held-out test images), from `model.evaluate(test_ds)` in the notebook.

## Files

| File | Purpose |
|---|---|
| `pepper_leaf_disease_detection.ipynb` | Training notebook (Google Colab) |
| `plant_disease_model.keras` | Trained model |
| `app.py` | Streamlit web app |
| `requirements.txt` | Python packages the app needs |

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Limitations

- Trained only on bell pepper leaves photographed on a plain background. Photos taken in a field, or of any other plant, will get an answer that is not reliable.
- Detects one disease only (bacterial spot).

## Built with

Python, TensorFlow / Keras, Streamlit, Google Colab
