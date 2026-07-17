# Satellite Image Classification

A CNN that classifies satellite images into 10 land-cover types, things like forest, crops, highways, and rivers, trained on the EuroSAT dataset.

## What it does

Upload a satellite image and the model tells you what kind of land it's looking at. Built with TensorFlow/Keras, served through a simple Streamlit app.

**Classes**: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake

## Dataset

[EuroSAT](https://github.com/phelber/EuroSAT), ~27,000 RGB images, 64x64 pixels, sourced from Sentinel-2 satellite imagery. Split 80/10/10 into train/validation/test.

## Model

A CNN with 4 convolutional blocks (BatchNorm + MaxPooling), GlobalAveragePooling instead of Flatten, and dropout to control overfitting. Trained with Adam, ReduceLROnPlateau, and early stopping.

**Test accuracy: 95.9%**

The architecture went through a few rounds of tuning, dropping the image size from 128x128 to 64x64 for speed, then adjusting pooling depth and adding capacity/regularization to recover the accuracy that move initially cost. Full reasoning is in the notebook.

## Files

- cnn_satellite_image_classification.ipynb   # data prep, training, evaluation
- app.py                                      # Streamlit inference app
- saved_models/image_model_v2.keras           # trained model

## Running the app

pip install streamlit tensorflow pillow pandas
streamlit run app.py

Then upload an image in the browser.

## Known limitations

The model performs well on EuroSAT-style imagery but doesn't generalize to satellite/aerial photos from other sources, different sensors, resolutions, and color processing make real-world images look statistically different from what the model was trained on. See the notebook's limitations section for details.

## Improvements to try next

- Test against a different remote-sensing dataset (e.g. NWPU-RESISC45) for a real cross-domain accuracy check
- Train on multiple data sources instead of just EuroSAT
- Try transfer learning from a model pretrained on broader aerial imagery