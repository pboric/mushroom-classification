# Mushroom Classification

Classifying photos of mushrooms into nine common genera (Agaricus, Amanita, Boletus, Cortinarius, Entoloma,
Hygrocybe, Lactarius, Russula, Suillus) with transfer learning on a ResNet34, built with PyTorch Lightning.

## Results

Test set of 1,343 images:

| Version | Changes | Accuracy | Suillus F1 (hardest class) |
|---|---|---:|---:|
| v1 | ResNet34 fine-tuned with augmentation and early stopping | 88.0 % | 0.73 |
| v2 | Extra intermediate layers, three optimisers compared (AdamW, SGD with Nesterov momentum, RAdam) | **88.6 %** | 0.75 |

- In v2, AdamW was clearly the best optimiser; with the same settings SGD and RAdam reached only about 30 % accuracy.
- Boletus (F1 0.95) and Hygrocybe and Lactarius (0.90) are recognised best; Suillus, Entoloma and Agaricus are the
  hardest, and Lactarius and Russula are often confused with each other.

## What the EDA covers

- Class imbalance (Lactarius and Boletus are the largest classes, Agaricus and Suillus the smallest) and image sizes
- Colour and texture features (RGB statistics, local binary patterns), clustering and outlier detection
- Statistical tests on colour differences between genera: ANOVA, bootstrap confidence intervals, chi-square test,
  MANOVA and effect sizes

## Repository

| File | Content |
|---|---|
| [`EDA.ipynb`](EDA.ipynb) | Exploratory and statistical analysis of the images |
| [`Modeling_v1.ipynb`](Modeling_v1.ipynb) | First model: ResNet34 transfer learning and evaluation |
| [`Modeling_v2.ipynb`](Modeling_v2.ipynb) | Improved architecture and optimiser comparison |
| `config.py` | Paths and training settings (image size 224, batch size 32, learning rate 1e-4) |
| `mushroom_dataset.py`, `mushroom_data_module.py` | Dataset and Lightning data module for v1 |
| `improved_mushroom_dataset.py`, `improved_mushroom_data_module.py` | The same for v2 |

## Data

[Mushrooms classification: Common genus's images](https://www.kaggle.com/datasets/maysee/mushrooms-classification-common-genuss-images)
on Kaggle (not included).

## Run it

1. Download the dataset and set `DATA_PATH` in `config.py` to its folder.
2. `pip install -r requirements.txt` (a CUDA GPU is recommended).
3. Run `EDA.ipynb`, then `Modeling_v1.ipynb` and `Modeling_v2.ipynb`.

## License

MIT
