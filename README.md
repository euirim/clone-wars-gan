# CS236G Final Project: Clone Wars GAN

## Models

1. `dcgan`: Popular DCGAN model with 64x64 output.

2. `rel_dcgan`: DCGAN model with additional layers for 128x128 output and relativistic discriminator.

3. `ss_rel_dcgan`: `rel_dcgan` with self-supervision.

4. `ss_rel_gan_improved`: `ss_rel_dcgan` with spectral normalization and less frequent generator training.

## Getting Started

1. Install [conda](https://docs.conda.io/en/latest/) if you don't have it on your machine, ideally by installing [Anaconda](https://www.anaconda.com/).

2. Create the conda environment for the project.

```bash
conda env create -f env.yml
```

3. Activate the environment.

```bash
conda activate clone-wars
```

4. Install pre-commit hooks while in the root of this project.

```bash
pre-commit install
```

## Data Retrieval

The dataset contains copyrighted content so we cannot freely distribute it. Please contact us to obtain a private copy of this dataset.

## Training Model

1. Download dataset and place into `data/full` folder. This folder should contain another folder containing all images as its root.

2. Be sure to activate the clone-wars conda environment as specified in the *Getting Started* section.

3. Change directory into the model of your choice.

4. Run the training script.

```bash
python train.py
```

## Evaluating Model

1. Be sure you have fully trained a model.

2. Within the directory of the model of your choice (currently only rel_dcgan and ss_rel_dcgan), run the eval script.

```bash
python eval.py -n NUMBER_OF_IMAGES -d DEVICE MODEL_FILENAME
```

If the device flag is not specified, the model defaults to using GPU (if available). Type in "cpu" to use CPU.

## License
[MIT](https://choosealicense.com/licenses/mit/)
