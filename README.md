## InfoQGAN
In this project we recreated the research done by Lee et al 2023 (https://arxiv.org/pdf/2309.01363.pdf)
Our main task was to demonstrate a working InfoQGAN. To do this we started by creating a GAN followed by developing a MINE network and later applying these to a quantum generator producing our final InfoQGAN. The working of these models is demonstrated by modelling 3 datasets: a central square, and offset circle and two financial assets to promote the application of an InfoQGAN to MPT. 

`./QGAN.ipynb` and `./InfoQGAN.ipynb` are the two key files that create a working version of their respective models. 

## Repo Structure

The notebooks `InfoQGAN.ipynb` and `QGAN.ipynb` contained in the top level directory go through the complete training processes for all three distributions.

The `src` directory contains all implementations of the QGAN and InfoQGAN in `src/InfoQGAN.py`, `src/QGAN.py`, and `src/quantum_generator.py`.

`src/utils.py` contains methods used for data visualization and metric calculations.

The `experiments/` folder contains Jupyter notebooks that document the development process of the InfoQGAN. Eveything in these notebooks is contained in a clean way within the `src/` and top level `InfoQGAN.ipynb` and `QGAN.ipynb` notebooks.

## Installation

To install the necessary dependencies for this project, run the following line in your virtual environment:

```pip install -r requirements.txt```

Run `QGAN.ipynb` and `InfoQGAN.ipynb` to train models and view the generated distributions.

## References 

The 2D Kolmogorov–Smirnov test used to evaluate distributions was taken from: https://github.com/syrte/ndtest which is under an MIT License.
