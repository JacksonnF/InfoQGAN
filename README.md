## InfoQGAN
In this project we recreated the research done by Lee et al 2023 (https://arxiv.org/pdf/2309.01363.pdf)
Our main task was to demonstrate a working InfoQGAN. To do this we started by creating a GAN followed by developing a MINE network and later applying these to a quantum generator producing our final InfoQGAN. The working of these models is demonstrated by modelling 3 datasets: a central square, and offset circle and two financial assets to promote the application of an InfoQGAN to MPT. 

QGAN.ipynb and InfoQGAN.ipynb are the two key files that create a working version of their respective models. 

## Source Folder

The 'src' contains the source code for the creation of a QGAN and an InfoQGAN. The file 'utils' has functions for plotting and developing the models. 


## Experiments Folder

The `experiments` folder contains Jupyter notebooks that document the development process of the InfoQGAN. These notebooks provide insights into design decisions and parameter tunings. Whereas the Python files were derived from these notebooks, ensuring a tested and validated implementation. 

## Installation

To install the necessary dependencies for this project, run the following line:

```pip install -r requirements.txt```

Run `QGAN.ipynb and InfoQGAN.ipynb` to train models and view the generated distributions.
