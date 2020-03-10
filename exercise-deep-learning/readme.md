# 184.702 Machine Learning

## Exercice3, Group 12, Topic 3.4: Generation and evaluation of unstructured synthetic datasets

## Group members

Lukas Lehner 01126793
Majlinda Llugiqi 11931216
Mal Kurteshi 11924480

## Getting started

Set the MASTER folder as your project root folder.
Used package versions are found in requirements.txt.
Use `pip install -r requirements.txt` to install.

We used the following Cuda version for parallel processing, but code should work without Cuda too.
NVIDIA (R) Cuda compiler driver: Cuda compilation tools, release 10.2, V10.2.89

Execute `sig_gan.py` from the src folder for a standard GAN training on 6 classes or adjust the parameters.
If mlex.py and other custom modules can't be found try adding `./src` to your path by typing
```
import sys
sys.path.append('src')
```

into the python console. 
