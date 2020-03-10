# 184.702 Machine Learning
# 2019W
# Exercice3, Group 12, Topic 3.4: Generation and evaluation of unstructured synthetic datasets
# Lukas Lehner 01126793
# Majlinda Llugiqi 11931216
# Mal Kurteshi 11924480

# Run this file from ./src folder.

import torch
from torch import nn, optim
import mlex
from sig_discriminator import D34 as D
from sig_generator import G2a as G

# Setup new model
print('Setting up new model.')
M = mlex.SIG(random_state=1, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
M.set_dataset(dataset_name='FIDS30', data_root='../data/', image_size=(64, 64), transform=None)
M.load_split_dataset(test_size=0.1)
n = len(M.dataset.classes)
M.set_models('SIG', D, G, n, n, models_root='../models/')
M.set_params(0.001, 0.001, 0.95, 0.95,
             d_optimizer=optim.SGD, g_optimizer=optim.SGD, criterion=nn.CrossEntropyLoss(), spiked_noise=False)

# # Load model
# print('Loading model.')
# rel_path = 'models/SIG-2002260221/SIG-2002260221.pth'
# M = torch.load('../%s' % rel_path)

# # Train a GAN on a subset of classes
# myClasses = None  # ['apples', 1, 2, 3]
# M.train_GAN(train_classes=myClasses, n_epochs=50, g_steps=8,
#             save_all=True, save_gen_img_step=1, save_gen_img_count=4)

# Train some classes separately
myClasses = [1, 5, 10, 15, 20, 25]
for i in myClasses:
    M.train_GAN(train_classes=[i], n_epochs=25, g_steps=8,
                save_all=True, save_gen_img_step=10, save_gen_img_count=1)

# Train all classes separately
# M.train_GAN_each_class(n_epochs=5, g_steps=8, save_all=True, save_gen_img_step=4, save_gen_img_count=4)

# Generate dataset. Not all classes need to have been trained.
dataset_dir = M.gen_dataset(img_count=30, classes=myClasses, gen_data_root='../generated_datasets/')

# Train Evaluation Model on generated dataset
# custom_path = '../generated_datasets/SIG-2002260221_gen_data_set_1'
M.train_E(dataset_dir, n_epochs=25)

# # Evaluate Evaluation Model on original dataset
M.eval_E(print_predictions=False)
