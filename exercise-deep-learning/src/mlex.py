# Collection of functions for ML exercise 3, topic 3.4
import csv
import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torch import nn, optim
import torchvision
from torchvision import transforms
from sklearn.model_selection import train_test_split


class SIG:

    def __init__(self, random_state=0, device=torch.device('cpu')):
        # Meta
        self.name = None
        self.__id__ = datetime.datetime.now().strftime("%y%m%d%H%M")
        self.random_state = random_state
        self.device = device

        # Models
        self.model_dir = None
        self.D_fun = None  # = E_fun
        self.D_fun_input = None
        self.G_fun = None
        self.G_fun_input = None
        self.D = {}  # dict
        self.G = {}  # dict
        self.E = None
        self.E_classes = None

        # Dataset
        self.dataset_name = None
        self.dataset_dir = None
        self.dataset = None
        self.n_classes = None
        self.image_size = None
        self.transform = None
        self.test_size = None
        self.test_sampler = None
        self.train_sampler = None

        # Training parameters
        self.dlr = 0
        self.glr = 0
        self.dm = 0
        self.gm = 0
        self.criterion = None
        self.d_optimizer_fun = None
        self.g_optimizer_fun = None
        self.d_optimizer = {}
        self.g_optimizer = {}
        self.e_optimizer = None
        # self.d_cur_loss = {}
        # self.g_cur_loss = {}
        self.d_trained_epochs = {}
        self.g_trained_epochs = {}
        self.e_trained_epochs = 0
        self.spiked_noise = False

    # SIG saving
    def check_dir(self, directory=None):
        try:
            os.mkdir(self.model_dir)
        except FileExistsError:
            pass
        if directory is not None:
            try:
                os.mkdir(directory)
            except FileExistsError:
                pass

    def save(self):
        self.check_dir()
        torch.save(self, self.get_path() + '.pth')

    # Model handling
    def set_models(self, name, D_fun, G_fun, D_fun_input=None, G_fun_input=None, models_root='../models/'):
        self.name = name
        self.D_fun = D_fun
        self.D_fun_input = D_fun_input
        self.G_fun = G_fun
        self.G_fun_input = G_fun_input
        self.model_dir = models_root + '%s-%s' % (self.name, self.__id__)

    def get_path(self):
        return self.model_dir + '/%s-%s' % (self.name, self.__id__)

    # Dataset
    def set_dataset(self, dataset_name='FIDS30', data_root='../data/', image_size=(64, 64), transform=None):
        self.dataset_name = dataset_name
        if data_root is not None:
            self.dataset_dir = data_root + dataset_name
        self.image_size = image_size
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(self.image_size[0]),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        self.transform = transform

    def load_split_dataset(self, test_size=0.1):
        self.test_size = test_size

        try:
            os.mkdir(self.dataset_dir + '/_fake')
        except FileExistsError:
            pass

        self.dataset = torchvision.datasets.ImageFolder(root=self.dataset_dir + '/',
                                                        transform=self.transform)

        train_idx, test_idx = train_test_split(
            np.arange(len(self.dataset)),
            test_size=test_size,
            stratify=self.dataset.targets,
            random_state=self.random_state
        )
        self.train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
        self.test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_idx)
        self.n_classes = len(self.dataset.classes)

    def get_loader(self, sampler=None, batch_size=16, num_workers=0):
        if sampler is None:
            sampler = self.train_sampler

        return torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True
        )

    def get_class_loader(self, arg_classes, batch_size=8, num_workers=0, train_data_only=True):

        if type(arg_classes) is list or type(arg_classes) is dict:
            classes = arg_classes.copy()
        else:
            classes = [arg_classes]

        for i, c in enumerate(classes):
            if type(c) is not int:
                try:
                    classes[i] = self.dataset.classes.index(c)
                except ValueError:
                    raise Exception('Error during get_class_loader(): class %s not found.' % c)
            else:
                if c >= self.n_classes or c < 0:
                    raise Exception('Error during get_class_loader(): class %d out of range.' % c)

        indices = [i for i, x in enumerate(self.dataset.targets) if [j for j, y in enumerate(classes) if x == y]]

        # delete test indices
        if train_data_only:
            try:
                for i in [x for x in self.test_sampler.indices if [y for y in indices if x == y]]:
                    indices.remove(i)
            except AttributeError:
                pass

        if not len(indices):
            raise Exception('Error in get_class_loader(): Loader is empty. Classes:', classes)

        return self.get_loader(torch.utils.data.sampler.SubsetRandomSampler(indices), batch_size, num_workers)

    # Training parameters
    def set_params(self, dlr, glr, dm, gm,
                   d_optimizer=optim.SGD, g_optimizer=optim.SGD, criterion=nn.CrossEntropyLoss(), spiked_noise=False):
        self.dlr = dlr
        self.glr = glr
        self.dm = dm
        self.gm = gm
        self.d_optimizer_fun = d_optimizer  # (self.D[].parameters(), lr=dlr, momentum=dm)
        self.g_optimizer_fun = g_optimizer  # (self.G[].parameters(), lr=glr, momentum=gm)
        self.criterion = criterion
        self.spiked_noise = spiked_noise

    # Training
    def train_GAN(self, train_classes=None, n_epochs=1, g_steps=5,
                  save_all=False, checkpoint_step=0, unload_to_cpu=False,
                  save_gen_img_step=0, save_gen_img_count=4):  # TODO: checkpoints

        fake_class_idx = self.dataset.classes.index('_fake')
        m = ''
        device = self.device

        if train_classes is None:
            m = 'all'
            train_loader = self.get_loader()
            class_idxs = [j for j in range(len(self.dataset.classes))]
        else:
            if type(train_classes) is not list:
                train_classes = [train_classes]
            train_classes = train_classes.copy()

            for i, c in enumerate(train_classes):
                if type(c) is int:
                    c = self.dataset.classes[c]
                    train_classes[i] = c
                m += c + '-'
            train_loader = self.get_class_loader(train_classes)
            class_idxs = [i for i, x in enumerate(self.dataset.classes) if
                          [j for j, y in enumerate(train_classes) if x == y]]

        # First training
        try:
            self.D[m]
        except KeyError:
            try:
                self.D[m] = self.D_fun(self.D_fun_input)
            except TypeError:
                self.D[m] = self.D_fun()
            self.d_trained_epochs[m] = 0
            self.d_optimizer[m] = self.d_optimizer_fun(self.D[m].parameters(), lr=self.dlr, momentum=self.dm)
        try:
            self.G[m]
        except KeyError:
            try:
                self.G[m] = self.G_fun(self.G_fun_input)
            except TypeError:
                self.G[m] = self.G_fun()
            self.g_trained_epochs[m] = 0
            self.g_optimizer[m] = self.g_optimizer_fun(self.G[m].parameters(), lr=self.glr, momentum=self.gm)

        #
        self.D[m] = self.D[m].to(device)
        self.G[m] = self.G[m].to(device)

        n_batches = int(len(train_loader.sampler.indices) / train_loader.batch_size) + 1
        since = datetime.datetime.now()
        print('Starting training on class(es) %s.' % m)

        for epoch in range(n_epochs):
            e_start = datetime.datetime.now()
            print(e_start.strftime("%H.%M:%S:"), 'Epoch %d of %d' % (epoch + 1, n_epochs))
            if epoch is 0:
                print('Training discriminator: ', end='')

            for i, data in enumerate(train_loader, 0):
                if epoch is 0:
                    print('%d%% ' % (100 * (i + 1) / n_batches), end='')

                self.D[m].zero_grad()

                d_real_data, d_real_labels = data[0].to(device), data[1].to(device)
                d_real_decision = self.D[m](d_real_data)
                d_real_error = self.criterion(d_real_decision, d_real_labels)
                d_real_error.backward()

                d_gen_input = self.G[m].get_noise_batch(d_real_labels, self.spiked_noise).to(device)
                d_fake_labels = (torch.ones(len(d_real_labels)) * fake_class_idx).type(torch.long).to(device)
                d_fake_data = self.G[m](d_gen_input).detach()
                d_fake_decision = self.D[m](d_fake_data)
                d_fake_error = self.criterion(d_fake_decision, d_fake_labels)
                d_fake_error.backward()
                self.d_optimizer[m].step()

            self.d_trained_epochs[m] += 1

            if epoch is 0:
                print('\nTraining generator: ', end='')
            for j in range(g_steps):  # TODO optimize parallelism
                if epoch is 0:
                    print('%d%% ' % (100 * (j + 1) / g_steps), end='')
                for k in class_idxs:
                    # Training for class k
                    self.G[m].zero_grad()
                    g_labels = k * torch.ones(train_loader.batch_size).type(torch.long).to(device)

                    gen_input = self.G[m].get_noise_batch(g_labels, self.spiked_noise).to(device)
                    g_fake_data = self.G[m](gen_input)
                    dg_fake_decision = self.D[m](g_fake_data)
                    g_error = self.criterion(dg_fake_decision, g_labels)

                    g_error.backward()
                    self.g_optimizer[m].step()

            self.g_trained_epochs[m] += 1

            if save_gen_img_step is not 0 and (epoch % save_gen_img_step is 0 or epoch is n_epochs - 1):
                self.gen_and_save_img(m, save_gen_img_count)

            if epoch is 0:
                delta_t = (datetime.datetime.now() - since).seconds
                print(
                    '\nFirst of %d epochs complete. Time passed: %dh %dm %ds. Estimated total runtime: %dh %dm %ds' % (
                        n_epochs,
                        delta_t / 3600, delta_t / 60 % 60, delta_t % 60,
                        (n_epochs * delta_t) / 3600, (n_epochs * delta_t) / 60 % 60, (n_epochs * delta_t) % 60
                    ))

        delta_t = (datetime.datetime.now() - since).seconds
        print('\n Training complete. Time elapsed: %dh %dm %ds.\n' % (delta_t / 3600, delta_t / 60 % 60, delta_t % 60))

        if save_all:
            self.save()

        if unload_to_cpu:
            device = torch.device('cpu')
            self.D[m] = self.D[m].to(device)
            self.G[m] = self.G[m].to(device)

    def train_GAN_each_class(self, n_epochs=10, g_steps=10, save_all=True, save_gen_img_step=1, save_gen_img_count=16):
        print('Training all %d classes.' % (self.n_classes - 1))
        for i, c in enumerate(self.dataset.classes):
            if c != '_fake':
                print('Training class %s (%d of %d).' % (c, i + 1, self.n_classes))
                self.train_GAN(train_classes=[c], n_epochs=n_epochs, g_steps=g_steps,
                               save_all=save_all, unload_to_cpu=True,
                               save_gen_img_step=save_gen_img_step, save_gen_img_count=save_gen_img_count)

    def train_E(self, dataset_root, n_epochs):
        device = self.device

        try:
            train_loader = get_dataset_loader(dataset_root, self.transform)
        except FileNotFoundError:
            print(
                'Error during train_E(): No generated dataset found at %s. Generate it first using gen_dataset(). E '
                'not trained.' % dataset_root)
            return

        if self.E is None:
            try:
                self.E = self.D_fun(self.D_fun_input)
            except TypeError:
                self.E = self.D_fun()
            self.e_trained_epochs = 0
            self.E_classes = train_loader.dataset.classes
            self.e_optimizer = self.d_optimizer_fun(self.E.parameters(), lr=self.dlr, momentum=self.dm)
        else:
            if self.E_classes != train_loader.dataset.classes:
                raise Exception('Error in train_E(): Previously trained classes and new training classes dont match: '
                                '%s, %s. Set self.E to None and then retrain.' % (
                                    self.E_classes, train_loader.dataset.classes))

        self.E.to(device)
        n_batches = int(len(train_loader.sampler) / train_loader.batch_size) + 1
        since = datetime.datetime.now()

        for epoch in range(n_epochs):
            e_start = datetime.datetime.now()
            print(e_start.strftime("%H.%M:%S:"), 'Epoch %d of %d' % (epoch + 1, n_epochs))
            if epoch is 0:
                print('Training Evaluator: ', end='')
            for i, data in enumerate(train_loader, 0):
                if epoch is 0:
                    print('%d%% ' % (100 * (i + 1) / n_batches), end='')

                self.E.zero_grad()

                imgs, labels = data[0].to(device), data[1].to(device)
                decision = self.E(imgs)
                error = self.criterion(decision, labels)
                error.backward()
                self.e_optimizer.step()

            self.e_trained_epochs += 1

            if epoch is 0:
                delta_t = (datetime.datetime.now() - since).seconds
                print(
                    '\nFirst of %d epochs complete. Time passed: %dh %dm %ds. Estimated total runtime: %dh %dm %ds' % (
                        n_epochs,
                        delta_t / 3600, delta_t / 60 % 60, delta_t % 60,
                        (n_epochs * delta_t) / 3600, (n_epochs * delta_t) / 60 % 60, (n_epochs * delta_t) % 60
                    ))

        delta_t = (datetime.datetime.now() - since).seconds
        print('\n Training complete. Time elapsed: %dh %dm %ds.\n' % (delta_t / 3600, delta_t / 60 % 60, delta_t % 60))

        if True:
            self.save()

    # Evaluation
    def eval_E(self, batch_size=8, print_predictions=False):
        print('Evaluating the Evaluator:')
        device = self.device
        if self.E is None:
            raise Warning('Warning in eval_E(): Model E has not been trained.')
        self.E = self.E.to(device)

        total = 0
        correct = 0
        with torch.no_grad():
            for i, c in enumerate(self.E_classes):
                loader = self.get_class_loader(c, batch_size=batch_size, train_data_only=False)
                g_idx = self.dataset.classes.index(c)

                class_correct = 0
                class_total = 0
                for j, data in enumerate(loader, 0):
                    imgs, _ = data[0].to(device), data[1].to(device)
                    outputs = self.E(imgs)
                    _, predicted = torch.max(outputs.data, 1)
                    labels = i * torch.ones(len(predicted)).to(device)
                    class_correct += (predicted == labels).sum().item()
                    class_total += len(labels)

                    if print_predictions:
                        print('\nInput class: %s' % c)
                        print('Predicted classes: \n',
                              ' '.join('%15s' % self.E_classes[predicted[j]] for j in range(len(predicted))))

                print('Correct predictions for class %15s: %2d/%2d (%2d%%)'
                      % (c, class_correct, class_total,
                         100 * class_correct / class_total))
                correct += class_correct
                total += class_total

        accuracy = 100 * correct / total
        print('\nAccuracy: %.2f%% (%d/%d)' % (accuracy, correct, total))
        print('E trained epochs: %d' % self.e_trained_epochs)
        print('Chance when random guessing: %.2f%%' % (100 / len(self.E_classes)))
        self.E = self.E.to(torch.device('cpu'))

    def gen_and_save_img(self, model_name, img_count=4, classes_idx=None):
        m = model_name
        device = self.device
        img_dir = '%s/%s' % (self.model_dir, model_name)
        self.check_dir(img_dir)
        path = '%s/%s-%s-%sgen-img-at-epoch-%d' % (img_dir, self.name, self.__id__,
                                                   model_name, self.g_trained_epochs[m])
        self.check_dir()
        if not self.spiked_noise or classes_idx is None:
            with torch.no_grad():
                gen_input = self.G[m].get_noise_batch(torch.tensor(range(img_count)), spiked=self.spiked_noise).to(
                    device)
                g_fake_data = self.G[m](gen_input).to(device)
                torchvision.utils.save_image(g_fake_data, path + '.png', normalize=False)
        else:
            with torch.no_grad():
                if type(classes_idx) is not list:
                    classes_idx = [classes_idx]
                for i in classes_idx:
                    gen_input = self.G[m].get_noise_batch(torch.ones(img_count) * i, spiked=self.spiked_noise).to(
                        device)
                    g_fake_data = self.G[m](gen_input).to(device)
                    torchvision.utils.save_image(g_fake_data,
                                                 path + '-' + self.dataset.classes[i] + '.png',
                                                 normalize=False)

    def gen_dataset(self, img_count, classes=None, gen_data_root='../generated_datasets/', device=torch.device('cpu')):
        dataset_dir = None

        if not self.spiked_noise:

            if classes is None:
                classes = [j for j in range(len(self.dataset.classes))]

            if type(classes) is not list:
                classes = [classes]
            else:
                classes = classes.copy()
            for i, c in enumerate(classes):
                if type(c) is int:
                    c = self.dataset.classes[c]
                    classes[i] = c

            dataset_dir = gen_data_root + '%s-%s_gen_data_set' % (self.name, self.__id__)
            k = 0
            while k > -1:
                try:
                    os.mkdir(dataset_dir + '_' + str(k))
                    dataset_dir += '_' + str(k)
                    k = -1
                except FileExistsError:
                    k += 1
            torch.save(self.g_trained_epochs, dataset_dir + '/g_trained_epochs.dict')

            with torch.no_grad():
                for c in classes:
                    try:
                        self.G[c + '-'] = self.G[c + '-'].to(device)
                    except KeyError:
                        if c != '_fake':
                            print('gen_dataset(): Class %10s is not trained. Has been skipped.' % c)
                        continue

                    class_dir = dataset_dir + '/' + c
                    try:
                        os.mkdir(class_dir)
                    except FileExistsError:
                        pass

                    label = (self.dataset.classes.index(c) * torch.tensor([1])).to(device)
                    for i in range(img_count):
                        noise = self.G[c + '-'].get_noise_batch(label, spiked=False)
                        img = self.G[c + '-'](noise)
                        torchvision.utils.save_image(img,
                                                     class_dir + '/%d.png' % i,
                                                     normalize=False)
        else:
            print('Spiked gen_dataset() version not yet implemented.')

        return dataset_dir


def get_dataset_loader(dataset_root, transform, sampler=None, batch_size=16, num_workers=0, shuffle=True):
    return torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(root=dataset_root, transform=transform),
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True
    )


# TODO
def writeToCSV(nameOfCSV, setup, mlmodel):
    file_exists = os.path.isfile('../reports/' + nameOfCSV + '.csv')
    with open(r'../reports/' + nameOfCSV + '.csv', 'a') as csvfile:
        headers = ['ID', 'Model_Name', 'Batch_size', 'Accuracy']
        writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n', fieldnames=headers)
        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header
        writer.writerow(
            {'ID': str(setup.setup_id), 'Model_Name': mlmodel.model_name, 'Batch_size': str(setup.batch_size),
             'Accuracy': str(mlmodel.accuracy)})


# TODO
def imgshow(img, labels=None):
    img = img / 2 + 0.5  # TODO: unnormalize with respect to transforms other than 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# TODO
def show_rand_img(loader, n_im=8):
    dataiter = iter(loader)
    images, labels = dataiter.next()
    print(' '.join('%5s' % loader.dataset.classes[labels[j]] for j in range(n_im)))
    imgshow(torchvision.utils.make_grid(images))


# TODO
def plotCorrelationMatrixHeatMap(dataset, pic_name):
    plt.rcParams.update({'font.size': 35})
    corrmat = dataset.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(50, 50))
    g = sns.heatmap(dataset[top_corr_features].corr())
    plt.savefig("./graphs/image_segmentation/" + pic_name + ".png")
