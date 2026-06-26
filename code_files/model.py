import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import glob
import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, roc_auc_score, precision_recall_curve, auc
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import scikitplot as skplt
import sys
from PIL import Image

#function to set cuda usage
def set_device():
    if torch.cuda.is_available:
       dev = 'cuda:3'            # which cuda device to be used for training the model on GPU
    else:
       dev = 'cpu'
    return torch.device(dev)

# function to get mean and standard deviation of the tensors from EEG Images
def get_mean_and_std(loader):
    mean = 0.
    std = 0.
    total_images_count = 0
    for images, _ in loader:
        image_count_in_a_batch = images.size(0)   #in current batch
        # print(images.shape)
        images = images.view(image_count_in_a_batch, images.size(1), -1) # need to reshape to get the mean and std
        # print(images.shape)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += image_count_in_a_batch
    mean /= total_images_count
    std /= total_images_count
    return mean, std


# function to show transformed images if one wants to have a look of teh images before and after normalization
def show_transformed_images(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    batch = next(iter(loader))
    images, labels = batch

    grid = torchvision.utils.make_grid(images, nrow=3)
    plt.figure(figsize=(11, 11))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    print('labels: ', labels)


#function to evaluate the model on test set
def evaluate_model_on_test_set(model, test_loader):
    lst_all_predicted, lst_all_labels = [], []
    # switch the model from training mode to evaluation mode
    model.eval()  # will notify all your layers that now we are in validation mode, in this way the dropout will work in the evaluation mode instead of training mode
    predicted_correctly_on_epoch = 0
    total = 0
    device = set_device()
    # deactivate the auto-gradient engine to reduce th ememory usage and speed up the computations however we will not be able to back propagate
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(
                0)  # to keep track of how many images we have in total as the last batch may have less than declared batch size

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)  # 1 specifies 1 dimension to reduce

            predicted_correctly_on_epoch += (predicted == labels).sum().item()
            for i in range(len(predicted.tolist())):
                lst_all_predicted.append(predicted.tolist()[i])  # tolist() converts tensor to list
                lst_all_labels.append(labels.tolist()[i])

    epoch_acc = 100.00 * predicted_correctly_on_epoch / total
    print("     - Testing dataset.  Got %d out of %d images correctly (%.3f%%). "
          % (predicted_correctly_on_epoch, total, epoch_acc))
    return lst_all_labels, lst_all_predicted

#function to evaluate the model on test set
def evaluate_model_on_test_set_with_stat_dwt(model, test_loader):
    lst_all_predicted, lst_all_labels = [], []
    # switch the model from training mode to evaluation mode
    model.eval()  # will notify all your layers that now we are in validation mode, in this way the dropout will work in the evaluation mode instead of training mode
    predicted_correctly_on_epoch = 0
    total = 0
    device = set_device()
    # deactivate the auto-gradient engine to reduce th ememory usage and speed up the computations however we will not be able to back propagate
    with torch.no_grad():
        for data in test_loader:
            images, labels, static_feats = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(
                0)  # to keep track of how many images we have in total as the last batch may have less than declared batch size


            _, predicted = torch.max(outputs.data, 1)  # 1 specifies 1 dimension to reduce

            outputs = model(images)
            # Fuse ResNet output with static features
            fused_output = torch.cat((outputs, static_feats), dim=1)
            # Perform forward pass through the modified model
            logits = model.fc(fused_output)
            _, predicted = torch.max(logits.data, 1)  # 1 specifies 1 dimension to reduce
            predicted_correctly_on_epoch += (predicted == labels).sum().item()
            for i in range(len(predicted.tolist())):
                lst_all_predicted.append(predicted.tolist()[i])  # tolist() converts tensor to list
                lst_all_labels.append(labels.tolist()[i])

    epoch_acc = 100.00 * predicted_correctly_on_epoch / total
    print("     - Testing dataset.  Got %d out of %d images correctly (%.3f%%). "
          % (predicted_correctly_on_epoch, total, epoch_acc))
    return lst_all_labels, lst_all_predicted

#function to find average of last three epoch results(P, R, F1, Macro AVg P, Macro AVg R, Macro AVg F1)
def avg_scores_last_3_epochs(lst_dct):
    avg_P_s, avg_R_s, avg_F1_s, avg_P_ns, avg_R_ns, avg_F1_ns, avg_macro_P, avg_macro_R, avg_macro_F1 = 0,0,0,0,0,0,0,0,0
    for i in range(3):
        avg_P_ns += lst_dct[-(i+1)]['0']['precision']
        avg_R_ns += lst_dct[-(i+1)]['0']['recall']
        avg_F1_ns += lst_dct[-(i+1)]['0']['f1-score']
        avg_P_s += lst_dct[-(i+1)]['1']['precision']
        avg_R_s += lst_dct[-(i+1)]['1']['recall']
        avg_F1_s += lst_dct[-(i+1)]['1']['f1-score']
        avg_macro_P += lst_dct[-(i+1)]['macro avg']['precision']
        avg_macro_R += lst_dct[-(i+1)]['macro avg']['recall']
        avg_macro_F1 += lst_dct[-(i+1)]['macro avg']['f1-score']
    avg_P_ns=round(avg_P_ns/3,2)
    avg_R_ns=round(avg_R_ns/3,2)
    avg_F1_ns=round(avg_F1_ns/3,2)
    avg_P_s=round(avg_P_s/3,2)
    avg_R_s=round(avg_R_s/3,2)
    avg_F1_s=round(avg_F1_s/3,2)
    avg_macro_P=round(avg_macro_P/3,2)
    avg_macro_R=round(avg_macro_R/3,2)
    avg_macro_F1=round(avg_macro_F1/3,2)
    return avg_P_s,avg_R_s,avg_F1_s,avg_P_ns,avg_R_ns,avg_F1_ns,avg_macro_P,avg_macro_R,avg_macro_F1

# train the model while returning result dict for averaging the results of last 3 epochs
def train_nn_avg_last_3_epochs(model, train_loader, test_loader, criterion, optimizer, n_epochs):  # criterion = loss_fn
    device = set_device()
    lst_dct = []
    global y_true, y_pred
    for epoch in range(n_epochs):
        print('Epoch Number %d' % (epoch + 1))
        model.train()
        # to see how many images were classified correctly
        running_loss = 0.0
        running_correct = 0.0
        total = 0

        # now iterate through all of the batches
        for data in train_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            # before starting backpropagation we set grad to 0 so parameters gets updated correctly
            optimizer.zero_grad()

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)  # 1 specifies 1 dimension to reduce
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_correct += (labels == predicted).sum().item()

        epoch_loss = running_loss / len(train_loader)
        # check how many % of images were classified correctly
        epoch_acc = 100.00 * running_correct / total

        print("     - Training dataset.  Got %d out of %d images correctly (%.3f%%). Epoch loss:  %.3f"
              % (running_correct, total, epoch_acc, epoch_loss))

        y_true, y_pred = evaluate_model_on_test_set(model, test_loader)
        dct = classification_report(y_true, y_pred, output_dict=True)
        lst_dct.append(dct)

    print('Finished! ')
    return lst_dct

# model training
def train_nn(model, train_loader, test_loader, criterion, optimizer, n_epochs):  # criterion = loss_fn
    device = set_device()
    global y_true, y_pred
    for epoch in range(n_epochs):
        print('Epoch Number %d' % (epoch + 1))
        model.train()
        # to see how many images were classified correctly
        running_loss = 0.0
        running_correct = 0.0
        total = 0

        # now iterate through all of the batches
        for data in train_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            # before starting backpropagation we set grad to 0 so parameters gets updated correctly
            optimizer.zero_grad()

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)  # 1 specifies 1 dimension to reduce
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_correct += (labels == predicted).sum().item()

        epoch_loss = running_loss / len(train_loader)
        # check how many % of images were classified correctly
        epoch_acc = 100.00 * running_correct / total

        print("     - Training dataset.  Got %d out of %d images correctly (%.3f%%). Epoch loss:  %.3f"
              % (running_correct, total, epoch_acc, epoch_loss))

        y_true, y_pred = evaluate_model_on_test_set(model, test_loader)

    print('Finished! ')
    return model

#function to set model train test loader and other model parameters
def set_model(train_dataset_path,test_dataset_path):
    training_transforms = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=training_transforms)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=8, shuffle=False)
    mean, std = get_mean_and_std(train_loader)  #(tensor([ 0.4033,  0.4020,  0.3351]), tensor([ 0.1889,  0.1776,  0.1754]))
    training_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))])
    train_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=training_transforms)
    test_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))])
    test_dataset = torchvision.datasets.ImageFolder(root=test_dataset_path, transform=test_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)
    # we will use mini-batch gradient descent as algorithm for learning, so need batches
    resnet18_model = models.resnet18(pretrained=True) #True gives the model already trained on 1000 imagenet class, #False gives model with random weights to start from scratch
    num_ftrs = resnet18_model.fc.in_features  # size of each input sample
    number_of_classes = 2  # earlier was 30
    resnet18_model.fc = nn.Linear(num_ftrs,number_of_classes) #prepare matrices for forward propagation by taking inp & out
    device = set_device()
    print('device being used is: ', device)
    resnet18_model = resnet18_model.to(device)
    loss_fn = nn.CrossEntropyLoss() #gives greater penalty when incorrect predictions are done with high confidence
    optimizer = optim.SGD(resnet18_model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.003) # weight_decay to prevent the overfitting as our train dataset is small
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Model will run on ',device)
    #n_epochs = 150 if pretrained is False else 5 epochs
    return resnet18_model, train_loader, test_loader, loss_fn, optimizer

# function to oversample the seizure onset training set(minority class)
def oversample_minority_class(train_dataset_path):
    # Oversample the minority(seizure onset) class of the train set to avoid any bias
    ### to be run only once for oversampling of the minority class
    sz_train_dataset_path = train_dataset_path + '/sz'  # assuming "sz" is the folder which contains all seizure onset training images
    ##copying the files 15 times to equalize both classes as the seizure-onsets:non-seizure is in ratio 1:15.
    for fin in glob.glob(sz_train_dataset_path + '/*'):
        for i in range(15):
            new_fin = fin.replace('.png', '__' + str(i + 1) + '.png')
            shutil.copy(fin, new_fin)

#defining a class to load dataset of dwt static features into the data loader along with images
class StaticFeaturesDataset(Dataset):
    def __init__(self, static_features):
        self.static_features = static_features

    def __len__(self):
        return len(self.static_features)

    def __getitem__(self, idx):
        static_feat = self.static_features[idx]
        return static_feat

# set model parameters and train test loaders along with provision of passing dwt static features fused with ResNET18 output to FCNN
def set_model_with_stat_dwt(train_dataset_path, test_dataset_path, valid_fnsz_onset_dwt_feat_test_df,
valid_fnsz_onset_dwt_feat_train_df, valid_non_seiz_dwt_feat_test_df, valid_non_seiz_dwt_feat_train_df):
    # perform normalization of images and then load them in train loader along with static features
    training_transforms = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=training_transforms)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=8, shuffle=False)
    mean, std = get_mean_and_std(train_loader)
    training_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))])
    train_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=training_transforms)
    test_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))])
    test_dataset = torchvision.datasets.ImageFolder(root=test_dataset_path, transform=test_transforms)

    #creating instances of StaticFeaturesDataset for train and test data
    #find how many times '_pd1' is present to get no of nodes for which the dwt df was made during feature extraction
    lst_all_features = list(valid_fnsz_onset_dwt_feat_train_df.columns)
    ctr_pd1 = 0
    for item in lst_all_features:
        if '_pd1' in item:
            ctr_pd1+=1
    #ctr_pd1 will give the number of times i.e. for the number of nodes permutation_entropy feature was made
    tot_num_dwt_feat_in_dfs = len(ctr_pd1)*35   #is the total number of dwt features present in the df
    #in dfs dwt features start from the 6th column, prior to that exists patient info fields

    train_static_dataset = StaticFeaturesDataset(pd.concat([valid_fnsz_onset_dwt_feat_train_df[,5:tot_num_dwt_feat_in_dfs-1],valid_non_seiz_dwt_feat_train_df[,5:tot_num_dwt_feat_in_dfs-1]], ignore_index=True))
    test_static_dataset = StaticFeaturesDataset(pd.concat([valid_fnsz_onset_dwt_feat_test_df[,5:tot_num_dwt_feat_in_dfs-1],valid_non_seiz_dwt_feat_test_df[,5:tot_num_dwt_feat_in_dfs-1]], ignore_index=True))

    train_loader = DataLoader(torch.utils.data.ConcatDataset([train_dataset, train_static_dataset]), batch_size=8,
                              shuffle=True)
    test_loader = DataLoader(torch.utils.data.ConcatDataset([test_dataset, test_static_dataset]), batch_size=8,
                             shuffle=False)

    resnet18_model = models.resnet18(pretrained=True)  #True gives the model already trained on 1000 imagenet class, #False gives model with random weights to start from scratch
    num_ftrs = resnet18_model.fc.in_features + tot_num_dwt_feat_in_dfs  # size of each input sample
    number_of_classes = 2
    resnet18_model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),  #additional layer for fusion
        nn.ReLU(inplace=True),
        nn.Linear(512, number_of_classes)  # Binary classification (change 2 to the number of classes)
    )
    device = set_device()
    print('device being used is: ', device)
    resnet18_model = resnet18_model.to(device)
    loss_fn = nn.CrossEntropyLoss()  # gives greater penalty when incorrect predictions are done with high confidence
    optimizer = optim.SGD(resnet18_model.parameters(), lr=0.001, momentum=0.9,
                          weight_decay=0.003)  # weight_decay to prevent the overfitting as our train dataset is small
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Model will run on ', device)
    return resnet18_model, train_loader, test_loader, loss_fn, optimizer

# train teh neural netwrok and report scores by averaging teh last 3 epochs results; along with provision of passing dwt static features fused with ResNET18 output to FCNN
def train_nn_avg_last_3_epochs_with_stat_dwt(resnet18_model, train_loader, test_loader, criterion, optimizer, num_epochs):
    device = set_device()
    lst_dct = []
    global y_true, y_pred
    for epoch in range(num_epochs):
        print('Epoch Number %d' % (epoch + 1))
        resnet18_model.train()
        # to see how many images were classified correctly
        running_loss = 0.0
        running_correct = 0.0
        total = 0

        # now iterate through all batches
        for data in train_loader:
            images, labels, static_feats = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            # before starting backpropagation we set grad to 0 so parameters gets updated correctly
            optimizer.zero_grad()

            outputs = resnet18_model(images)
            # Fuse ResNet output with static features
            fused_output = torch.cat((outputs, static_feats), dim=1)
            # Perform forward pass through the modified model
            logits = resnet18_model.fc(fused_output)
            _, predicted = torch.max(logits.data, 1)  # 1 specifies 1 dimension to reduce
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_correct += (labels == predicted).sum().item()

        epoch_loss = running_loss / len(train_loader)
        # check how many % of images were classified correctly
        epoch_acc = 100.00 * running_correct / total

        print("     - Training dataset.  Got %d out of %d images correctly (%.3f%%). Epoch loss:  %.3f"
              % (running_correct, total, epoch_acc, epoch_loss))

        y_true, y_pred = evaluate_model_on_test_set_with_stat_dwt(resnet18_model, test_loader)
        dct = classification_report(y_true, y_pred, output_dict=True)
        lst_dct.append(dct)

    print('Finished! ')
    return lst_dct

#function to run the model and see average of last 3 epoch results
def run_model_with_avg_epochs_results(train_dataset_path, test_dataset_path, num_epochs, stat_dwt=None):
    oversample_minority_class(train_dataset_path)
    if stat_dwt is None:
       #setting the model for run
       resnet18_model, train_loader, test_loader, loss_fn, optimizer = set_model(train_dataset_path, test_dataset_path)
       lst_dct = train_nn_avg_last_3_epochs(resnet18_model, train_loader, test_loader, loss_fn, optimizer, num_epochs)
    elif stat_dwt == 'Y':
        # load dwt file corresponding to train and test sets as processed earlier in features section
        valid_fnsz_onset_dwt_feat_test_df = pd.read_pickle(train_dataset_path.replace('/all_cb/train/sz','/') + 'fnsz_onset_dwt_feat_test.pkl')
        valid_fnsz_onset_dwt_feat_train_df = pd.read_pickle(train_dataset_path.replace('/all_cb/train/sz','/') + 'fnsz_onset_dwt_feat_train.pkl')
        valid_non_seiz_dwt_feat_test_df = pd.read_pickle(train_dataset_path.replace('/all_cb/train/sz','/') + 'non_seiz_dwt_feat_test.pkl')
        valid_non_seiz_dwt_feat_train_df = pd.read_pickle(train_dataset_path.replace('/all_cb/train/sz','/') + 'non_seiz_dwt_feat_train.pkl')
        # setting the model for run
        resnet18_model, train_loader, test_loader, loss_fn, optimizer = set_model_with_stat_dwt(train_dataset_path, test_dataset_path, valid_fnsz_onset_dwt_feat_test_df,
valid_fnsz_onset_dwt_feat_train_df, valid_non_seiz_dwt_feat_test_df, valid_non_seiz_dwt_feat_train_df)
        lst_dct = train_nn_avg_last_3_epochs_with_stat_dwt(resnet18_model, train_loader, test_loader, loss_fn, optimizer, num_epochs)
    avg_P_s, avg_R_s, avg_F1_s, avg_P_ns, avg_R_ns, avg_F1_ns, avg_macro_P, avg_macro_R, avg_macro_F1 = avg_scores_last_3_epochs(lst_dct)
    print('Last 3 Epochs:- avg_P_s: {}, avg_R_s: {}, avg_F1_s: {}, avg_P_ns: {}, avg_R_ns: {}, avg_F1_ns: {}, avg_macro_P: {}, avg_macro_R: {}, avg_macro_F1: {}'.format(
            avg_P_s, avg_R_s, avg_F1_s, avg_P_ns, avg_R_ns, avg_F1_ns, avg_macro_P, avg_macro_R, avg_macro_F1))



# function to run model and print last epoch results
def run_model(train_dataset_path, test_dataset_path, num_epochs):
    resnet18_model, train_loader, test_loader, loss_fn, optimizer = set_model(train_dataset_path, test_dataset_path)
    y_true, y_pred, target_names = [], [], ['class_' + str(i + 1) for i in range(2)]
    train_nn(resnet18_model, train_loader, test_loader, loss_fn, optimizer, num_epochs)
    print('\nClassification Report for non-seiz(label -> 0) vs seiz(label -> 1): \n\n',
          classification_report(y_true, y_pred))


if __name__ == '__main__':
    # list of EEG Graph nodes presented in lists as per different brain regions
    lst_temporal, lst_central, lst_frontal, lst_parietal, lst_occipital = ['T3', 'T5', 'T4', 'T6'], ['C3', 'CZ', 'C4'], ['F3', 'F7', 'FP1', 'FZ', 'FP2', 'F4', 'F8'], ['P3', 'PZ', 'P4'], ['O1', 'O2']
    lst_all = lst_temporal + lst_central + lst_frontal + lst_parietal + lst_occipital
    # Check if correct number of arguments is provided
    if len(sys.argv) < 3:
        print('Please at least give train_dataset_path and test_dataset_path to proceed!')
        print("Usage: python3 model.py train_dataset_path test_dataset_path num_epochs")
        sys.exit()
    else:
        try:
            train_dataset_path = sys.argv[1]
            test_dataset_path = sys.argv[2]
            num_epochs = sys.argv[3]      # by default 10
        except Exception as ex:
            if len(sys.argv) == 3:
                num_epochs = 10  # by default we keep the number of epochs to be 10
            else:
                print('Please at least give train_dataset_path and test_dataset_path to proceed!')
                print("Usage: python3 model.py train_dataset_path test_dataset_path num_epochs")
                sys.exit()
    # run the ResNet18 model and print average of last 3 epoch results
    run_model_with_avg_epochs_results(train_dataset_path, test_dataset_path, num_epochs)
    # run the ResNet18 model and print last epoch results
    run_model(train_dataset_path, test_dataset_path, num_epochs)
    # run the ResNet18 model with static dwt features fused in FCNN and print average of last 3 epoch results
    run_model_with_avg_epochs_results(train_dataset_path, test_dataset_path, num_epochs, 'Y')