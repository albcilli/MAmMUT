import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.init as init
import pandas as pd
import os
import numpy as np
import cv2
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from data import SortDataset;
from model_deepfashion import FashionNet;
from sklearn.metrics import fbeta_score, accuracy_score, precision_score, recall_score
import gc

# Set the device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fulldata = pd.read_csv('./DeepFashion_normalized.csv')
images = np.empty((len(fulldata), 300, 200))
for i in range(len(fulldata)):
    imagedata = cv2.imread(fulldata['image_name'].iloc[i],0)
    imagedata = imagedata/255
    images[i, :, :] = imagedata

x = fulldata.iloc[:, 4:-1].values
y = fulldata.iloc[:, -1].values

seed = 1000
batch_size = 1024
num_classes = 46
n_folds = 5
k = 1

rf_test_acc = 0
rf_test_prec_w = 0
rf_test_rec_w = 0
rf_test_f1_w = 0
rf_test_f2_w = 0

nn_test_acc = 0
nn_test_prec_w = 0
nn_test_rec_w = 0
nn_test_f1_w = 0
nn_test_f2_w = 0



test_split = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
for trainval_index, test_index in test_split.split(x, y):
    print('\nFold %1.0f' % (k))
    
    x_test, y_test = x[test_index], y[test_index]
    images_test = images[test_index, :, :]
    x_trainval, y_trainval = x[trainval_index], y[trainval_index]
    images_trainval = images[trainval_index, :, :]
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=10)
    
    val_split = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=seed) #for nested-CV change with kfold
    for train_index, val_index in val_split.split(x_trainval, y_trainval):
        x_train, x_val = x_trainval[train_index], x_trainval[val_index]
        y_train, y_val = y_trainval[train_index], y_trainval[val_index]
      
        clf.fit(x_train, y_train)
        rf_train_acc = clf.score(x_train, y_train)
        rf_val_acc = clf.score(x_val, y_val)
        print('Random forest training and validation accuracies: %1.3f, %1.3f' % (rf_train_acc, rf_val_acc))

        rf_prob = clf.predict_proba(x_trainval)
        newdata = pd.DataFrame(rf_prob, columns = ['prob_' + str(i) for i in range(num_classes)])
        newdata['target'] = y_trainval
        dataset_train = newdata.iloc[train_index, :], images_trainval[train_index, :, :]
        dataset_val =  newdata.iloc[val_index, :], images_trainval[val_index, : ,:]


        train_dataset = SortDataset(dataset_train)
        val_dataset = SortDataset(dataset_val)

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Instantiate the model
        model = FashionNet(input_size=train_dataset.x1.size(2)).to(device)
           
        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        num_epochs = 20
        best_acc = 0.0
        for epoch in range(num_epochs):
            running_loss = 0.0
            running_corrects = 0
    
            # Training
            model.train()
            for inputs1, inputs2, labels in train_loader:
                inputs1 = inputs1.to(device)
                inputs2 = inputs2.to(device)
                labels = labels.to(device)
        
                optimizer.zero_grad()
        
                # Forward pass
                outputs = model(inputs1, inputs2)
                loss = criterion(outputs, labels)
        
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
        
                running_loss += loss.item() * inputs1.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
    

            # Print the average training loss and accuracy for this epoch
            train_loss = running_loss / len(train_loader.dataset)
            train_acc = running_corrects.double() / len(train_loader.dataset)

            # Evaluation
            model.eval()
            num_correct = 0
            num_samples = 0

            with torch.no_grad():
                for inputs1, inputs2, labels in val_loader:
                    inputs1 = inputs1.to(device)
                    inputs2 = inputs2.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs1, inputs2)
                    _, preds = torch.max(outputs, 1)

                    num_correct += torch.sum(preds == labels.data)
                    num_samples += labels.size(0)

            val_acc = num_correct.double() / num_samples

            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.3f}, Train Accuracy: {train_acc:1.3f}, Validation Accuracy: {val_acc:1.3f}")

            if val_acc > best_acc:
                #torch.save(model, 'best-model.pt')
                torch.save(model.state_dict(), 'df-best-model-parameters.pt')
                best_acc=val_acc
                print('Saving new best model')
    
    rf_prob_test = clf.predict_proba(x_test)
    rf_test_acc += clf.score(x_test, y_test)
    rf_test_prec_w += precision_score(y_test, clf.predict(x_test), average='weighted', zero_division=0)
    rf_test_rec_w += recall_score(y_test, clf.predict(x_test), average='weighted', zero_division=0)
    rf_test_f1_w += fbeta_score(y_test, clf.predict(x_test), average='weighted', beta=1)
    rf_test_f2_w += fbeta_score(y_test, clf.predict(x_test), average='weighted', beta=2)

    newdata_test = pd.DataFrame(rf_prob_test, columns = ['prob_' + str(i) for i in range(num_classes)])
    newdata_test['target'] = y_test
    dataset_test = newdata_test, images_test


    test_dataset = SortDataset(dataset_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    final_model = FashionNet(input_size=test_dataset.x1.size(2)).to(device)
    final_model.load_state_dict(torch.load('df-best-model-parameters.pt'))

    num_samples = len(y_test)
    _, class_frequencies = np.unique(y_test, return_counts=True)

    weighted_true_positives = np.zeros(num_classes)
    weighted_false_positives = np.zeros(num_classes)
    weighted_false_negatives = np.zeros(num_classes)

    final_model.eval()
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for inputs1, inputs2, labels in test_loader:
            inputs1 = inputs1.to(device)
            inputs2 = inputs2.to(device)
            labels = labels.to(device)

            outputs = final_model(inputs1, inputs2)
            _, preds = torch.max(outputs, 1)

            for i in range(num_classes):
                true_positives = torch.sum((preds == i) & (labels == i)).item()
                false_positives = torch.sum((preds == i) & (labels != i)).item()
                false_negatives = torch.sum((preds != i) & (labels == i)).item()

                weighted_true_positives[i] += true_positives
                weighted_false_positives[i] += false_positives
                weighted_false_negatives[i] += false_negatives

            num_correct += torch.sum(preds == labels.data)
            num_samples += labels.size(0)

    beta = 2
    weighted_precision = sum(weighted_true_positives[i] / (weighted_true_positives[i] + weighted_false_positives[i]) * (class_frequencies[i] / num_samples) if weighted_true_positives[i] + weighted_false_positives[i] != 0 else 0 for i in range(num_classes))
    weighted_recall = sum(weighted_true_positives[i] / (weighted_true_positives[i] + weighted_false_negatives[i]) * (class_frequencies[i] / num_samples) if weighted_true_positives[i] + weighted_false_negatives[i] != 0 else 0 for i in range(num_classes))
    weighted_f1 = sum((2 * weighted_true_positives[i] / (2*weighted_true_positives[i] + weighted_false_positives[i] + weighted_false_negatives[i])) * (class_frequencies[i] / num_samples) for i in range(num_classes))
    weighted_f2 = sum((1 + beta**2) * weighted_true_positives[i] / ((1 + beta**2)*weighted_true_positives[i] + weighted_false_positives[i] + (beta**2)*weighted_false_negatives[i]) * (class_frequencies[i] / num_samples) for i in range(num_classes))    

    accuracy = num_correct.double() / num_samples

    nn_test_acc += accuracy
    nn_test_prec_w += weighted_precision
    nn_test_rec_w += weighted_recall
    nn_test_f1_w += weighted_f1
    nn_test_f2_w += weighted_f2

    del x_test, y_test, images_test, x_trainval, y_trainval, x_train, y_train, x_val, y_val
    del dataset_train, dataset_val, train_dataset, val_dataset, dataset_test, test_dataset
    del train_loader, val_loader, test_loader, model, final_model
    del newdata, newdata_test, images_trainval
    gc.collect()
    
    k += 1

rf_test_acc = rf_test_acc/n_folds
rf_test_prec_w = rf_test_prec_w/n_folds
rf_test_rec_w = rf_test_rec_w/n_folds
rf_test_f1_w = rf_test_f1_w/n_folds
rf_test_f2_w = rf_test_f2_w/n_folds

nn_test_acc = nn_test_acc/n_folds
nn_test_prec_w = nn_test_prec_w/n_folds
nn_test_rec_w = nn_test_rec_w/n_folds
nn_test_f1_w = nn_test_f1_w/n_folds
nn_test_f2_w = nn_test_f2_w/n_folds

print('\n\nRandom forest testing metrics (weighted):\n Accuracy = %1.3f\n Precision = %1.3f\n Recall = %1.3f\n F1 = %1.3f\n F2 = %1.3f\n' % (rf_test_acc, rf_test_prec_w, rf_test_rec_w, rf_test_f1_w, rf_test_f2_w))
print('Neural Network testing metrics (weighted):\n Accuracy = %1.3f\n Precision = %1.3f\n Recall = %1.3f\n F1 = %1.3f\n F2 = %1.3f\n' % (nn_test_acc, nn_test_prec_w, nn_test_rec_w, nn_test_f1_w, nn_test_f2_w))

