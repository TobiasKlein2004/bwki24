import torch
import torch_directml

import torch.nn as nn

from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

device_name = torch_directml.device_name(0)
device = torch.device(torch_directml.device(0))




def testModel(model, dataloader, NUM_CLASSES, criterion, dataset):
    print("Testing...")

    preds = []
    actual = []

    tot_loss = tot_acc = count = 0

    for images, labels in tqdm(dataloader):
        with torch.set_grad_enabled(False):
            output = model(images.to(device))
            ohe_label = nn.functional.one_hot(labels, num_classes=NUM_CLASSES)
            out_labels = torch.argmax(output, dim=1)


            tot_loss += criterion(output, ohe_label.float().to(device))
            tot_acc += (labels.to(device) == out_labels).sum()/len(labels)
            count += 1

        preds += out_labels.tolist()
        actual += labels.tolist()

    print(f"Test Loss: {tot_loss / count}, Test Accuracy: {tot_acc / count}")

    class_labels = sorted(dataset.class_lbl.keys())
    print(class_labels)

    cm = confusion_matrix(actual, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)

    disp.plot()
    plt.show()

    cm_np = np.array(cm)
    stats = pd.DataFrame(index=class_labels)
    stats['Precision'] = [cm_np[i, i]/np.sum(cm_np[:, i]) for i in range(len(cm_np))]
    stats['Recall'] = [cm_np[i, i]/np.sum(cm_np[i, :]) for i in range(len(cm_np))]
    stats