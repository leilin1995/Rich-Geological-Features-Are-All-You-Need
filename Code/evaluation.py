"""
__author__: Lei Lin
__project__: evaluation.py
__time__: 2024/3/27 
__email__: leilin1117@outlook.com
"""

import numpy as np
import os
import h5py
import argparse
import time

parser = argparse.ArgumentParser(description="Unet3D for fault segmentation in evaluation")
parser.add_argument("--save_dir", default="../Results/Unet3D", type=str,
                    help="directory to save the files in evaluation")
parser.add_argument("--save_name", default="evaluated_metrics.txt", type=str,
                    help="directory to save the files in evaluation")
parser.add_argument("--label_dir", default="../Dataset/Test", type=str, help="directory of label files")
parser.add_argument("--pred_dir", default="../Results/Unet3D/predictTest", type=str,
                    help="directory of predicted files")
parser.add_argument("--pred_internal_path", default="predict", type=str,
                    help="Internal path of .hdf5 of predicted data")
parser.add_argument("--label_internal_path", default="label", type=str,
                    help="Internal path of .hdf5 of target label data")
parser.add_argument("--classes", default=2, type=int,
                    help="number of classes")
parser.add_argument("--threshold", default=0.5, type=float,
                    help="threshold of label segmentation,target pred > threshold,background label < threshold")


def main():
    args = parser.parse_args()
    save_dir = args.save_dir
    print(f"Saving directory is {save_dir}.")
    label_folder = args.label_dir
    pred_folder = args.pred_dir
    label_internal_path = args.label_internal_path
    pred_internal_path = args.pred_internal_path
    num_classes = args.classes
    threshold = args.threshold

    label_list = os.listdir(label_folder)
    pred_list = os.listdir(pred_folder)
    assert len(label_list) == len(
        pred_list), "The number of real images is not equal to the number of predicted images"
    confusion_matrix = ConfusionMatrix(num_classes=num_classes, threshold=threshold)
    num_list = len(label_list)
    # num_list = tqdm(num_list)
    start = time.time()
    for i in range(num_list):
        print(f"Evaluate '{pred_list[i]}'...")
        # read data
        with h5py.File(os.path.join(pred_folder, pred_list[i]), mode="r") as f:
            pred = f[pred_internal_path][:]
        # if pred dim more than 3,get target probability map
        if len(pred.shape) > 3:
            if pred.shape[0] == 1:
                pred = np.squeeze(pred, axis=0)
            elif pred.shape[0] == 2:
                pred = pred[1,]  # channel 0: background,channel 1: target
            else:
                raise Exception(
                    "Dimension error! The desired number of channels for binary segmentation is 1 or 2.")
        with h5py.File(os.path.join(label_folder, label_list[i]), mode="r") as f:
            label = f[label_internal_path][:]
        confusion_matrix.update(pred, label)
    evaluator = Evaluator(confusion_matrix.matrix)
    Pixel_Accuracy = evaluator.Pixel_Accuracy()
    Mean_Pixel_Accuracy = evaluator.Mean_Pixel_Accuracy()
    IOU = evaluator.IOU()
    Dice = evaluator.Dice()
    Precision_rate = evaluator.Precision_rate()
    Recall_rate = evaluator.Recall_rate()
    F1 = evaluator.F1_score()
    print(f'Finished evaluation in {time.time() - start:.2f} seconds')
    print(

        "Pixel_Accuracy:{:.3f}\nMean_Pixel_Accuracy:{:.3f}\nPrecision_rate:{:.3f}\nRecall_rate:{:.3f}\nIOU:{:.3f}\nDice:{:.3f}\nF1_score:{:.3f}".
        format(Pixel_Accuracy, Mean_Pixel_Accuracy, Precision_rate, Recall_rate, IOU, Dice, F1))
    save_path = os.path.join(save_dir, args.save_name)
    with open(save_path, "w") as f:
        f.write(
            "Pixel_Accuracy:{:.3f}\nMean_Pixel_Accuracy:{:.3f}\nPrecision_rate:{:.3f}\nRecall_rate:{:.3f}\nIOU:{:.3f}\nDice:{:.3f}\nF1_score:{:.3f}".
            format(Pixel_Accuracy, Mean_Pixel_Accuracy, Precision_rate, Recall_rate, IOU, Dice, F1))
        f.close()


"""here are six semantic segmentation evaluation index,whice are calculated by confusion matrix
    PA:
        PA is pixel accuracy
        PA = TP + TN / TP +TN + FP + FN

    MPA:
        Calculate the proportion between the correct pixel number of each category and all pixel points of this category, and then average it
        MPA = (TP / (TP + FN) + TN / (TN + FP)) / 2

    PR:
        PR is precision rate
        PR = TP / TP + FP

    RE:
        RE is recall rate
        RE = TP / TP + FN

    Dice:
        Dice coefficient is a measure of set similarity
        Dice = 2TP / FP + 2TP + FN

    IOU:
        Intersection over Union
        IOU = TP / FP + TP + FN

"""


class ConfusionMatrix():
    def __init__(self, num_classes=2, threshold=0.5):
        """
        caulate confusion matrix
        Args:
            num_classes: category.
            threshold: threshold of label segmentation,target pred > threshold,background label < threshold
        """

        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.threshold = threshold

    def update(self, preds, labels):
        """

        Args:
            preds: Prediction probability map
            labels: Fault label map

        Returns:

        """
        preds[preds > self.threshold] = 1
        preds[preds <= self.threshold] = 0
        labels = labels.flatten()
        preds = preds.flatten()
        for p, t in zip(preds, labels):
            self.matrix[int(t), int(p)] += 1

    def summary(self):
        # ACC=Accuracy，PR=Precision，RE=Recall
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]

        ACC = sum_TP / np.sum(self.matrix)
        TP = self.matrix[1, 1]
        FP = self.matrix[0, 1]
        FN = self.matrix[1, 0]

        PR = TP / (TP + FP)
        RE = TP / (TP + FN)
        return ACC, PR, RE


class Evaluator():
    def __init__(self, confusion_matrix, num_classes=2):
        self.num_classes = num_classes
        self.matrix = confusion_matrix

    def Pixel_Accuracy(self):
        PA = np.sum(np.diag(self.matrix)) / np.sum(self.matrix)
        return PA

    def Mean_Pixel_Accuracy(self):
        MPA_array = np.diag(self.matrix) / np.sum(self.matrix, axis=1)
        MPA = 0.2 * MPA_array[0] + 0.8 * MPA_array[1]
        return MPA

    def IOU(self):
        iou = self.matrix[1, 1] / (self.matrix[1, 0] + self.matrix[1, 1] + self.matrix[0, 1])
        return iou

    def Dice(self):
        dice = 2 * self.matrix[1, 1] / (self.matrix[1, 0] + 2 * self.matrix[1, 1] + self.matrix[0, 1])
        return dice

    def Precision_rate(self):
        pr = self.matrix[1, 1] / (self.matrix[1, 1] + self.matrix[0, 1])
        return pr

    def Recall_rate(self):
        re = self.matrix[1, 1] / (self.matrix[1, 1] + self.matrix[1, 0])
        return re

    def F1_score(self):
        pr = self.matrix[1, 1] / (self.matrix[1, 1] + self.matrix[0, 1])
        re = self.matrix[1, 1] / (self.matrix[1, 1] + self.matrix[1, 0])
        F1 = 2 * (pr * re) / (pr + re)
        return F1


if __name__ == '__main__':
    main()
