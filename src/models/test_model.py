# -*- coding: utf-8 -*-
import sys
sys.path.append("../src")
sys.path.append("../src/data/")

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import tensorflow as tf
from src.data.dataloader import *
from tqdm import tqdm
from config import *
from utils import *
from loss import *
import numpy as np
import logging
import os

def iou_score(pred, orig):
    """Compute IoU for binary masks."""
    intersection = np.logical_and(pred, orig).sum()
    union = np.logical_or(pred, orig).sum()
    return intersection / union if union != 0 else 1.0

def dice_coef(pred, orig):
    """Compute Dice Similarity Coefficient for binary masks."""
    intersection = np.logical_and(pred, orig).sum()
    total = pred.sum() + orig.sum()
    return 2 * intersection / total if total != 0 else 1.0

def main():
    """Testing performance of ACLNet model with per-image and global metrics."""
    
    # Load data
    trainGen, testGen = getDataLoader(batch_size=1)

    # Load model
    model = tf.keras.models.load_model(
        os.path.join(WEIGHTS_DIR, 'ACLNet_Best.h5'), 
        custom_objects={"diceCoef": diceCoef, "bceDiceLoss": bceDiceLoss}
    )

    # Evaluate using Keras built-in metrics
    model.evaluate(testGen)

    # Collect predictions and ground truth
    original, prediction = [], []

    with tqdm(total=int(6468 * TEST_SIZE)) as pbar:
        for data in testGen:
            image, mask = data
            seg = model.predict(image)
            original.append(mask[0].argmax(-1))
            prediction.append(seg[0].argmax(-1))
            pbar.update(1)

    original = np.array(original)
    prediction = np.array(prediction)

    # Lists for per-image metrics
    precisions, recalls, f1scores, error_rates, ious, dscs = [], [], [], [], [], []

    # Arrays for global metrics
    orig_all, pred_all = [], []

    with tqdm(total=len(original)) as pbar:
        for orig, pred in zip(original, prediction):
            try:
                precision, recall, f1score, error_rate = score_card(pred, orig)

                # Convert to binary masks if needed
                orig_bin = orig > 0
                pred_bin = pred > 0

                iou = iou_score(pred_bin, orig_bin)
                dsc = dice_coef(pred_bin, orig_bin)

                precisions.append(precision)
                recalls.append(recall)
                f1scores.append(f1score)
                error_rates.append(error_rate)
                ious.append(iou)
                dscs.append(dsc)
            except:
                print('skipped')


    # Logging per-image metrics
    logging.info("[Info] Mean Precision: {}".format(np.mean(precisions)))
    logging.info("[Info] Mean Recall: {}".format(np.mean(recalls)))
    logging.info("[Info] Mean F1-Score: {}".format(np.mean(f1scores)))
    logging.info("[Info] Mean Error Rate: {}".format(np.mean(error_rates)))
    logging.info("[Info] Mean IoU: {}".format(np.mean(ious)))
    logging.info("[Info] Mean DSC: {}".format(np.mean(dscs)))

    # ---------------------------------------------------------------------
    # Save box plots of precision, recall, F1-score, error rate, IoU and DSC
    # ---------------------------------------------------------------------
    metrics_dict = {
        "Precision": precisions,
        "Recall": recalls,
        "F1-Score": f1scores,
        "Error Rate": error_rates,
        "IoU": ious,
        "DSC": dscs,
    }

    for metric_name, values in metrics_dict.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(values)
        ax.set_title(f"{metric_name} Distribution")
        ax.set_ylabel(metric_name)
        ax.set_xticks([1])
        ax.set_xticklabels([metric_name])
        plt.tight_layout()
        plt.savefig(os.path.join(INFERENCE_DIR, f"{metric_name}_Boxplot.pdf"))
        plt.close(fig)

    # ---------------------------------------------------------------------
    # Save the true and predicted mask next to each other for random images
    # ---------------------------------------------------------------------
    import random

    n_samples = 5  # number of random test samples to visualize
    rand_indices = random.sample(range(len(original)), n_samples)

    for i, idx in enumerate(rand_indices):
        orig_mask = original[idx]
        pred_mask = prediction[idx]

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(orig_mask, cmap="gray")
        axes[0].set_title("Ground Truth")
        axes[0].axis("off")

        axes[1].imshow(pred_mask, cmap="gray")
        axes[1].set_title("Prediction")
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(INFERENCE_DIR, f"mask_comparison_{i}.pdf"))
        plt.close(fig)



    # Matthews Correlation Coefficient
    logging.info("[Info] Matthews Correlation Coefficient (MCC): ")
    logging.info(matthews_corrcoef(orig_all, pred_all))

    # ROC-AUC Curve
    logging.info("[Info] ROC_AUC Curve: ")
    fpr, tpr, thresholds = roc_curve(orig_all, pred_all)
    auc_score = auc(fpr, tpr)

    fig, ax = plt.subplots(1,1)
    ax.plot(fpr, tpr, label='ROC curve ACLNet (area = %0.4f)' % auc_score)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    plt.savefig(os.path.join(INFERENCE_DIR, "ROC_Curve_ACLNet.pdf"))

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO, 
        filename=os.path.join(LOG_DIR, 'app.log'), 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
        filemode='w'
    )

    sys.stdout = LoggerWriter(logging.info)
    sys.stderr = LoggerWriter(logging.error)

    main()
