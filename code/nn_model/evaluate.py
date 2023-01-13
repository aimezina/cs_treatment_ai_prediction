import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve, roc_auc_score, \
    accuracy_score, precision_score, recall_score, f1_score, multilabel_confusion_matrix

from data_processing import prepare_dataset
from code.params import IMG_SIZE


def evaluate(model):
    train_df, test_df, all_labels = prepare_dataset()
    weight_path = "outputs/{}_weights.best.hdf5".format('xray_class')

    test_core_idg = ImageDataGenerator(
        rescale=1. / 255
    )

    test_X, test_Y = next(test_core_idg.flow_from_dataframe(
        dataframe=test_df,
        directory=None,
        x_col='path',
        y_col='newLabel',
        class_mode='categorical',
        classes=all_labels,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='rgb',
        batch_size=-1)
    )

    # load the best weights
    model.load_weights(weight_path)
    pred_Y = model.predict(test_X, batch_size=1, verbose=True)
    print(pred_Y)

    for c_label, p_count, t_count in zip(all_labels,
                                         100 * np.mean(pred_Y, 0),
                                         100 * np.mean(test_Y, 0)):
        print('%s: Dx: %2.2f%%, PDx: %2.2f%%' % (c_label, t_count, p_count))

    from statistics import mean
    auc_rocs, thresholds, sensitivity, specificity, accuracy, precision, recall, f1 = get_roc_curve(all_labels,
                                                                                                    pred_Y,
                                                                                                    test_Y)
    from tabulate import tabulate
    table = zip(all_labels, auc_rocs)
    print(f"Mean AUC : {mean(auc_rocs)}")
    print(tabulate(table, headers=['Pathology', 'AUC'], tablefmt='fancy_grid'))

    from tabulate import tabulate
    table = zip(all_labels, auc_rocs, thresholds, sensitivity, specificity, accuracy, precision, recall, f1)
    print(tabulate(table, headers=['Pathology', 'AUC', 'Threshold Value', 'Sensitivity', 'Specificity', 'Accuracy',
                                   'Precision', 'Recall', 'F1 Score'], tablefmt='fancy_grid'))

    original_stdout = sys.stdout
    table = zip(all_labels, auc_rocs, thresholds, sensitivity, specificity, accuracy, precision, recall, f1)
    with open(f'outputs/results_test.txt', 'w', encoding="utf-8") as f:
        sys.stdout = f
        print(f"Mean AUC : {mean(auc_rocs)}")
        print(f"Mean sensitivity : {mean(sensitivity)}")
        print(f"Mean specificity : {mean(specificity)}")
        print(f"Mean accuracy : {mean(accuracy)}")
        print(f"Mean precision : {mean(precision)}")
        print(f"Mean recall : {mean(recall)}")
        print(f"Mean f1 : {mean(f1)}")

        print(tabulate(table, headers=['Pathology', 'AUC', 'Threshold Value', 'Sensitivity', 'Specificity', 'Accuracy',
                                       'Precision', 'Recall', 'F1 Score'], tablefmt='fancy_grid'))
        sys.stdout = original_stdout


def get_roc_curve(labels, predicted_vals, test_Y):
    auc_roc_vals = []
    optimal_thresholds = []
    sensitivity = []
    specificity = []
    accuracy = []
    precision = []
    recall = []
    f1 = []
    for i in range(len(labels)):
        try:
            gt = test_Y[:, i]
            pred = predicted_vals[:, i]
            auc_roc = roc_auc_score(gt, pred)  # return
            auc_roc_vals.append(auc_roc)
            fpr, tpr, thresholds = roc_curve(gt, pred)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            y_pred = pred > optimal_threshold
            acc = accuracy_score(gt, y_pred)
            prec = precision_score(gt, y_pred)
            rec = recall_score(gt, y_pred)
            f1_s = f1_score(gt, y_pred)
            accuracy.append(acc)
            precision.append(prec)
            recall.append(rec)
            f1.append(f1_s)
            optimal_thresholds.append(
                optimal_threshold)
            optimal_tpr = round(tpr[optimal_idx], 3)
            sensitivity.append(optimal_tpr)
            specificity.append(1 - fpr[optimal_idx])
            plt.figure(1, figsize=(10, 10))
            plt.plot([0, 1], [0, 1], 'k--')  # black dash line
            plt.plot(fpr, tpr,
                     label=labels[i] + " (" + str(round(auc_roc, 3)) + ")")
            plt.xlabel('False positive rate', fontsize=16)
            plt.tick_params(axis='x', labelsize=16)
            plt.ylabel('True positive rate', fontsize=16)
            plt.tick_params(axis='y', labelsize=16)
            plt.title('ROC curve').set_fontsize(16)
            plt.legend(loc='best', fontsize=16)

            plt.savefig('outputs/roc_auc.png')

            cm = multilabel_confusion_matrix(test_Y, y_pred)
            cm_df = pd.DataFrame(cm)
            plt.figure(figsize=(12, 10))
            plt.title('Confusion Matrix')
            seaborn.heatmap(cm_df, annot=True, cmap='Blues', square=True)
        except:
            print(
                f"Error in generating ROC curve for {labels[i]}. "
                f"Dataset lacks enough examples."
            )
    plt.show()
    return auc_roc_vals, optimal_thresholds, sensitivity, specificity, accuracy, precision, recall, f1
