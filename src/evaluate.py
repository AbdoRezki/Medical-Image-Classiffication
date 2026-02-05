import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)
import tensorflow as tf

from data_loader import ChestXrayDataLoader


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()


def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """
    Plot ROC curve
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to: {save_path}")
    
    plt.show()


def plot_precision_recall_curve(y_true, y_pred_proba, save_path=None):
    """
    Plot Precision-Recall curve
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save plot
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Precision-Recall curve saved to: {save_path}")
    
    plt.show()


def evaluate_model(args):
    """
    Evaluate trained model on test set
    
    Args:
        args: Command-line arguments
    """
    print("=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    
    # Load model
    print(f"\n[1/4] Loading model from {args.model_path}...")
    model = tf.keras.models.load_model(args.model_path)
    print(f"Model loaded: {model.name}")
    
    # Load test data
    print(f"\n[2/4] Loading test data from {args.data_dir}...")
    data_loader = ChestXrayDataLoader(
        data_dir=args.data_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size
    )
    
    _, _, test_gen = data_loader.create_generators(augmentation=False)
    
    # Get predictions
    print(f"\n[3/4] Making predictions on test set...")
    y_pred_proba = model.predict(test_gen, verbose=1)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    y_true = test_gen.classes
    
    # Get class names
    class_names = list(test_gen.class_indices.keys())
    
    # Print classification report
    print(f"\n[4/4] Generating evaluation metrics...")
    print("\n" + "=" * 80)
    print("CLASSIFICATION REPORT")
    print("=" * 80)
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Calculate additional metrics
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn)  # Recall for pneumonia
    specificity = tn / (tn + fp)  # Recall for normal
    
    print("=" * 80)
    print("ADDITIONAL METRICS")
    print("=" * 80)
    print(f"Sensitivity (Recall for Pneumonia): {sensitivity:.4f}")
    print(f"Specificity (Recall for Normal):    {specificity:.4f}")
    print(f"\nTrue Negatives:  {tn:4d}")
    print(f"False Positives: {fp:4d}")
    print(f"False Negatives: {fn:4d}")
    print(f"True Positives:  {tp:4d}")
    
    # Generate and save plots
    if args.save_plots:
        print("\n" + "=" * 80)
        print("GENERATING PLOTS")
        print("=" * 80)
        
        output_dir = os.path.dirname(args.model_path)
        
        # Confusion matrix
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
        
        # ROC curve
        roc_path = os.path.join(output_dir, 'roc_curve.png')
        plot_roc_curve(y_true, y_pred_proba, roc_path)
        
        # Precision-Recall curve
        pr_path = os.path.join(output_dir, 'precision_recall_curve.png')
        plot_precision_recall_curve(y_true, y_pred_proba, pr_path)
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate pneumonia detection model'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default='../models/best_resnet50_model.h5',
        help='Path to trained model'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../data/chest_xray',
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--img-size',
        type=int,
        default=224,
        help='Image size (assumes square images)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--save-plots',
        action='store_true',
        default=True,
        help='Save evaluation plots'
    )
    
    args = parser.parse_args()
    
    # Evaluate model
    evaluate_model(args)


if __name__ == "__main__":
    main()