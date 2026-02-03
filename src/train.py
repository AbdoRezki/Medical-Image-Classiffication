import os
import argparse
import json
from datetime import datetime
import tensorflow as tf

from data_loader import ChestXrayDataLoader
from model import create_model, create_custom_cnn, get_callbacks


def train_model(args):
    print("=" * 80)
    print("PNEUMONIA DETECTION MODEL TRAINING")
    print("=" * 80)
    
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    
    # Create models directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Initialize data loader
    print(f"\n[1/6] Loading data from {args.data_dir}...")
    data_loader = ChestXrayDataLoader(
        data_dir=args.data_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size
    )
    
    # Get dataset info
    dataset_info = data_loader.get_dataset_info()
    print("\nDataset Information:")
    for key, value in dataset_info.items():
        print(f"  {key}: {value}")
    
    # Create data generators
    print(f"\n[2/6] Creating data generators...")
    train_gen, val_gen, test_gen = data_loader.create_generators(
        augmentation=args.augmentation
    )
    
    # Compute class weights if needed
    class_weights = None
    if args.use_class_weights:
        print(f"\n[3/6] Computing class weights...")
        class_weights = data_loader.compute_class_weights()
    else:
        print(f"\n[3/6] Not using class weights")
    
    # Create model
    print(f"\n[4/6] Creating {args.model} model...")
    if args.model == 'custom':
        model = create_custom_cnn(img_size=(args.img_size, args.img_size))
    else:
        model = create_model(
            model_name=args.model,
            img_size=(args.img_size, args.img_size),
            trainable_base=args.trainable_base
        )
    
    print(f"\nModel: {model.name}")
    print(f"Total parameters: {model.count_params():,}")
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create callbacks
    model_path = os.path.join(args.model_dir, f'best_{args.model}_model.h5')
    callbacks = get_callbacks(model_path=model_path)
    
    # Train model
    print(f"\n[5/6] Training model for {args.epochs} epochs...")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {model.optimizer.learning_rate.numpy()}")
    print("-" * 80)
    
    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print(f"\n[6/6] Evaluating on test set...")
    test_loss, test_acc, test_precision, test_recall, test_auc = model.evaluate(
        test_gen,
        verbose=1
    )
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall:    {test_recall:.4f}")
    print(f"Test AUC:       {test_auc:.4f}")
    print(f"Test Loss:      {test_loss:.4f}")
    
    # Save training history
    history_path = os.path.join(args.model_dir, f'{args.model}_history.json')
    import numpy as np
    
    def convert_to_python_type(obj):
        """Convert numpy types to Python types"""
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    with open(history_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        history_dict = {
            key: [convert_to_python_type(val) for val in values]
            for key, values in history.history.items()
        }
        json.dump(history_dict, f, indent=2)
    print(f"\nTraining history saved to: {history_path}")
    
    # Save model info
    import numpy as np
    
    def serialize_value(v):
        """Recursively convert all values to JSON-serializable types"""
        if isinstance(v, (np.integer, np.int32, np.int64)):
            return int(v)
        elif isinstance(v, (np.floating, np.float32, np.float64)):
            return float(v)
        elif isinstance(v, dict):
            return {k: serialize_value(val) for k, val in v.items()}
        elif isinstance(v, (list, tuple)):
            return [serialize_value(item) for item in v]
        else:
            return v
    
    model_info = {
        'model_name': args.model,
        'img_size': int(args.img_size),
        'batch_size': int(args.batch_size),
        'epochs_trained': int(len(history.history['loss'])),
        'test_accuracy': float(test_acc),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_auc': float(test_auc),
        'test_loss': float(test_loss),
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_info': serialize_value(dataset_info)
    }
    
    info_path = os.path.join(args.model_dir, f'{args.model}_info.json')
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"Model info saved to: {info_path}")
    
    print(f"\nBest model saved to: {model_path}")
    print("=" * 80)
    
    return model, history


def main():
    parser = argparse.ArgumentParser(
        description='Train pneumonia detection model'
    )
    
    # Data parameters
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../data/chest_xray',
        help='Path to dataset directory'
    )
    
    # Model parameters
    parser.add_argument(
        '--model',
        type=str,
        default='resnet50',
        choices=['resnet50', 'efficientnetb0', 'vgg16', 'custom'],
        help='Model architecture to use'
    )
    parser.add_argument(
        '--img-size',
        type=int,
        default=224,
        help='Image size (assumes square images)'
    )
    parser.add_argument(
        '--trainable-base',
        action='store_true',
        help='Make base model layers trainable (fine-tuning)'
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of epochs to train'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '--augmentation',
        action='store_true',
        default=True,
        help='Use data augmentation'
    )
    parser.add_argument(
        '--no-augmentation',
        action='store_false',
        dest='augmentation',
        help='Disable data augmentation'
    )
    parser.add_argument(
        '--use-class-weights',
        action='store_true',
        default=True,
        help='Use class weights to handle imbalance'
    )
    parser.add_argument(
        '--no-class-weights',
        action='store_false',
        dest='use_class_weights',
        help='Disable class weights'
    )
    
    # Output parameters
    parser.add_argument(
        '--model-dir',
        type=str,
        default='../models',
        help='Directory to save models'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='../logs',
        help='Directory for TensorBoard logs'
    )
    
    args = parser.parse_args()
    
    # Train model
    model, history = train_model(args)


if __name__ == "__main__":
    main()