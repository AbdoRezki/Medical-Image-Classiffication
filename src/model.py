import tensorflow as tf
from keras import layers, models
from keras.applications import ResNet50, EfficientNetB0, VGG16


def create_model(model_name='resnet50', img_size=(224, 224), trainable_base=False):
    input_shape = (*img_size, 3)
    
    # Select base model
    if model_name.lower() == 'resnet50':
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    elif model_name.lower() == 'efficientnetb0':
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    elif model_name.lower() == 'vgg16':
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Freeze or unfreeze base model
    base_model.trainable = trainable_base
    
    # Build model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.6),  # Increased from 0.5
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),  # Increased from 0.3
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ], name=f'{model_name}_pneumonia_detector')
    
    # Compile model with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),  # Lower from 0.0001
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    return model


def create_custom_cnn(img_size=(224, 224)):
    input_shape = (*img_size, 3)
    
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ], name='custom_cnn_pneumonia_detector')
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    return model


def get_callbacks(model_path='../models/best_model.h5'):
    callbacks = [
        # Save best model
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir='../logs',
            histogram_freq=1
        )
    ]
    
    return callbacks


if __name__ == "__main__":
    # Test model creation
    print("=== Testing Model Creation ===\n")
    
    # Test ResNet50
    print("Creating ResNet50 model...")
    model_resnet = create_model('resnet50')
    print(f"Total parameters: {model_resnet.count_params():,}")
    print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model_resnet.trainable_weights]):,}\n")
    
    # Test EfficientNetB0
    print("Creating EfficientNetB0 model...")
    model_efficient = create_model('efficientnetb0')
    print(f"Total parameters: {model_efficient.count_params():,}")
    print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model_efficient.trainable_weights]):,}\n")
    
    # Test custom CNN
    print("Creating custom CNN model...")
    model_custom = create_custom_cnn()
    print(f"Total parameters: {model_custom.count_params():,}")
    print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model_custom.trainable_weights]):,}\n")
    
    # Print model summary
    print("=== ResNet50 Model Summary ===")
    model_resnet.summary()