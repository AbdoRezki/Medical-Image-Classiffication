import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight


class ChestXrayDataLoader:   
    def __init__(self, data_dir, img_size=(224, 224), batch_size=32):

        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        
        # Define data directories
        self.train_dir = os.path.join(data_dir, 'train')
        self.val_dir = os.path.join(data_dir, 'val')
        self.test_dir = os.path.join(data_dir, 'test')
        
        self._validate_directories()
    
    def _validate_directories(self):
        for dir_path in [self.train_dir, self.val_dir, self.test_dir]:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Directory not found: {dir_path}")
    
    def create_generators(self, augmentation=True):
        if augmentation:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)
        
        # Validation and test data (no augmentation, only rescaling)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=True,
            seed=42
        )
        
        val_generator = test_datagen.flow_from_directory(
            self.val_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        test_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        return train_generator, val_generator, test_generator
    
    def compute_class_weights(self):
        train_generator, _, _ = self.create_generators(augmentation=False)
        
        # Get class labels
        class_labels = train_generator.classes
        
        # Compute weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(class_labels),
            y=class_labels
        )
        
        # Convert to dictionary
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        
        print(f"Class weights: {class_weight_dict}")
        return class_weight_dict
    
    def get_dataset_info(self):
        train_generator, val_generator, test_generator = self.create_generators(augmentation=False)
        
        info = {
            'train_samples': train_generator.samples,
            'val_samples': val_generator.samples,
            'test_samples': test_generator.samples,
            'num_classes': train_generator.num_classes,
            'class_names': list(train_generator.class_indices.keys()),
            'class_distribution_train': {
                class_name: np.sum(train_generator.classes == class_idx)
                for class_name, class_idx in train_generator.class_indices.items()
            }
        }
        
        return info


def load_and_preprocess_image(image_path, img_size=(224, 224)):
    # Load image
    img = tf.keras.preprocessing.image.load_img(
        image_path,
        target_size=img_size
    )
    
    # Convert to array
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    # Rescale
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


if __name__ == "__main__":
    # Test the data loader
    data_dir = "../data/chest_xray"
    
    if os.path.exists(data_dir):
        loader = ChestXrayDataLoader(data_dir)
        
        # Print dataset info
        info = loader.get_dataset_info()
        print("\n=== Dataset Information ===")
        for key, value in info.items():
            print(f"{key}: {value}")
        
        # Compute class weights
        print("\n=== Computing Class Weights ===")
        class_weights = loader.compute_class_weights()
        
        # Create generators
        print("\n=== Creating Data Generators ===")
        train_gen, val_gen, test_gen = loader.create_generators()
        print("Data generators created successfully!")
    else:
        print(f"Data directory not found: {data_dir}")
        print("Please download the dataset first.")