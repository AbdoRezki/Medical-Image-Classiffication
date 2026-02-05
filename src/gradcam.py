import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from keras.models import Model


class GradCAM:
    def __init__(self, model, layer_name=None):
        self.model = model
        self.layer_name = layer_name or self._find_target_layer()
        
        # Create gradient model
        self.grad_model = self._build_grad_model()
    
    def _find_target_layer(self):
        # Common last conv layer names for popular architectures
        target_layers = [
            'conv5_block3_out',    # ResNet50
            'top_conv',            # EfficientNet
            'block7a_project_conv', # EfficientNet alternative
            'conv2d',              # Custom CNN
        ]
        
        # First, check if model has nested layers (transfer learning)
        base_model = None
        for layer in self.model.layers:
            if hasattr(layer, 'layers') and len(layer.layers) > 10:
                # This is likely the base model (ResNet, EfficientNet, etc.)
                base_model = layer
                break
        
        if base_model is not None:
            # Try to find target layer in base model
            for target_name in target_layers:
                try:
                    base_model.get_layer(target_name)
                    print(f"Using layer: {target_name}")
                    return target_name
                except:
                    continue
            
            # If not found, use the last conv layer in base model
            for layer in reversed(base_model.layers):
                if 'conv' in layer.__class__.__name__.lower():
                    print(f"Using layer: {layer.name}")
                    return layer.name
        
        # Fallback: search in main model
        for layer in reversed(self.model.layers):
            if 'conv' in layer.__class__.__name__.lower():
                print(f"Using layer: {layer.name}")
                return layer.name
        
        raise ValueError("Could not find suitable convolutional layer for Grad-CAM")
    
    def _build_grad_model(self):
        # Find the base model if it exists
        base_model = None
        for layer in self.model.layers:
            if hasattr(layer, 'layers') and len(layer.layers) > 10:
                base_model = layer
                break
        
        if base_model is not None:
            # Get the conv layer from base model
            try:
                conv_layer = base_model.get_layer(self.layer_name)
                
                # Create model that outputs conv layer output and final prediction
                grad_model = Model(
                    inputs=self.model.input,
                    outputs=[conv_layer.output, self.model.output]
                )
                return grad_model
            except:
                pass
        
        # Fallback: try to get layer from main model
        try:
            conv_layer = self.model.get_layer(self.layer_name)
            grad_model = Model(
                inputs=self.model.input,
                outputs=[conv_layer.output, self.model.output]
            )
            return grad_model
        except:
            raise ValueError(f"Could not build gradient model for layer: {self.layer_name}")
    
    def compute_heatmap(self, image, pred_index=None):
        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # Get convolutional layer output and predictions
            conv_outputs, predictions = self.grad_model(image)
            
            # Get the predicted class if not specified
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            
            # For binary classification, we want the positive class probability
            # predictions is shape (batch, 1) with sigmoid output
            class_channel = predictions[:, 0] if predictions.shape[-1] == 1 else predictions[:, pred_index]
        
        # Compute gradients of class output with respect to feature map
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Compute the guided gradients (global average pooling)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by importance
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        
        return heatmap.numpy()
    
    def overlay_heatmap(self, heatmap, image, alpha=0.4, colormap=cv2.COLORMAP_JET):
        # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convert heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, colormap)
        
        # Convert BGR to RGB
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Ensure image is in the right format
        if image.max() <= 1.0:
            image = np.uint8(255 * image)
        
        # Overlay heatmap on image
        superimposed = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        
        return superimposed
    
    def visualize(self, image, original_image=None, save_path=None):
        # Compute heatmap
        heatmap = self.compute_heatmap(image)
        
        # Get prediction
        prediction = self.model.predict(image, verbose=0)[0][0]
        predicted_class = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        # Prepare original image
        if original_image is None:
            original_image = image[0]
        
        # Create overlay
        overlay = self.overlay_heatmap(heatmap, original_image)
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(overlay)
        axes[2].set_title(f'Prediction: {predicted_class}\nConfidence: {confidence:.2%}')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
        
        return heatmap, overlay


if __name__ == "__main__":
    print("Grad-CAM module loaded successfully!")
    print("\nThis version is optimized for transfer learning models.")
    print("Supports: ResNet50, EfficientNet, VGG16, and custom CNNs")