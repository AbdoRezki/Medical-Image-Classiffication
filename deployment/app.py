import os
import sys
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# Add parent directory to path to import custom modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.gradcam import GradCAM


# Configure page
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .normal {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .pneumonia {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    .info-box {
        padding: 1rem;
        background-color: #e7f3ff;
        border-left: 4px solid #1f77b4;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path):
    """Load and cache the trained model"""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def preprocess_image(image, img_size=(224, 224)):
    """Preprocess uploaded image for model prediction"""
    # Resize image
    image = image.resize(img_size)
    
    # Convert to array
    img_array = np.array(image)
    
    # Convert to RGB if grayscale
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    
    # Normalize
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def make_prediction(model, image):
    """Make prediction on preprocessed image"""
    prediction = model.predict(image, verbose=0)[0][0]
    
    if prediction > 0.5:
        diagnosis = "PNEUMONIA"
        confidence = prediction
        class_idx = 1
    else:
        diagnosis = "NORMAL"
        confidence = 1 - prediction
        class_idx = 0
    
    return diagnosis, confidence, class_idx


def generate_gradcam(model, image, original_image):
    """Generate Grad-CAM visualization"""
    try:
        gradcam = GradCAM(model)
        heatmap = gradcam.compute_heatmap(image)
        overlay = gradcam.overlay_heatmap(heatmap, np.array(original_image))
        return heatmap, overlay
    except Exception as e:
        st.warning(f"Could not generate Grad-CAM: {e}")
        return None, None


def main():
    # Header
    st.markdown('<h1 class="main-header">🫁 Chest X-Ray Pneumonia Detection</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered medical imaging analysis using deep learning</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.info("""
        This application uses deep learning to detect pneumonia from chest X-ray images.
        
        **Model**: Transfer Learning with ResNet50/EfficientNet
        
        **Features**:
        - Binary classification (Normal vs Pneumonia)
        - Grad-CAM visualization
        - Confidence scores
        
        **Disclaimer**: This is a demonstration tool for educational purposes only. 
        Always consult healthcare professionals for medical diagnosis.
        """)
        
        st.header("⚙️ Settings")
        
        # Model selection
        model_files = {
            "ResNet50": "../models/best_resnet50_model.h5",
            "EfficientNetB0": "../models/best_efficientnetb0_model.h5",
        }
        
        available_models = {
            name: path for name, path in model_files.items() 
            if os.path.exists(path)
        }
        
        if not available_models:
            st.error("No trained models found! Please train a model first.")
            st.stop()
        
        selected_model = st.selectbox(
            "Select Model",
            options=list(available_models.keys())
        )
        
        show_gradcam = st.checkbox("Show Grad-CAM visualization", value=True)
        
        st.header("📊 Model Info")
        model_path = available_models[selected_model]
        if os.path.exists(model_path):
            st.success(f"✓ Model loaded: {selected_model}")
        else:
            st.error(f"✗ Model not found: {model_path}")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload X-Ray Image")
        
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a chest X-ray image for pneumonia detection"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded X-Ray', width='stretch')
            
            # Analyze button
            if st.button("🔍 Analyze X-Ray", type="primary", width='stretch'):
                with st.spinner("Analyzing image..."):
                    # Load model
                    model = load_model(model_path)
                    
                    if model is None:
                        st.error("Failed to load model!")
                        st.stop()
                    
                    # Preprocess image
                    processed_image = preprocess_image(image)
                    
                    # Make prediction
                    diagnosis, confidence, class_idx = make_prediction(model, processed_image)
                    
                    # Store results in session state
                    st.session_state['diagnosis'] = diagnosis
                    st.session_state['confidence'] = confidence
                    st.session_state['image'] = image
                    st.session_state['processed_image'] = processed_image
                    st.session_state['model'] = model
                    
                    st.success("Analysis complete!")
    
    with col2:
        st.header("📋 Results")
        
        if 'diagnosis' in st.session_state:
            diagnosis = st.session_state['diagnosis']
            confidence = st.session_state['confidence']
            
            # Display prediction
            box_class = "pneumonia" if diagnosis == "PNEUMONIA" else "normal"
            st.markdown(f"""
                <div class="prediction-box {box_class}">
                    <h2>Diagnosis: {diagnosis}</h2>
                    <h3>Confidence: {confidence:.2%}</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Confidence bar
            st.progress(float(confidence))
            
            # Additional info
            if diagnosis == "PNEUMONIA":
                st.markdown("""
                <div class="info-box">
                    <strong>⚠️ Pneumonia Detected</strong><br>
                    The model detected signs of pneumonia in the X-ray. 
                    Please consult a healthcare professional for proper diagnosis and treatment.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-box">
                    <strong>✓ No Pneumonia Detected</strong><br>
                    The model did not detect signs of pneumonia. 
                    However, always consult healthcare professionals for medical concerns.
                </div>
                """, unsafe_allow_html=True)
            
            # Grad-CAM visualization
            if show_gradcam:
                st.header("🔥 Grad-CAM Visualization")
                st.write("Heatmap showing which regions influenced the model's decision:")
                
                with st.spinner("Generating visualization..."):
                    heatmap, overlay = generate_gradcam(
                        st.session_state['model'],
                        st.session_state['processed_image'],
                        st.session_state['image']
                    )
                    
                    if overlay is not None:
                        # Display Grad-CAM
                        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                        
                        axes[0].imshow(heatmap, cmap='jet')
                        axes[0].set_title('Activation Heatmap', fontsize=14)
                        axes[0].axis('off')
                        
                        axes[1].imshow(overlay)
                        axes[1].set_title('Overlay on X-Ray', fontsize=14)
                        axes[1].axis('off')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        st.caption("""
                        Red/yellow regions indicate areas the model focused on when making its decision. 
                        In healthy lungs, the model should focus on lung fields and look for opacity or consolidation patterns.
                        """)
        else:
            st.info("👈 Upload an X-ray image and click 'Analyze' to see results")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>Built with Streamlit and TensorFlow | For educational purposes only</p>
            <p><strong>Disclaimer</strong>: This tool is not a substitute for professional medical advice, 
            diagnosis, or treatment. Always seek the advice of qualified health providers.</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()