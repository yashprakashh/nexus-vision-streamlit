import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os

st.set_page_config(
    page_title="Nexus Vision - Smart Navigation Assistant",
    page_icon="🔍",
    layout="wide"
)

# Class names for your 22 specialized classes
class_names = [
    'bench', 'bicycle', 'bus', 'bus_stop', 'car', 'crutch', 'curb', 
    'dog', 'fire_hydrant', 'motorcycle', 'person', 'pole', 
    'spherical_roadblock', 'stairs', 'stop_sign', 'street_light', 
    'traffic_light', 'train', 'tree', 'truck', 'warning_column', 
    'waste_container'
]

# Priority tiers
tier1_objects = {'stairs', 'curb', 'car', 'bus', 'truck', 'bicycle'}
tier2_objects = {'person', 'stop_sign', 'traffic_light', 'bench', 'fire_hydrant', 'pole'}

@st.cache_resource
def load_model():
    """Load YOLO model with maximum safety"""
    try:
        if os.path.exists('best.pt'):
            model = YOLO('best.pt')
            st.success("✅ Custom model loaded successfully!")
            return model
        else:
            st.warning("⚠️ Custom model not found, using YOLOv8n pretrained")
            model = YOLO('yolov8n.pt')  # Fallback to pretrained
            return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

def simple_detect(image):
    """Ultra-simple detection function"""
    model = load_model()
    
    if model is None:
        return image, "❌ Model not available"
    
    try:
        # Convert PIL to numpy
        img_array = np.array(image)
        
        # Run detection
        results = model.predict(img_array, conf=0.25, verbose=False)
        
        # Get annotated image directly from YOLO
        if results and len(results) > 0:
            annotated = results[0].plot()
            
            # Convert BGR to RGB for Streamlit
            if len(annotated.shape) == 3:
                annotated = annotated[:, :, ::-1]  # BGR to RGB
            
            # Extract detections for message
            detections = []
            if hasattr(results[0], 'boxes') and results[0].boxes is not None:
                for box in results[0].boxes:
                    try:
                        class_id = int(box.cls.item())
                        confidence = float(box.conf.item())
                        
                        if class_id < len(class_names):
                            detections.append(class_names[class_id])
                    except:
                        continue
            
            # Generate message
            if detections:
                unique_objects = set(detections)
                critical = unique_objects.intersection(tier1_objects)
                common = unique_objects.intersection(tier2_objects)
                
                if critical:
                    message = f"🚨 CRITICAL: {', '.join(critical)} detected!"
                elif common:
                    message = f"⚠️ ALERT: {', '.join(common)} detected"
                else:
                    message = f"✅ Objects detected: {', '.join(unique_objects)}"
            else:
                message = "✅ Clear path - no objects detected"
            
            return annotated, message
        else:
            return img_array, "✅ No objects detected"
            
    except Exception as e:
        st.error(f"Detection error: {e}")
        return np.array(image), f"❌ Processing failed: {str(e)}"

# Main App
def main():
    st.title("🔍 Nexus Vision - Smart Navigation Assistant")
    st.markdown("**AI-powered object detection for visually impaired navigation**")
    
    # Model status
    model = load_model()
    if model:
        st.success("🤖 Model ready for detection")
    
    # Camera input
    st.header("📸 Camera Input")
    camera_input = st.camera_input("Take a picture for object detection")
    
    if camera_input is not None:
        st.success("✅ Image captured!")
        
        with st.spinner('🔄 Analyzing...'):
            # Process image
            result_image, message = simple_detect(camera_input)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(camera_input, use_container_width=True)
            
            with col2:
                st.subheader("Detection Results")
                st.image(result_image, use_container_width=True)
        
        # Navigation message
        st.header("🔊 Navigation Alert")
        st.info(message)
        
        # Object categories info
        with st.expander("🎯 Object Categories"):
            st.markdown("""
            **🚨 Critical Objects:** stairs, curb, car, bus, truck, bicycle  
            **⚠️ Common Objects:** person, stop_sign, traffic_light, bench, fire_hydrant, pole  
            **📍 Other Objects:** All remaining 10+ classes
            """)

if __name__ == "__main__":
    main()
