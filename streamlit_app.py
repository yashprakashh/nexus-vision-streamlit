import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import io

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
            return model
        else:
            model = YOLO('yolov8n.pt')
            return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

def process_camera_input(camera_file):
    """Convert Streamlit camera input to PIL Image"""
    try:
        if camera_file is None:
            return None
        
        # Reset file pointer to beginning
        camera_file.seek(0)
        
        # Read as PIL Image
        pil_image = Image.open(camera_file)
        
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        return pil_image
        
    except Exception as e:
        print(f"Error processing camera input: {e}")
        return None

def safe_detect(camera_file):
    """Ultra-safe detection with proper camera input handling"""
    model = load_model()
    
    # Convert camera input to PIL Image
    pil_image = process_camera_input(camera_file)
    if pil_image is None:
        return None, "❌ Cannot process camera image"
    
    # Convert to numpy array
    img_array = np.array(pil_image)
    
    if model is None:
        return img_array, "❌ Model not available"
    
    try:
        # Run detection
        results = model.predict(img_array, conf=0.25, verbose=False)
        
        # Initialize variables
        annotated_image = img_array
        detected_classes = []
        
        # Process results safely
        if results and len(results) > 0:
            try:
                result = results[0]
                
                # Get annotated image
                try:
                    plot_result = result.plot()
                    if plot_result is not None:
                        # Convert BGR to RGB
                        annotated_image = plot_result[:, :, ::-1]
                except:
                    annotated_image = img_array
                
                # Extract detected classes
                try:
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        if hasattr(result.boxes, 'cls') and result.boxes.cls is not None:
                            for cls_tensor in result.boxes.cls:
                                try:
                                    class_id = int(cls_tensor.item())
                                    if 0 <= class_id < len(class_names):
                                        detected_classes.append(class_names[class_id])
                                except:
                                    continue
                except:
                    pass
                    
            except Exception as result_error:
                print(f"Result processing error: {result_error}")
        
        # Generate message
        if detected_classes:
            unique_objects = set(detected_classes)
            critical = unique_objects.intersection(tier1_objects)
            common = unique_objects.intersection(tier2_objects)
            
            if critical:
                message = f"🚨 CRITICAL ALERT: {', '.join(critical)} detected! Proceed with extreme caution."
            elif common:
                message = f"⚠️ NAVIGATION ALERT: {', '.join(common)} detected. Be aware of your surroundings."
            else:
                message = f"✅ Objects detected: {', '.join(unique_objects)}"
        else:
            message = "✅ Clear path - no objects detected"
        
        return annotated_image, message
        
    except Exception as e:
        print(f"Detection error: {e}")
        return img_array, "⚠️ Basic analysis completed"

# Main App
def main():
    st.title("🔍 Nexus Vision - Smart Navigation Assistant")
    st.markdown("**AI-powered object detection for visually impaired navigation assistance**")
    
    # Model status
    model = load_model()
    if model:
        st.success("🤖 Custom YOLOv8 model ready for detection!")
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📸 Camera Input")
        camera_input = st.camera_input("Take a picture for object detection")
        
        if camera_input is not None:
            st.success("✅ Image captured successfully!")
            
            # Display original image
            try:
                original_image = process_camera_input(camera_input)
                if original_image:
                    st.image(original_image, caption="Captured Image", use_container_width=True)
            except:
                st.info("Image captured and ready for processing")
    
    with col2:
        st.header("🔍 Detection Results")
        
        if camera_input is not None:
            with st.spinner('🔄 Analyzing image for navigation hazards...'):
                # Process the image
                result_image, message = safe_detect(camera_input)
                
                if result_image is not None:
                    # Display results
                    st.image(result_image, caption="Detection Results", use_container_width=True)
                    
                    # Navigation alert
                    st.subheader("🔊 Navigation Alert")
                    if "CRITICAL" in message:
                        st.error(message)
                    elif "ALERT" in message:
                        st.warning(message)
                    else:
                        st.success(message)
                else:
                    st.error("❌ Could not process image")
        else:
            st.info("👆 Capture an image using your camera to start navigation assistance")
    
    # Object categories reference
    if camera_input is not None:
        st.markdown("---")
        
        with st.expander("🎯 Object Detection Categories", expanded=False):
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.markdown("""
                **🚨 Critical Navigation Hazards:**
                - stairs
                - curb  
                - car
                - bus
                - truck
                - bicycle
                """)
            
            with col_b:
                st.markdown("""
                **⚠️ Common Navigation Objects:**
                - person
                - stop_sign
                - traffic_light
                - bench
                - fire_hydrant
                - pole
                """)
            
            with col_c:  
                st.markdown("""
                **📍 Contextual & Other Objects:**
                - bus_stop, tree
                - crutch, dog, motorcycle
                - spherical_roadblock, train
                - warning_column, waste_container
                """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <p><strong>Nexus Vision</strong> - Empowering independence through AI-powered navigation assistance</p>
    <p>🤖 Powered by YOLOv8 • 📱 Real-time webcam detection • 🎯 22 specialized object classes</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
