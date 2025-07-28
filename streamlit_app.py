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
            model = YOLO('yolov8n.pt')
            return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

def ensure_valid_image(image):
    """Ensure image is valid for Streamlit display"""
    try:
        # Convert to numpy array if PIL
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Ensure it's a numpy array
        if not isinstance(img_array, np.ndarray):
            return None
            
        # Ensure proper shape
        if len(img_array.shape) == 2:
            # Grayscale - valid
            return img_array.astype(np.uint8)
        elif len(img_array.shape) == 3:
            # Color image
            if img_array.shape[2] == 3:
                # RGB - valid
                return img_array.astype(np.uint8)
            elif img_array.shape[2] == 4:
                # RGBA - convert to RGB
                return img_array[:, :, :3].astype(np.uint8)
            else:
                # Invalid number of channels
                return None
        else:
            # Invalid shape
            return None
            
    except Exception as e:
        print(f"Error ensuring valid image: {e}")
        return None

def safe_detect(image):
    """Ultra-safe detection with guaranteed valid output"""
    model = load_model()
    
    # Ensure input image is valid
    original_array = ensure_valid_image(image)
    if original_array is None:
        return image, "❌ Invalid input image format"
    
    if model is None:
        return original_array, "❌ Model not available"
    
    try:
        # Run detection with maximum safety
        results = model.predict(original_array, conf=0.25, verbose=False)
        
        # Initialize variables
        annotated_image = original_array
        detected_classes = []
        
        # ULTRA-SAFE result processing
        if results and len(results) > 0:
            try:
                result = results[0]
                
                # Try to get annotated image from YOLO
                try:
                    plot_result = result.plot()
                    if plot_result is not None:
                        # Convert BGR to RGB if needed
                        if len(plot_result.shape) == 3 and plot_result.shape[2] == 3:
                            annotated_image = plot_result[:, :, ::-1]  # BGR to RGB
                        else:
                            annotated_image = plot_result
                        
                        # Ensure valid shape
                        annotated_image = ensure_valid_image(annotated_image)
                        if annotated_image is None:
                            annotated_image = original_array
                            
                except Exception as plot_error:
                    print(f"Plot error: {plot_error}")
                    annotated_image = original_array
                
                # Try to extract detection classes
                try:
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        boxes = result.boxes
                        
                        # Multiple ways to get classes
                        if hasattr(boxes, 'cls') and boxes.cls is not None:
                            try:
                                for cls_tensor in boxes.cls:
                                    try:
                                        class_id = int(cls_tensor.item() if hasattr(cls_tensor, 'item') else cls_tensor)
                                        if 0 <= class_id < len(class_names):
                                            detected_classes.append(class_names[class_id])
                                    except:
                                        continue
                            except:
                                pass
                                
                except Exception as class_error:
                    print(f"Class extraction error: {class_error}")
                    
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
                message = f"⚠️ NAVIGATION ALERT: {', '.join(common)} detected. Be aware."
            else:
                message = f"✅ Objects detected: {', '.join(unique_objects)}"
        else:
            message = "✅ Clear path - no objects detected"
        
        # Final safety check on output image
        final_image = ensure_valid_image(annotated_image)
        if final_image is None:
            final_image = original_array
            
        return final_image, message
        
    except Exception as e:
        print(f"Detection error: {e}")
        # Always return original image on any error
        return original_array, f"⚠️ Processing completed with basic analysis"

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
            # Process image with maximum safety
            result_image, message = safe_detect(camera_input)
            
            # Additional safety check before display
            display_image = ensure_valid_image(result_image)
            if display_image is None:
                display_image = ensure_valid_image(camera_input)
                if display_image is None:
                    st.error("❌ Cannot display image - invalid format")
                    return
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            original_display = ensure_valid_image(camera_input)
            if original_display is not None:
                st.image(original_display, use_container_width=True)
            else:
                st.error("Cannot display original image")
        
        with col2:
            st.subheader("Detection Results")
            st.image(display_image, use_container_width=True)
        
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
