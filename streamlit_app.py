import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from ultralytics import YOLO
import os

# Page configuration
st.set_page_config(
    page_title="Nexus Vision - Smart Navigation Assistant",
    page_icon="🔍",
    layout="wide"
)

# Load model with caching
@st.cache_resource
def load_model():
    """Load the trained YOLOv8 model"""
    try:
        model = YOLO('best.pt')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Initialize model
model = load_model()

# Your 22 specialized classes
class_names = [
    'bench', 'bicycle', 'bus', 'bus_stop', 'car', 'crutch', 'curb', 
    'dog', 'fire_hydrant', 'motorcycle', 'person', 'pole', 
    'spherical_roadblock', 'stairs', 'stop_sign', 'street_light', 
    'traffic_light', 'train', 'tree', 'truck', 'warning_column', 
    'waste_container'
]

# Priority tiers for navigation assistance
tier1_objects = {'stairs', 'curb', 'car', 'bus', 'truck', 'bicycle'}
tier2_objects = {'person', 'stop_sign', 'traffic_light', 'bench', 'fire_hydrant', 'pole'}
tier3_objects = {'bus_stop', 'tree'}

def ensure_valid_image_shape(image):
    """Ensure image has valid numpy array shape for Streamlit"""
    try:
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if not isinstance(image, np.ndarray):
            return None
        
        # Ensure proper shape for Streamlit (height, width) or (height, width, channels)
        if len(image.shape) == 2:
            # Grayscale image - valid
            return image
        elif len(image.shape) == 3:
            # Color image - ensure 3 channels (RGB)
            if image.shape[2] in [3, 4]:  # RGB or RGBA
                return image[:, :, :3]  # Convert RGBA to RGB if needed
            else:
                return None
        else:
            # Invalid shape
            return None
    except Exception as e:
        print(f"Error ensuring valid image shape: {e}")
        return None

def safe_process_yolo_results(results):
    """Safely extract detections from YOLO results with comprehensive error handling"""
    detections = []
    
    try:
        if not results or len(results) == 0:
            return detections
        
        result = results[0]
        
        # Multiple ways to access YOLO results depending on version
        boxes_data = None
        
        # Method 1: Direct boxes attribute
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes_data = result.boxes
        
        # Method 2: Check if it's a list/tensor directly
        elif hasattr(result, 'pred') and result.pred is not None:
            if len(result.pred) > 0:
                pred = result.pred[0]  # First prediction
                if len(pred) > 0:
                    # Extract from prediction tensor
                    for detection in pred:
                        if len(detection) >= 6:  # x1, y1, x2, y2, conf, class
                            class_id = int(detection[5].item() if hasattr(detection[5], 'item') else detection[5])
                            confidence = float(detection[4].item() if hasattr(detection[4], 'item') else detection[4])
                            
                            if 0 <= class_id < len(class_names):
                                detections.append({
                                    'class': class_names[class_id],
                                    'confidence': confidence,
                                    'bbox': detection[:4].tolist() if hasattr(detection[:4], 'tolist') else list(detection[:4])
                                })
                    return detections
        
        # Method 3: Process boxes_data if we have it
        if boxes_data is not None:
            try:
                # Get number of detections
                num_detections = 0
                
                if hasattr(boxes_data, 'cls') and boxes_data.cls is not None:
                    num_detections = len(boxes_data.cls)
                elif hasattr(boxes_data, 'data') and boxes_data.data is not None:
                    num_detections = len(boxes_data.data)
                
                for i in range(num_detections):
                    try:
                        # Extract class ID
                        class_id = None
                        confidence = 0.0
                        bbox = None
                        
                        # Try to get class ID
                        if hasattr(boxes_data, 'cls') and len(boxes_data.cls) > i:
                            cls_val = boxes_data.cls[i]
                            class_id = int(cls_val.item() if hasattr(cls_val, 'item') else cls_val)
                        
                        # Try to get confidence
                        if hasattr(boxes_data, 'conf') and len(boxes_data.conf) > i:
                            conf_val = boxes_data.conf[i]
                            confidence = float(conf_val.item() if hasattr(conf_val, 'item') else conf_val)
                        
                        # Try to get bounding box
                        if hasattr(boxes_data, 'xyxy') and len(boxes_data.xyxy) > i:
                            bbox_val = boxes_data.xyxy[i]
                            bbox = bbox_val.tolist() if hasattr(bbox_val, 'tolist') else list(bbox_val)
                        
                        # Validate and add detection
                        if class_id is not None and 0 <= class_id < len(class_names):
                            detections.append({
                                'class': class_names[class_id],
                                'confidence': confidence,
                                'bbox': bbox
                            })
                    
                    except Exception as detection_error:
                        print(f"Error processing detection {i}: {detection_error}")
                        continue
            
            except Exception as boxes_error:
                print(f"Error processing boxes data: {boxes_error}")
        
    except Exception as e:
        print(f"Error in safe_process_yolo_results: {e}")
    
    return detections

def draw_simple_boxes(image, detections):
    """Draw bounding boxes using PIL with maximum safety"""
    try:
        # Ensure we have a PIL image
        if isinstance(image, np.ndarray):
            # Ensure valid shape
            if len(image.shape) == 3 and image.shape[2] == 3:
                pil_image = Image.fromarray(image.astype(np.uint8))
            elif len(image.shape) == 2:
                pil_image = Image.fromarray(image.astype(np.uint8))
            else:
                print(f"Invalid image shape for drawing: {image.shape}")
                return image
        else:
            pil_image = image
        
        draw = ImageDraw.Draw(pil_image)
        
        # Draw boxes for each detection
        for detection in detections:
            try:
                class_name = detection['class']
                confidence = detection['confidence']
                bbox = detection.get('bbox')
                
                if bbox and len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[:4]
                    
                    # Determine color based on priority
                    if class_name in tier1_objects:
                        color = "red"
                    elif class_name in tier2_objects:
                        color = "orange"
                    elif class_name in tier3_objects:
                        color = "blue"
                    else:
                        color = "green"
                    
                    # Draw rectangle
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                    
                    # Draw label
                    label = f"{class_name}: {confidence:.2f}"
                    draw.rectangle([x1, y1-15, x1+120, y1], fill=color)
                    draw.text((x1+2, y1-13), label, fill="white")
            
            except Exception as box_error:
                print(f"Error drawing box for {detection.get('class', 'unknown')}: {box_error}")
                continue
        
        # Convert back to numpy array with proper shape
        result_array = np.array(pil_image)
        return ensure_valid_image_shape(result_array)
    
    except Exception as e:
        print(f"Error in draw_simple_boxes: {e}")
        return ensure_valid_image_shape(image)

def process_detection(image):
    """Ultra-safe image processing with comprehensive error handling"""
    if model is None:
        return None, "❌ Model not loaded", "Model not available"
    
    if image is None:
        return None, "❌ No image provided", "No image provided"
    
    try:
        # Convert to numpy array
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Ensure valid input shape
        img_array = ensure_valid_image_shape(img_array)
        if img_array is None:
            return None, "❌ Invalid image format", "Invalid image format"
        
        # Run YOLO inference
        results = model(img_array, conf=0.25, verbose=False)
        
        # Safely extract detections
        detections = safe_process_yolo_results(results)
        
        # Draw boxes if we have detections
        if detections:
            annotated_image = draw_simple_boxes(img_array, detections)
        else:
            annotated_image = ensure_valid_image_shape(img_array)
        
        # Ensure we return a valid image
        if annotated_image is None:
            annotated_image = ensure_valid_image_shape(img_array)
        
        # Generate messages
        detected_objects = {det['class'] for det in detections}
        audio_message = generate_audio_message(detected_objects)
        detailed_results = format_detection_results(detections)
        
        return annotated_image, audio_message, detailed_results
        
    except Exception as e:
        error_msg = f"Processing error: {str(e)}"
        print(f"Full error in process_detection: {e}")
        
        # Return original image with valid shape
        safe_image = ensure_valid_image_shape(image)
        return safe_image, f"❌ Error: {error_msg}", f"Error: {error_msg}"

def generate_audio_message(detected_objects):
    """Generate priority-based audio announcement"""
    if not detected_objects:
        return "✅ Clear path ahead. No objects detected."
    
    critical_objects = detected_objects.intersection(tier1_objects)
    common_objects = detected_objects.intersection(tier2_objects)
    contextual_objects = detected_objects.intersection(tier3_objects)
    
    if critical_objects:
        return f"🚨 **CRITICAL ALERT**: {', '.join(sorted(critical_objects))} detected. Please proceed with extreme caution!"
    elif common_objects:
        return f"⚠️ **NAVIGATION ALERT**: {', '.join(sorted(common_objects))} detected. Be aware of your surroundings."
    elif contextual_objects:
        return f"ℹ️ **CONTEXT INFO**: {', '.join(sorted(contextual_objects))} nearby. Safe to proceed."
    else:
        other_objects = detected_objects - tier1_objects - tier2_objects - tier3_objects
        return f"📍 Objects detected: {', '.join(sorted(other_objects))}."

def format_detection_results(detections):
    """Format detection results with confidence scores and priority levels"""
    if not detections:
        return "No objects detected in the scene."
    
    # Group by class and show highest confidence
    class_detections = {}
    for det in detections:
        class_name = det['class']
        confidence = det['confidence']
        
        if class_name not in class_detections or confidence > class_detections[class_name]['confidence']:
            class_detections[class_name] = det
    
    # Sort by priority and confidence
    results = []
    for class_name in sorted(class_detections.keys()):
        det = class_detections[class_name]
        confidence = det['confidence']
        
        # Determine priority tier
        if class_name in tier1_objects:
            tier = "🚨 **CRITICAL**"
            priority = 1
        elif class_name in tier2_objects:
            tier = "⚠️ **COMMON**"
            priority = 2
        elif class_name in tier3_objects:
            tier = "ℹ️ **CONTEXT**"
            priority = 3
        else:
            tier = "📍 **OTHER**"
            priority = 4
        
        results.append((priority, f"{tier} | **{class_name}** | Confidence: {confidence:.2f}"))
    
    # Sort by priority then by name
    results.sort(key=lambda x: (x[0], x[1]))
    return "\n".join([result[1] for result in results])

# Main App Interface
def main():
    st.title("🔍 Nexus Vision - Smart Navigation Assistant")
    st.markdown("**AI-powered object detection for visually impaired navigation assistance**")
    
    # Sidebar with information
    with st.sidebar:
        st.header("🎯 Detection Categories")
        st.markdown("""
        **🚨 Critical Objects:**  
        stairs, curb, car, bus, truck, bicycle
        
        **⚠️ Common Objects:**  
        person, stop_sign, traffic_light, bench, fire_hydrant, pole
        
        **ℹ️ Contextual Objects:**  
        bus_stop, tree
        
        **📍 Other Objects:**  
        crutch, dog, motorcycle, spherical_roadblock, train, warning_column, waste_container
        """)
        
        st.header("🚀 Features")
        st.markdown("""
        - ✅ Real-time webcam detection
        - ✅ 22 specialized object classes
        - ✅ Priority-based navigation alerts
        - ✅ Audio-ready announcements
        - ✅ Detailed confidence scoring
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📸 Camera Input")
        camera_input = st.camera_input("Take a picture for object detection")
        
        if camera_input is not None:
            st.success("✅ Image captured successfully!")
    
    with col2:
        st.header("🔍 Detection Results")
        
        if camera_input is not None:
            # Process the image
            with st.spinner('🔄 Analyzing image...'):
                annotated_image, audio_message, detailed_results = process_detection(camera_input)
            
            # Display results with additional safety check
            if annotated_image is not None:
                # ULTRA-SAFE: Double-check image shape before display
                try:
                    safe_image = ensure_valid_image_shape(annotated_image)
                    if safe_image is not None:
                        st.image(safe_image, caption="Detection Results", use_container_width=True)
                    else:
                        st.error("❌ Unable to display processed image")
                except Exception as img_error:
                    st.error(f"❌ Image display error: {img_error}")
                
                # Audio message (priority-based)
                st.header("🔊 Navigation Announcement")
                st.info(audio_message)
                
                # Detailed results
                st.header("📊 Detailed Detection Results")
                st.markdown(detailed_results)
            else:
                st.error("❌ Failed to process image")
        else:
            st.info("👆 Capture an image using your camera to start detection")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
    <p><strong>Nexus Vision</strong> - Empowering independence through AI-powered navigation assistance</p>
    <p>Built with YOLOv8 • Optimized for real-time performance • 22 specialized object classes</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
