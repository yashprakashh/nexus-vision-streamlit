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

def draw_bounding_boxes(image, results):
    """Draw bounding boxes on image using PIL with robust error handling"""
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        draw = ImageDraw.Draw(image)
        
        # Check if results exist and have boxes
        if not results or len(results) == 0:
            return np.array(image)
        
        result = results[0]  # Get first result
        
        # Check if boxes exist
        if not hasattr(result, 'boxes') or result.boxes is None:
            return np.array(image)
        
        boxes = result.boxes
        
        # Handle different box formats
        if hasattr(boxes, 'xyxy') and boxes.xyxy is not None and len(boxes.xyxy) > 0:
            for i, box in enumerate(boxes.xyxy):
                try:
                    # Get coordinates - handle both tensor and list formats
                    if hasattr(box, 'tolist'):
                        coords = box.tolist()
                    else:
                        coords = box
                    
                    if len(coords) >= 4:
                        x1, y1, x2, y2 = coords[:4]
                        
                        # Get class info
                        if hasattr(boxes, 'cls') and len(boxes.cls) > i:
                            class_id = int(boxes.cls[i].item()) if hasattr(boxes.cls[i], 'item') else int(boxes.cls[i])
                            
                            if class_id < len(class_names):
                                class_name = class_names[class_id]
                                
                                # Get confidence
                                confidence = 0.0
                                if hasattr(boxes, 'conf') and len(boxes.conf) > i:
                                    confidence = float(boxes.conf[i].item()) if hasattr(boxes.conf[i], 'item') else float(boxes.conf[i])
                                
                                # Determine color based on priority
                                if class_name in tier1_objects:
                                    color = "red"
                                elif class_name in tier2_objects:
                                    color = "orange"
                                elif class_name in tier3_objects:
                                    color = "blue"
                                else:
                                    color = "green"
                                
                                # Draw bounding box
                                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                                
                                # Draw label background
                                label = f"{class_name}: {confidence:.2f}"
                                bbox = draw.textbbox((x1, y1-20), label)
                                draw.rectangle(bbox, fill=color)
                                
                                # Draw label text
                                draw.text((x1+2, y1-18), label, fill="white")
                
                except Exception as box_error:
                    print(f"Error processing individual box {i}: {box_error}")
                    continue
        
        return np.array(image)
        
    except Exception as e:
        print(f"Error in draw_bounding_boxes: {e}")
        # Return original image if drawing fails
        return np.array(image) if isinstance(image, Image.Image) else image

def process_detection(image):
    """Process image and return results with robust error handling"""
    if model is None:
        return None, "❌ Model not loaded", "No detection available"
    
    if image is None:
        return None, "❌ No image provided", "No image provided"
    
    try:
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Run YOLO inference with error handling
        results = model(img_array, conf=0.25, verbose=False)
        
        # Initialize detection variables
        detections = []
        detected_objects = set()
        
        # Process results with robust error handling
        if results and len(results) > 0:
            result = results[0]
            
            # Check if boxes exist and process them
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                
                # Handle both tensor and direct access
                try:
                    if hasattr(boxes, 'cls') and boxes.cls is not None:
                        num_detections = len(boxes.cls)
                        
                        for i in range(num_detections):
                            try:
                                # Get class ID
                                class_id = int(boxes.cls[i].item()) if hasattr(boxes.cls[i], 'item') else int(boxes.cls[i])
                                
                                # Get confidence
                                confidence = 0.0
                                if hasattr(boxes, 'conf') and len(boxes.conf) > i:
                                    confidence = float(boxes.conf[i].item()) if hasattr(boxes.conf[i], 'item') else float(boxes.conf[i])
                                
                                # Validate class ID
                                if 0 <= class_id < len(class_names):
                                    class_name = class_names[class_id]
                                    detected_objects.add(class_name)
                                    
                                    detections.append({
                                        'class': class_name,
                                        'confidence': confidence
                                    })
                                    
                            except Exception as detection_error:
                                print(f"Error processing detection {i}: {detection_error}")
                                continue
                                
                except Exception as boxes_error:
                    print(f"Error accessing boxes: {boxes_error}")
        
        # Draw annotations
        annotated_image = draw_bounding_boxes(img_array, results)
        
        # Generate messages
        audio_message = generate_audio_message(detected_objects)
        detailed_results = format_detection_results(detections)
        
        return annotated_image, audio_message, detailed_results
        
    except Exception as e:
        error_msg = f"Processing error: {str(e)}"
        print(f"Full error in process_detection: {e}")
        return np.array(image), f"❌ Error: {error_msg}", f"Error: {error_msg}"

def generate_audio_message(detected_objects):
    """Generate priority-based audio announcement"""
    if not detected_objects:
        return "✅ Clear path ahead. No objects detected."
    
    # Check for critical objects first
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
            
            # Display results
            if annotated_image is not None:
                # FIXED: use_container_width instead of use_column_width
                st.image(annotated_image, caption="Detection Results", use_container_width=True)
                
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
