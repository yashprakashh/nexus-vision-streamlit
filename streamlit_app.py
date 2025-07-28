import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import tempfile
import cv2
from gtts import gTTS
import base64

# Enhanced page configuration
st.set_page_config(
    page_title="Nexus Vision - Smart Navigation Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced CSS styling for professional UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 3rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header h3 {
        font-size: 1.2rem;
        margin: 0.5rem 0;
        opacity: 0.9;
    }
    
    .alert-critical {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 5px solid #d63031;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
        animation: pulse 2s infinite;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #fdcb6e, #e17055);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 5px solid #e17055;
        box-shadow: 0 4px 15px rgba(253, 203, 110, 0.3);
    }
    
    .alert-safe {
        background: linear-gradient(135deg, #00b894, #00cec9);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 5px solid #00b894;
        box-shadow: 0 4px 15px rgba(0, 184, 148, 0.3);
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    .category-box {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .category-box:hover {
        transform: translateY(-3px);
    }
    
    .critical-category {
        background: linear-gradient(135deg, #ffebee, #ffcdd2);
        border-left: 5px solid #f44336;
    }
    
    .warning-category {
        background: linear-gradient(135deg, #fff3e0, #ffe0b2);
        border-left: 5px solid #ff9800;
    }
    
    .info-category {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        border-left: 5px solid #2196f3;
    }
    
    .metric-container {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        margin: 10px 0;
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-online { background-color: #00b894; }
    .status-offline { background-color: #d63031; }
    
    .feature-box {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 15px 0;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .feature-box:hover {
        transform: translateY(-5px);
    }
    
    .audio-indicator {
        background: linear-gradient(135deg, #a29bfe, #6c5ce7);
        color: white;
        padding: 10px 20px;
        border-radius: 25px;
        display: inline-block;
        margin: 10px 0;
        animation: audioWave 1.5s ease-in-out infinite;
    }
    
    @keyframes audioWave {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
</style>
""", unsafe_allow_html=True)

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

@st.cache_resource
def load_model():
    """Load YOLO model with enhanced status reporting"""
    try:
        if os.path.exists('best.pt'):
            model = YOLO('best.pt')
            return model, "custom"
        else:
            model = YOLO('yolov8n.pt')
            return model, "pretrained"
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None, "failed"

def process_camera_input(camera_file):
    """Convert Streamlit camera input to PIL Image with enhanced error handling"""
    try:
        if camera_file is None:
            return None
        
        camera_file.seek(0)
        pil_image = Image.open(camera_file)
        
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        return pil_image
        
    except Exception as e:
        print(f"Error processing camera input: {e}")
        return None

def fix_color_conversion(opencv_image):
    """FIXED: Properly convert OpenCV BGR to RGB for Streamlit"""
    try:
        if len(opencv_image.shape) == 3 and opencv_image.shape[2] == 3:
            # Use OpenCV's proper color conversion instead of array slicing
            rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
            return rgb_image
        return opencv_image
    except Exception as e:
        print(f"Color conversion error: {e}")
        return opencv_image

def generate_audio_alert(message, lang='en'):
    """Generate audio from text message using gTTS"""
    try:
        # Clean message for better TTS
        clean_message = message.replace('🚨', 'CRITICAL ALERT: ')
        clean_message = clean_message.replace('⚠️', 'WARNING: ')
        clean_message = clean_message.replace('✅', '')
        clean_message = clean_message.replace('**', '')
        
        # Create TTS object
        tts = gTTS(text=clean_message, lang=lang, slow=False)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tts.save(tmp_file.name)
            return tmp_file.name
            
    except Exception as e:
        print(f"Audio generation error: {e}")
        return None

def display_alert_with_audio(message, alert_type):
    """Display alert with both text and audio"""
    
    # Display enhanced text alert with styling
    if alert_type == "critical":
        st.markdown(f'''
        <div class="alert-critical">
            <h3>🚨 CRITICAL NAVIGATION HAZARD</h3>
            <p style="font-size: 1.2em; margin: 10px 0;"><strong>{message}</strong></p>
            <p style="opacity: 0.9;">⚠️ STOP and assess your surroundings before proceeding</p>
        </div>
        ''', unsafe_allow_html=True)
        
    elif alert_type == "warning":
        st.markdown(f'''
        <div class="alert-warning">
            <h3>⚠️ NAVIGATION ALERT</h3>
            <p style="font-size: 1.1em; margin: 10px 0;"><strong>{message}</strong></p>
            <p style="opacity: 0.9;">Please proceed with caution and awareness</p>
        </div>
        ''', unsafe_allow_html=True)
        
    else:
        st.markdown(f'''
        <div class="alert-safe">
            <h3>✅ NAVIGATION STATUS</h3>
            <p style="font-size: 1.1em; margin: 10px 0;"><strong>{message}</strong></p>
            <p style="opacity: 0.9;">Path appears clear for navigation</p>
        </div>
        ''', unsafe_allow_html=True)
    
    # Generate and display audio
    st.markdown('<div class="audio-indicator">🔊 Generating Audio Alert...</div>', unsafe_allow_html=True)
    
    audio_file = generate_audio_alert(message)
    if audio_file:
        try:
            with open(audio_file, 'rb') as f:
                audio_bytes = f.read()
            
            # Display audio player with autoplay
            st.audio(audio_bytes, format='audio/mp3')
            
            # Clean up temporary file
            try:
                os.unlink(audio_file)
            except:
                pass
                
        except Exception as e:
            st.warning(f"⚠️ Audio playback failed: {e}")
    else:
        st.warning("⚠️ Could not generate audio alert")

def safe_detect(camera_file):
    """Enhanced detection with proper color conversion and comprehensive analysis"""
    model, model_type = load_model()
    
    pil_image = process_camera_input(camera_file)
    if pil_image is None:
        return None, "❌ Cannot process camera image", {}
    
    img_array = np.array(pil_image)
    
    if model is None:
        return img_array, "❌ Model not available", {}
    
    try:
        # Run YOLO detection
        results = model.predict(img_array, conf=0.25, verbose=False)
        
        # Initialize variables
        annotated_image = img_array
        detected_classes = []
        confidence_scores = []
        
        # Process results with enhanced error handling
        if results and len(results) > 0:
            try:
                result = results[0]
                
                # Get annotated image with FIXED color conversion
                try:
                    plot_result = result.plot()
                    if plot_result is not None:
                        # FIXED: Use proper OpenCV color conversion
                        annotated_image = fix_color_conversion(plot_result)
                    else:
                        annotated_image = img_array
                except Exception as plot_error:
                    print(f"Plot error: {plot_error}")
                    annotated_image = img_array
                
                # Extract detected classes and confidence scores  
                try:
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        boxes = result.boxes
                        
                        if hasattr(boxes, 'cls') and boxes.cls is not None:
                            for i, cls_tensor in enumerate(boxes.cls):
                                try:
                                    class_id = int(cls_tensor.item())
                                    
                                    # Get confidence score
                                    confidence = 0.0
                                    if hasattr(boxes, 'conf') and len(boxes.conf) > i:
                                        confidence = float(boxes.conf[i].item())
                                    
                                    if 0 <= class_id < len(class_names):
                                        detected_classes.append(class_names[class_id])
                                        confidence_scores.append(confidence)
                                        
                                except Exception as detection_error:
                                    print(f"Individual detection error: {detection_error}")
                                    continue
                                    
                except Exception as class_error:
                    print(f"Class extraction error: {class_error}")
                    
            except Exception as result_error:
                print(f"Result processing error: {result_error}")
        
        # Enhanced message generation with detailed analysis
        stats = {
            "total": len(detected_classes),
            "critical": 0,
            "common": 0,
            "contextual": 0,
            "other": 0,
            "avg_confidence": 0.0
        }
        
        if detected_classes:
            unique_objects = set(detected_classes)
            critical = unique_objects.intersection(tier1_objects)
            common = unique_objects.intersection(tier2_objects)
            contextual = unique_objects.intersection(tier3_objects)
            other = unique_objects - critical - common - contextual
            
            stats["critical"] = len(critical)
            stats["common"] = len(common)
            stats["contextual"] = len(contextual)
            stats["other"] = len(other)
            stats["avg_confidence"] = np.mean(confidence_scores) if confidence_scores else 0.0
            
            # Generate priority-based messages
            if critical:
                message = f"CRITICAL NAVIGATION HAZARDS detected: {', '.join(sorted(critical))}. STOP and proceed with extreme caution!"
                alert_type = "critical"
            elif common:
                message = f"Common navigation objects detected: {', '.join(sorted(common))}. Please be aware of your surroundings."
                alert_type = "warning"
            else:
                all_objects = critical.union(common).union(contextual).union(other)
                message = f"Objects detected in area: {', '.join(sorted(all_objects))}. Path appears manageable."
                alert_type = "safe"
        else:
            message = "Clear path ahead - no objects detected in immediate area."
            alert_type = "safe"
        
        detailed_info = {
            "alert_type": alert_type,
            "stats": stats,
            "objects": detected_classes,
            "confidence_scores": confidence_scores,
            "model_type": model_type
        }
        
        return annotated_image, message, detailed_info
        
    except Exception as e:
        print(f"Detection error: {e}")
        return img_array, "Analysis completed with basic processing", {}

def main():
    # Enhanced header with gradient and animations
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Nexus Vision</h1>
        <h3>AI-Powered Smart Navigation Assistant</h3>
        <p>🤖 Advanced object detection • 🔊 Audio alerts • 🎯 22 specialized classes</p>
        <p><em>Empowering independence through intelligent navigation assistance</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced sidebar with system status and information
    with st.sidebar:
        st.markdown("## 🤖 System Status")
        
        model, model_type = load_model()
        if model:
            if model_type == "custom":
                st.markdown('''
                <div style="background: #e8f5e8; padding: 10px; border-radius: 8px; margin: 10px 0;">
                    <span class="status-indicator status-online"></span>
                    <strong>✅ Custom YOLOv8 Model Active</strong><br>
                    <small>🎯 Optimized for navigation with 22 specialized classes</small>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown('''
                <div style="background: #fff3e0; padding: 10px; border-radius: 8px; margin: 10px 0;">
                    <span class="status-indicator status-online"></span>
                    <strong>⚠️ Using YOLOv8n Pretrained</strong><br>
                    <small>Custom model not found, using fallback</small>
                </div>
                ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div style="background: #ffebee; padding: 10px; border-radius: 8px; margin: 10px 0;">
                <span class="status-indicator status-offline"></span>
                <strong>❌ Model Not Available</strong><br>
                <small>Detection system offline</small>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("## 📊 Object Categories")
        
        # Enhanced category display with styling
        st.markdown('''
        <div class="category-box critical-category">
            <strong>🚨 Critical Navigation Hazards</strong><br>
            <small>stairs • curb • car • bus • truck • bicycle</small><br>
            <em>Immediate attention required</em>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown('''
        <div class="category-box warning-category">
            <strong>⚠️ Common Navigation Objects</strong><br>
            <small>person • stop_sign • traffic_light • bench • fire_hydrant • pole</small><br>
            <em>Proceed with awareness</em>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown('''
        <div class="category-box info-category">
            <strong>📍 Contextual & Other Objects</strong><br>
            <small>bus_stop • tree • crutch • dog • motorcycle • spherical_roadblock • train • warning_column • waste_container</small><br>
            <em>Environmental context</em>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("## 🚀 Features")
        st.markdown("""
        - 🎯 **22 Specialized Classes** - Navigation-optimized detection
        - 🔊 **Audio Alerts** - Text-to-speech announcements
        - 📱 **Webcam Integration** - Instant photo analysis
        - 🚨 **Priority System** - Critical/Warning/Safe classifications
        - 🤖 **AI-Powered** - YOLOv8 computer vision
        - ♿ **Accessibility Focus** - Designed for visually impaired users
        """)
    
    # Main content area with enhanced layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📸 Camera Input")
        st.markdown("*Capture an image from your device camera for AI analysis*")
        
        camera_input = st.camera_input("📷 Take Picture for Navigation Analysis")
        
        if camera_input is not None:
            st.success("✅ Image captured successfully!")
            
            # Display original image with enhanced styling
            original_image = process_camera_input(camera_input)
            if original_image:
                st.image(
                    original_image, 
                    caption="📷 Original Camera Capture", 
                    use_container_width=True
                )
                
                # Image information
                img_array = np.array(original_image)
                st.info(f"📊 Image Size: {img_array.shape[1]}×{img_array.shape[0]} pixels")
    
    with col2:
        st.markdown("### 🔍 AI Detection Analysis")
        st.markdown("*Real-time object detection and navigation assistance*")
        
        if camera_input is not None:
            with st.spinner('🤖 AI Analysis in Progress...'):
                result_image, message, metadata = safe_detect(camera_input)
                
                if result_image is not None:
                    # Display detection results
                    st.image(
                        result_image, 
                        caption="🎯 AI Detection Results with Annotations", 
                        use_container_width=True
                    )
                    
                    # Display enhanced navigation alert with audio
                    st.markdown("### 🔊 Navigation Alert")
                    alert_type = metadata.get("alert_type", "safe")
                    display_alert_with_audio(message, alert_type)
                    
                else:
                    st.error("❌ Could not process image for detection")
        else:
            st.info("👆 **Capture an image** using your camera to start AI-powered navigation assistance")
            
            # Instructions for use
            st.markdown("""
            **How to use Nexus Vision:**
            1. 📷 Click 'Take Picture' above
            2. 🤖 AI will analyze the image
            3. 🔊 Listen to audio navigation alerts
            4. 🚶‍♂️ Navigate safely based on guidance
            """)
    
    # Detection statistics and detailed analysis
    if camera_input is not None:
        result_image, message, metadata = safe_detect(camera_input)
        
        if metadata and "stats" in metadata:
            st.markdown("---")
            st.markdown("### 📊 Detection Statistics")
            
            stats = metadata["stats"]
            
            # Display metrics in enhanced cards
            col_a, col_b, col_c, col_d, col_e = st.columns(5)
            
            with col_a:
                st.markdown(f'''
                <div class="metric-container">
                    <h3 style="color: #2c3e50; margin: 0;">{stats["total"]}</h3>
                    <p style="margin: 5px 0;">Total Objects</p>
                </div>
                ''', unsafe_allow_html=True)
            
            with col_b:
                st.markdown(f'''
                <div class="metric-container">
                    <h3 style="color: #e74c3c; margin: 0;">{stats["critical"]}</h3>
                    <p style="margin: 5px 0;">Critical</p>
                </div>
                ''', unsafe_allow_html=True)
            
            with col_c:
                st.markdown(f'''
                <div class="metric-container">
                    <h3 style="color: #f39c12; margin: 0;">{stats["common"]}</h3>
                    <p style="margin: 5px 0;">Common</p>
                </div>
                ''', unsafe_allow_html=True)
            
            with col_d:
                st.markdown(f'''
                <div class="metric-container">
                    <h3 style="color: #3498db; margin: 0;">{stats["contextual"]}</h3>
                    <p style="margin: 5px 0;">Context</p>
                </div>
                ''', unsafe_allow_html=True)
            
            with col_e:
                confidence_pct = f"{stats['avg_confidence']*100:.1f}%" if stats['avg_confidence'] > 0 else "N/A"
                st.markdown(f'''
                <div class="metric-container">
                    <h3 style="color: #27ae60; margin: 0;">{confidence_pct}</h3>
                    <p style="margin: 5px 0;">Confidence</p>
                </div>
                ''', unsafe_allow_html=True)
            
            # Detailed object list if any detected
            if metadata.get("objects"):
                st.markdown("### 📋 Detected Objects")
                objects = metadata["objects"]
                confidence_scores = metadata.get("confidence_scores", [])
                
                for i, obj in enumerate(objects):
                    confidence = confidence_scores[i] if i < len(confidence_scores) else 0.0
                    
                    # Determine category for styling
                    if obj in tier1_objects:
                        category_color = "#e74c3c"
                        category_icon = "🚨"
                        category_name = "Critical"
                    elif obj in tier2_objects:
                        category_color = "#f39c12"
                        category_icon = "⚠️"
                        category_name = "Common"
                    elif obj in tier3_objects:
                        category_color = "#3498db"
                        category_icon = "ℹ️"
                        category_name = "Context"
                    else:
                        category_color = "#95a5a6"
                        category_icon = "📍"
                        category_name = "Other"
                    
                    st.markdown(f'''
                    <div style="background: white; padding: 10px; margin: 5px 0; border-radius: 8px; border-left: 4px solid {category_color}; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                        <strong>{category_icon} {obj.replace('_', ' ').title()}</strong> 
                        <span style="color: {category_color};">({category_name})</span>
                        <span style="float: right; color: #7f8c8d;">Confidence: {confidence:.2f}</span>
                    </div>
                    ''', unsafe_allow_html=True)
    
    # Enhanced footer with feature highlights
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; background: linear-gradient(135deg, #f8f9fa, #e9ecef); padding: 30px; border-radius: 15px; margin: 20px 0;'>
        <h2 style="color: #2c3e50; margin-bottom: 20px;">🚀 Nexus Vision Capabilities</h2>
        
        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 30px 0;'>
            <div class="feature-box">
                <h3>🎯 Smart Detection</h3>
                <p>22 specialized object classes optimized for navigation assistance and safety</p>
            </div>
            <div class="feature-box">
                <h3>🔊 Audio Alerts</h3>
                <p>Real-time text-to-speech announcements with priority-based navigation guidance</p>
            </div>
            <div class="feature-box">
                <h3>🤖 AI-Powered</h3>
                <p>Advanced YOLOv8 computer vision for accurate real-time object detection</p>
            </div>
            <div class="feature-box">
                <h3>♿ Accessibility</h3>
                <p>Designed specifically for visually impaired users with intuitive audio feedback</p>
            </div>
        </div>
        
        <div style="margin-top: 30px; padding: 20px; background: rgba(255,255,255,0.8); border-radius: 10px;">
            <h4 style="color: #6c5ce7;">🌟 Empowering Independence Through AI</h4>
            <p style="color: #636e72; font-style: italic;">Advanced computer vision technology making navigation safer and more accessible for everyone</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
