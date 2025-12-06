import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, render_template, jsonify
import os
import random

# --- 1. CONFIGURATION ---
PHONE_WIDTH = 360
PHONE_HEIGHT = 640
OPACITY = 0.5

# Predefined lipstick colors (BGR format)
LIPSTICK_COLORS = [
    (30, 0, 180),      # Classic Red
    (60, 20, 220),     # Bright Red
    (80, 30, 150),     # Deep Berry
    (100, 50, 200),    # Pink Red
    (120, 80, 180),    # Mauve
    (140, 100, 160),   # Dusty Rose
    (50, 20, 100),     # Deep Plum
    (90, 60, 140),     # Rose
    (40, 10, 80),      # Burgundy
    (150, 120, 200),   # Soft Pink
    (70, 40, 120),     # Wine
    (110, 70, 170),    # Coral Pink
]

# Current color index
current_color_index = 0

# Corrected Landmark Indices
LIPS_OUTER = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    375, 321, 405, 314, 17, 84, 181, 91, 146
]
LIPS_INNER = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
    324, 318, 402, 317, 14, 87, 178, 88, 95
]

# --- 2. GLOBAL INITIALIZATION ---
mp_face_mesh = mp.solutions.face_mesh
try:
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
except Exception as e:
    print(f"Error initializing FaceMesh: {e}")
    face_mesh = None

image_path = "LEBRON.png"
static_image = cv2.imread(image_path)

if static_image is None:
    print(f"CRITICAL ERROR: Image '{image_path}' not found or could not be loaded!")
    static_image = np.zeros((PHONE_HEIGHT, PHONE_WIDTH, 3), dtype=np.uint8)
    cv2.putText(static_image, "Image Missing!", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_FOLDER = os.path.join(os.path.dirname(BASE_DIR), 'templates')

app = Flask(__name__, template_folder=TEMPLATE_FOLDER)

# --- UTILITY FUNCTIONS ---

def get_points(landmark_indices, landmarks, width, height):
    """Converts normalized landmarks to pixel coordinates."""
    points = []
    for i in landmark_indices:
        pt = landmarks.landmark[i]
        x = int(pt.x * width)
        y = int(pt.y * height)
        points.append((x, y))
    return np.array([points], dtype=np.int32)

def process_frame(frame, lipstick_color):
    """Applies the phone format crop and the lipstick filter."""
    h_orig, w_orig, _ = frame.shape

    # 1. Crop to 9:16 Aspect Ratio
    target_ratio = PHONE_WIDTH / PHONE_HEIGHT
    img_ratio = w_orig / h_orig

    if img_ratio > target_ratio:
        new_width = int(h_orig * target_ratio)
        offset = (w_orig - new_width) // 2
        cropped_img = frame[:, offset:offset+new_width]
    else:
        new_height = int(w_orig / target_ratio)
        offset = (h_orig - new_height) // 2
        cropped_img = frame[offset:offset+new_height, :]

    image = cv2.resize(cropped_img, (PHONE_WIDTH, PHONE_HEIGHT), interpolation=cv2.INTER_AREA)

    if face_mesh is None:
        return image

    # 2. FaceMesh Processing
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        return image

    # 3. Lipstick Application Logic
    h, w, _ = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    face_landmarks = results.multi_face_landmarks[0]

    outer_pts = get_points(LIPS_OUTER, face_landmarks, w, h)
    inner_pts = get_points(LIPS_INNER, face_landmarks, w, h)

    cv2.fillPoly(mask, outer_pts, 255)
    cv2.fillPoly(mask, inner_pts, 0)

    mask_blurred = cv2.GaussianBlur(mask, (15, 15), 5)
    lipstick_layer = np.zeros_like(image)
    lipstick_layer[:] = lipstick_color

    mask_float = mask_blurred.astype(float) / 255.0 * OPACITY
    mask_float = cv2.merge([mask_float, mask_float, mask_float])

    image_float = image.astype(float)
    lipstick_float = lipstick_layer.astype(float)

    output = (image_float * (1.0 - mask_float) + lipstick_float * mask_float).astype(np.uint8)

    return output

def generate_frames():
    """Generates the MJPEG stream from the processed frame."""
    global current_color_index

    while True:
        try:
            current_color = LIPSTICK_COLORS[current_color_index]
            processed_image = process_frame(static_image, current_color)

            ret, buffer = cv2.imencode('.jpg', processed_image)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Error during frame generation: {e}")
            return

# --- FLASK ROUTES ---

@app.route('/')
def index():
    """Renders the HTML template."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Supplies the continuous image stream."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/next_color', methods=['POST'])
def next_color():
    """Changes to next random lipstick color."""
    global current_color_index
    # Get a random color that's different from current
    new_index = random.randint(0, len(LIPSTICK_COLORS) - 1)
    while new_index == current_color_index and len(LIPSTICK_COLORS) > 1:
        new_index = random.randint(0, len(LIPSTICK_COLORS) - 1)

    current_color_index = new_index

    # Convert BGR to RGB for display
    color_rgb = LIPSTICK_COLORS[current_color_index][::-1]

    return jsonify({
        'success': True,
        'color_index': current_color_index,
        'color_rgb': color_rgb
    })

if __name__ == '__main__':

    app.run(debug=False, host='0.0.0.0', threaded=True)