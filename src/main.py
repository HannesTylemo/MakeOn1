import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, render_template, jsonify, request, send_from_directory, redirect, url_for
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
import json
import base64
import uuid
from werkzeug.utils import secure_filename

# --- PATH CONFIGURATION ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR) # Up one level to Skintoner/

TEMPLATE_FOLDER = os.path.join(ROOT_DIR, 'templates')
STATIC_FOLDER = os.path.join(ROOT_DIR, 'static')
UPLOADS_FOLDER = os.path.join(STATIC_FOLDER, 'uploads')
PRODUCTS_FILE = os.path.join(CURRENT_DIR, 'product_catalog.json')

if not os.path.exists(UPLOADS_FOLDER):
    os.makedirs(UPLOADS_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder=TEMPLATE_FOLDER, static_folder=STATIC_FOLDER)
app.secret_key = 'skintoner_admin_key'
CORS(app)

# --- AUTH ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

@login_manager.user_loader
def load_user(user_id):
    if user_id == '1': return User('1', 'admin')
    return None

# --- PRODUCTS ---
products_db = []

def load_products():
    global products_db
    try:
        with open(PRODUCTS_FILE, 'r', encoding='utf-8') as f:
            products_db = json.load(f)
    except: products_db = []

def save_products():
    try:
        with open(PRODUCTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(products_db, f, indent=2)
    except: pass

load_products()

# --- ROUTES ---

@app.route('/')
def index():
    # FIXED: Now correctly points to index.html as you requested
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form.get('username') == 'admin' and request.form.get('password') == 'password123':
            login_user(User('1', 'admin'))
            return redirect(url_for('dashboard'))
        return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    load_products()
    return render_template('dashboard.html', products=products_db, user=current_user.username)

@app.route('/add_product', methods=['POST'])
@login_required
def add_product():
    global products_db
    image_url = ""
    file = request.files.get('product_image')
    if file and file.filename:
        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex[:6]}_{filename}"
        file.save(os.path.join(UPLOADS_FOLDER, unique_name))
        image_url = f"/static/uploads/{unique_name}"

    category = request.form.get('category')
    new_product = {
        "id": f"prod_{uuid.uuid4().hex[:8]}",
        "category": category,
        "brand": request.form.get('brand'),
        "product_name": request.form.get('product_name'),
        "hex_color": request.form.get('hex_color'),
        "price": request.form.get('price', "25.00"),
        "description": request.form.get('description', ""),
        "url": "#",
        "image_url": image_url
    }

    if category == 'lipstick':
        new_product['pigment'] = int(request.form.get('pigment', 70))
        new_product['shine'] = int(request.form.get('shine', 30))
        new_product['effect'] = request.form.get('effect', 'none')

    products_db.insert(0, new_product)
    save_products()
    return redirect(url_for('dashboard'))

@app.route('/edit_product', methods=['POST'])
@login_required
def edit_product():
    global products_db
    prod_id = request.form.get('id')

    # Find the product to edit
    product = next((p for p in products_db if p['id'] == prod_id), None)
    if not product:
        return redirect(url_for('dashboard'))

    # Handle image upload if provided
    file = request.files.get('product_image')
    if file and file.filename:
        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex[:6]}_{filename}"
        file.save(os.path.join(UPLOADS_FOLDER, unique_name))
        product['image_url'] = f"/static/uploads/{unique_name}"

    # Update basic product fields
    category = request.form.get('category')
    product['category'] = category
    product['brand'] = request.form.get('brand')
    product['product_name'] = request.form.get('product_name')
    product['hex_color'] = request.form.get('hex_color')
    product['price'] = request.form.get('price', "25.00")
    product['description'] = request.form.get('description', "")

    # Update category-specific fields
    if category == 'lipstick':
        product['pigment'] = int(request.form.get('pigment', 70))
        product['shine'] = int(request.form.get('shine', 30))
        product['effect'] = request.form.get('effect', 'none')
    else:
        # Remove lipstick-specific properties if changed to mascara
        product.pop('pigment', None)
        product.pop('shine', None)
        product.pop('effect', None)

    save_products()
    return redirect(url_for('dashboard'))

@app.route('/delete_product', methods=['POST'])
@login_required
def delete_product():
    global products_db
    prod_id = request.form.get('id')
    products_db = [p for p in products_db if p['id'] != prod_id]
    save_products()
    return redirect(url_for('dashboard'))

@app.route('/product/<product_id>')
def product_detail(product_id):
    load_products()
    product = next((p for p in products_db if p['id'] == product_id), None)
    if not product: return "Product not found", 404
    return render_template('product_detail.html', product=product)

@app.route('/mock')
def mock_site():
    return send_from_directory(CURRENT_DIR, 'shopify_mock.html')

@app.route('/widget.js')
def wjs(): return send_from_directory(STATIC_FOLDER, 'widget.js')

@app.route('/static/<path:f>')
def st(f): return send_from_directory(STATIC_FOLDER, f)

# --- ERROR CODES ---
ERROR_NO_FACE = 'no_face_detected'
ERROR_PROCESSING_FAILED = 'processing_failed'

# --- VTO LOGIC ---
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
LEFT_EYE_UPPER = [362, 382, 381, 380, 374, 373, 390, 249, 263]
RIGHT_EYE_UPPER = [33, 7, 163, 144, 145, 153, 154, 155, 133]

def hex_to_lab(hex_color):
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    bgr = (rgb[2], rgb[1], rgb[0])
    return cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0][0]

def get_points(indices, landmarks, w, h):
    return np.array([[(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in indices]], dtype=np.int32)

def create_mask(shape, outer, inner=None):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, outer, 255)
    if inner is not None: cv2.fillPoly(mask, inner, 0)
    return cv2.GaussianBlur(mask, (7, 7), 0)

def detect_hand_occlusion(img, lip_outer_pts, hand_results):
    """
    Detect if hands are occluding (covering) the lips area.
    Returns a mask where occluded areas are set to 0 (black).
    """
    h, w = img.shape[:2]
    occlusion_mask = np.ones((h, w), dtype=np.uint8) * 255  # Start with no occlusion

    if not hand_results.multi_hand_landmarks:
        return occlusion_mask  # No hands detected, no occlusion

    # Extract lip points - handle different array shapes robustly
    if isinstance(lip_outer_pts, np.ndarray):
        # Flatten to 2D array of points regardless of input shape
        lip_pts_flat = lip_outer_pts.reshape(-1, 2)
    else:
        lip_pts_flat = np.array(lip_outer_pts).reshape(-1, 2)

    lip_min_x, lip_min_y = lip_pts_flat.min(axis=0)
    lip_max_x, lip_max_y = lip_pts_flat.max(axis=0)

    # Check each detected hand
    for hand_landmarks in hand_results.multi_hand_landmarks:
        # Get all 21 hand landmarks
        hand_points = []
        for idx in range(21):
            landmark = hand_landmarks.landmark[idx]
            x, y = int(landmark.x * w), int(landmark.y * h)
            hand_points.append((x, y))

        hand_points = np.array(hand_points)

        # Create convex hull around the hand (MediaPipe always provides 21 points)
        hull = cv2.convexHull(hand_points)

        # Check if hand overlaps with lip area
        hand_min_x, hand_min_y = hand_points.min(axis=0)
        hand_max_x, hand_max_y = hand_points.max(axis=0)

        # If bounding boxes overlap, hand might be occluding lips
        if (hand_min_x < lip_max_x and hand_max_x > lip_min_x and
            hand_min_y < lip_max_y and hand_max_y > lip_min_y):
            # Fill the hand area in the occlusion mask with 0 (occluded)
            cv2.fillConvexPoly(occlusion_mask, hull, 0)

    return occlusion_mask

def apply_lipstick_physics(img, mask, hex_color, pigment, shine, effect, occlusion_mask=None):
    """
    Apply lipstick with physics-based rendering.
    If occlusion_mask is provided, lipstick will not be applied to occluded areas.
    """
    # Apply occlusion mask if provided
    if occlusion_mask is not None:
        # Combine lipstick mask with occlusion mask
        # Only apply lipstick where both masks allow (bitwise AND)
        mask = cv2.bitwise_and(mask, occlusion_mask)

    opacity = pigment / 100.0
    shine_factor = shine / 100.0
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    target_l, target_a, target_b = hex_to_lab(hex_color)
    alpha = (mask.astype(float) / 255.0) * opacity
    a_blended = (a * (1.0 - alpha) + np.full_like(a, target_a) * alpha).astype(np.uint8)
    b_blended = (b * (1.0 - alpha) + np.full_like(b, target_b) * alpha).astype(np.uint8)
    l_float = l.astype(float)
    ret, highlights = cv2.threshold(l, 160, 255, cv2.THRESH_BINARY)
    highlights = cv2.GaussianBlur(highlights, (15, 15), 0)
    highlight_mask = (highlights.astype(float) / 255.0) * alpha * shine_factor
    l_shiny = l_float + (highlight_mask * 60.0)
    if effect == 'metallic': l_shiny += (highlight_mask * 40.0)
    elif effect == 'glitter':
        noise = np.zeros_like(l)
        cv2.randn(noise, 0, 40)
        l_shiny += (noise * (mask.astype(float)/255.0) * shine_factor * 0.5)
    l_final = np.clip(l_shiny, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([l_final, a_blended, b_blended]), cv2.COLOR_LAB2BGR)

def apply_mascara(img, left_pts, right_pts, hex_color):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.polylines(mask, [left_pts], False, 255, 2, cv2.LINE_AA)
    cv2.polylines(mask, [right_pts], False, 255, 2, cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    img_float = img.astype(float)
    mask_f = (mask.astype(float) / 255.0)[:,:,np.newaxis] * 0.8
    return (img_float * (1.0 - mask_f)).astype(np.uint8)

active_image_data = None

@app.route('/frame')
def get_frame():
    """
    Generate a makeup preview frame with optional hand occlusion detection.
    Note: Hand detection is performed on each request for accuracy.
    This is appropriate since frames are generated on-demand, not in real-time video.
    Supports multiple product IDs separated by pipe (|) character.
    """
    global active_image_data
    if active_image_data is None: return "No Image", 404
    img = active_image_data.copy()
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks: return cv2.imencode('.jpg', img)[1].tobytes()
    lm = results.multi_face_landmarks[0]
    
    # Handle multiple product IDs separated by pipe character
    pid_param = request.args.get('id', '')
    if pid_param:
        product_ids = pid_param.split('|')
        
        # Apply each product in order
        for pid in product_ids:
            pid = pid.strip()
            if not pid:
                continue
                
            prod = next((p for p in products_db if p['id'] == pid), None)
            if prod:
                cat = prod.get('category', 'lipstick')
                if cat == 'lipstick':
                    outer = get_points(LIPS_OUTER, lm, w, h)
                    inner = get_points(LIPS_INNER, lm, w, h)
                    mask = create_mask(img.shape, outer, inner)

                    # Detect hand occlusion for realistic rendering
                    hand_results = hands.process(rgb)
                    occlusion_mask = detect_hand_occlusion(img, outer, hand_results)

                    # Apply lipstick with occlusion handling
                    img = apply_lipstick_physics(img, mask, prod.get('hex_color', '#cc0000'),
                                                prod.get('pigment', 70), prod.get('shine', 30),
                                                prod.get('effect', 'none'), occlusion_mask)
                elif cat == 'mascara':
                    l_eye = get_points(LEFT_EYE_UPPER, lm, w, h)[0]
                    r_eye = get_points(RIGHT_EYE_UPPER, lm, w, h)[0]
                    img = apply_mascara(img, l_eye, r_eye, prod.get('hex_color', '#000000'))
    
    ret, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return Response(buf.tobytes(), mimetype='image/jpeg')

@app.route('/upload_selfie', methods=['POST'])
def upload_selfie():
    global active_image_data
    try:
        # Get image data from request
        image_data = request.json.get('image')
        if not image_data:
            print("Error: No image data in request")
            return jsonify({'success': False, 'error': ERROR_PROCESSING_FAILED})

        # Split and decode base64 image - expect data:image/...;base64,<data> format
        if ',' not in image_data:
            print("Error: Invalid image data format - missing comma separator")
            return jsonify({'success': False, 'error': ERROR_PROCESSING_FAILED})

        d = image_data.split(',')[1]
        n = np.frombuffer(base64.b64decode(d), np.uint8)
        img = cv2.imdecode(n, cv2.IMREAD_COLOR)

        # Check if image was decoded successfully
        if img is None:
            print("Error: Failed to decode image")
            return jsonify({'success': False, 'error': ERROR_PROCESSING_FAILED})

        h, w = img.shape[:2]
        s = min(720/w, 960/h)
        img = cv2.resize(img, (int(w*s), int(h*s)))

        # Check for face detection
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return jsonify({'success': False, 'error': ERROR_NO_FACE})

        active_image_data = img
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error in upload_selfie: {e}")  # Log server-side
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': ERROR_PROCESSING_FAILED})

@app.route('/api/products')
def gp(): return jsonify({'products': products_db})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)