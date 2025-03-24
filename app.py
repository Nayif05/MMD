from flask import Flask, request, jsonify, send_from_directory, render_template, redirect, url_for, flash, session
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import jwt
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
from functools import wraps
import uuid

import google.generativeai as genai
# Add this with your other imports

# Configure Gemini API
genai.configure(api_key='AIzaSyAT0HpKPtW3wqQ8a7m5kG9Z0ju6X6PvQzI')
best_model = "models/gemini-1.5-pro-latest"
backup_model = "models/gemini-1.5-flash-latest"

# Initialize Flask app
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# Configure PostgreSQL database
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'postgresql://postgres:Nayifcm%402003@localhost:5432/MMD')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = '4f5d4e5c6b7a8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4'

app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    results = db.relationship('DetectionResult', backref='user', lazy=True)

class DetectionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    result = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'image_path': self.image_path,
            'result': self.result,
            'confidence': self.confidence,
            'created_at': self.created_at  # Keep it as a datetime object
        }

# Helper functions
def hash_password(password):
    return generate_password_hash(password)

def verify_password(stored_password, provided_password):
    return check_password_hash(stored_password, provided_password)

# Load the pre-trained CNN model
try:
    model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), 'alexnet.h5'))
    classes = {0: 'Parasitized', 1: 'Uninfected'}
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    # For demo purposes or if model file doesn't exist yet
    model = None

# Authentication decorator for logged-in users
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

def preprocess_image(img_path, target_size=(100, 100)):
    """
    Load and preprocess the image for the model.
    """
    img = Image.open(img_path)
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(image_path):
    """Preprocess image and make predictions using the model."""
    try:
        if model is None:
            print(f"Error processing image: {e}")
            return "Error", 0
            
        processed_image = preprocess_image(image_path)
        predictions = model.predict(processed_image)
    
        if predictions.shape[1] > 1:  # Multi-class classification
            predicted_class = np.argmax(predictions, axis=1)[0]
        else:  # Binary classification
            predicted_class = (predictions > 0.5).astype(int)[0][0]
    
        if predicted_class == 0:
            result = "Parasitized"
            certainty = 100 - np.amax(predictions) * 100
        else:
            result = "Uninfected"
            certainty = np.amax(predictions) * 100
        return result, float(certainty)
    except Exception as e:
        print(f"Error processing image: {e}")
        return "Error", 0

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        # Validate required fields
        if not email or not password:
            flash('Email and password are required', 'error')
            return render_template('login.html')
        
        user = User.query.filter_by(email=email).first()
        
        if not user or not verify_password(user.password, password):
            flash('Invalid email or password', 'error')
            return render_template('login.html')
        
        # Store user info in session
        session['user_id'] = user.id
        session['first_name'] = user.first_name
        session['last_name'] = user.last_name
        session['email'] = user.email
        
        flash('Login successful!', 'success')
        return redirect(url_for('dashboard'))
    
    return render_template('login.html')

# Create database tables within app context
with app.app_context():
    db.create_all()

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        
        # Validate required fields
        if not email or not password or not first_name or not last_name:
            flash('All fields are required', 'error')
            return render_template('signup.html')
        
        # Check if user already exists
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'error')
            return render_template('signup.html')
        
        # Create new user
        hashed_password = hash_password(password)
        new_user = User(
            email=email,
            password=hashed_password,
            first_name=first_name,
            last_name=last_name
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'success')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    user_id = session['user_id']
    results = DetectionResult.query.filter_by(user_id=user_id).order_by(DetectionResult.created_at.desc()).all()
    results_list = [result.to_dict() for result in results]  # Convert to dictionaries
    return render_template('dashboard.html', results=results_list)


@app.route('/delete_result/<int:result_id>', methods=['POST'])
@login_required
def delete_result(result_id):
    user_id = session['user_id']
    result = DetectionResult.query.filter_by(id=result_id, user_id=user_id).first()
    
    if not result:
        flash('Result not found or you do not have permission to delete it', 'error')
        return redirect(url_for('dashboard'))
    
    # Delete the image file if it exists
    if result.image_path:
        image_path = os.path.join(app.static_folder, result.image_path.lstrip('/'))
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception as e:
                print(f"Error deleting image file: {e}")
    
    # Delete the result from the database
    db.session.delete(result)
    db.session.commit()
    
    flash('Result deleted successfully', 'success')
    return redirect(url_for('dashboard'))

@app.route('/detect', methods=['GET', 'POST'])
@login_required
def detect():
    if request.method == 'POST':
        # Check if image is provided
        if 'image' not in request.files:
            flash('No image provided', 'error')
            return redirect(request.url)
        
        file = request.files['image']
        
        if file.filename == '':
            flash('No image selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Generate a unique filename
            unique_filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            filename = secure_filename(unique_filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Make prediction
            result, confidence = predict_image(file_path)
            
            # Save result to database
            detection_result = DetectionResult(
                user_id=session['user_id'],
                image_path=f"/uploads/{filename}",
                result=result,
                confidence=confidence
            )
            db.session.add(detection_result)
            db.session.commit()
            
            # After saving the detection result, get AI feedback
            ai_feedback = get_ai_description(result, confidence)
            
            return render_template('result.html', 
                                result=result, 
                                confidence=confidence, 
                                image_path=f"/uploads/{filename}",
                                ai_feedback=ai_feedback)
        
        flash('File type not allowed', 'error')
        return redirect(request.url)
    
    return render_template('detect.html')

def get_ai_description(result, confidence):
    """Generate concise AI feedback based on detection result."""
    if result == "Parasitized":
        prompt = f"""
        This blood sample shows malaria parasites with {confidence:.2f}% confidence.
        Provide medical feedback as a numbered list with these exact items:
        
        1. Immediate Action Required: [Your one-sentence instruction]
        2. Likely Treatments: [Your one-sentence description]
        3. While Waiting for Doctor: [Your one-sentence advice] 
        4. Potential Risks if Untreated: [Your one-sentence warning]
        
        Rules:
        - Each item must start with the number and exact heading shown
        - Keep each response after the colon extremely brief (max 10 words)
        - Do not use any asterisks, bullet points, or HTML tags
        - Each item must be on its own line
        - Never include 'html' in the response
        """
    else:
        prompt = f"""
        This blood sample shows no malaria parasites with {confidence:.2f}% confidence.
        Provide prevention advice as a numbered list with these exact items:
        
        1. Prevention Recommendations: [Your one-sentence advice]
        2. Early Symptoms to Watch: [Your one-sentence list] 
        3. When to Retest: [Your one-sentence guidance]
        
        Rules:
        - Each item must start with the number and exact heading shown
        - Keep each response after the colon extremely brief (max 10 words)
        - Do not use any asterisks, bullet points, or HTML tags
        - Each item must be on its own line
        - Never include 'html' in the response

        """

    for model in [best_model, backup_model]:
        try:
            response = genai.GenerativeModel(model_name=model).generate_content(prompt)
            if response and response.text:
                # Ensure the response is properly formatted
                feedback = response.text.replace('**', '')  # Remove any remaining asterisks
                return feedback
        except Exception as e:
            if "model_not_found" in str(e) or "quota_exceeded" in str(e):
                continue  # Try the next model
            else:
                print(f"Gemini API error: {e}")
                return "<strong>AI feedback unavailable</strong>. Please consult a doctor."

    return "<strong>AI feedback unavailable</strong>. Please consult a doctor."

@app.route('/results', methods=['GET'])
@login_required
def get_results():
    user_id = session['user_id']
    results = DetectionResult.query.filter_by(user_id=user_id).order_by(DetectionResult.created_at.desc()).all()
    return jsonify([result.to_dict() for result in results])

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(debug=True)
