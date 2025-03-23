from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash, make_response, send_file
import os
import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import bcrypt
from datetime import datetime

app = Flask(__name__)

# Configuration for SQLAlchemy with PostgreSQL
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'postgresql://postgres:Nayifcm%402003@localhost:5432/MMD')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = '4f5d4e5c6b7a8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4'

# Initialize SQLAlchemy
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)  # Increase from 120 to 255

    def __repr__(self):
        return f'<User {self.username}>'

# Prediction Model
class Prediction(db.Model):
    __tablename__ = 'predictions'
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(200), nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    certainty = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Prediction {self.image_path}>'

# Helper function to hash passwords
def hash_password(password):
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')  # Convert bytes to string before storing

# Helper function to verify passwords
def verify_password(stored_password, provided_password):
    return bcrypt.checkpw(provided_password.encode('utf-8'), stored_password.encode('utf-8'))

# Set the folder for uploaded files
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained CNN model
model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), 'alexnet.h5'))
classes = {0: 'Parasitized', 1: 'Uninfected'}

def preprocess_image(img_path, target_size):
    """
    Load and preprocess the image for the model.
    """
    img = Image.open(img_path)
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

target_size = (100, 100)

def predict_image(image_path):
    """Preprocess image and make predictions using the model."""
    try:
        processed_image = preprocess_image(image_path, target_size)
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
        return result, certainty
    except Exception as e:
        print(f"Error processing image: {e}")
        return "Error", 0

# Signup Route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if not username or not password:
            flash("Username and password are required", "error")
            return redirect(url_for('signup'))
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash("Username already exists", "error")
            return redirect(url_for('signup'))
        hashed_password = hash_password(password)
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash("Signup successful! Please login.", "success")
        return redirect(url_for('login'))
    return render_template('signup.html')

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if not username or not password:
            flash("Username and password are required", "error")
            return redirect(url_for('login'))
        user = User.query.filter_by(username=username).first()
        if not user:
            flash("User not found", "error")
            return redirect(url_for('login'))
        if not verify_password(user.password, password):
            flash("Invalid password", "error")
            return redirect(url_for('login'))
        session['username'] = username
        flash("Login successful!", "success")
        return redirect(url_for('landing_page'))
    return render_template('login.html')

# Logout Route
@app.route('/logout')
def logout():
    session.pop('username', None)
    flash("You have been logged out", "success")
    return redirect(url_for('landing_page'))

# Protect routes that require authentication
@app.before_request
def require_login():
    allowed_routes = ['login', 'signup', 'landing_page', 'static']
    if request.endpoint not in allowed_routes and 'username' not in session:
        flash("You need to login first", "error")
        return redirect(url_for('login'))

# Routes
@app.route("/")
def landing_page():
    return render_template("index.html")

@app.route("/form", methods=["GET", "POST"])
def input_form():
    if request.method == "POST":
        if 'file' not in request.files:
            flash("No file part", "error")
            return redirect(url_for('input_form'))
        file = request.files['file']
        if file.filename == '':
            flash("No selected file", "error")
            return redirect(url_for('input_form'))
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            file.save(file_path)

            prediction, certainty = predict_image(file_path)

            # Save prediction to the database
            new_prediction = Prediction(
                image_path=file_path,
                prediction=prediction,
                certainty=float(certainty)
            )

            db.session.add(new_prediction)
            db.session.commit()

            return render_template("result.html", prediction=prediction, certainty=certainty, image_path=file_path)
    return render_template("form.html")

@app.route("/dashboard")
def dashboard():
    records = Prediction.query.order_by(Prediction.timestamp.desc()).all()
    return render_template("dashboard.html", records=records)

@app.route("/delete/<int:id>", methods=["POST"])
def delete_prediction(id):
    if 'username' not in session:
        flash("You need to login first", "error")
        return redirect(url_for('login'))

    # Find the prediction record by ID
    prediction = Prediction.query.get_or_404(id)

    # Delete the associated image file (optional)
    if os.path.exists(prediction.image_path):
        os.remove(prediction.image_path)

    # Delete the record from the database
    db.session.delete(prediction)
    db.session.commit()

    flash("Record deleted successfully", "success")
    return redirect(url_for('dashboard'))

@app.route("/team")
def team():
    return render_template("team.html")

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404

# Create database tables
with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True)
