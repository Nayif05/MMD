from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
from flask import redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
import bcrypt

app = Flask(__name__)

# Configuration for SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  # SQLite database
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = '4f5d4e5c6b7a8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4'  # Required for session management

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

    def __repr__(self):
        return f'<User {self.username}>'

# Create the database and tables
with app.app_context():
    db.create_all()

# Helper function to hash passwords
def hash_password(password):
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed

# Helper function to verify passwords
def verify_password(stored_password, provided_password):
    return bcrypt.checkpw(provided_password.encode('utf-8'), stored_password)

# Set the folder for uploaded files
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained CNN model
model = tf.keras.models.load_model('./alexnet.h5')
classes = {0: 'Parasitized', 1: 'Uninfected'}


from tensorflow.keras.preprocessing import image
from PIL import Image

def preprocess_image(img_path, target_size):
    """
    Load and preprocess the image for the model.
    """
    # Load the image using PIL
    img = Image.open(img_path)
    
    # Resize the image to the target size expected by the model
    img = img.resize(target_size)
    
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    
    # Add a batch dimension (CNN models expect inputs in batches)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

target_size = (100, 100)

def predict_image(image_path):
    """Preprocess image and make predictions using the model."""
    try:
        processed_image = preprocess_image(image_path, target_size)
        predictions = model.predict(processed_image)
    
    # Interpret the output
        if predictions.shape[1] > 1:  # Multi-class classification
            predicted_class = np.argmax(predictions, axis=1)[0]
        else:  # Binary classification
            predicted_class = (predictions > 0.5).astype(int)[0][0]
    
        # Display "paratized" instead of 0
        if predicted_class == 0:
            result = "Paratized"
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

        # Check if user already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash("Username already exists", "error")
            return redirect(url_for('signup'))

        # Hash the password
        hashed_password = hash_password(password)

        # Create new user
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

        # Find the user in the database
        user = User.query.filter_by(username=username).first()
        if not user:
            flash("User not found", "error")
            return redirect(url_for('login'))

        # Verify the password
        if not verify_password(user.password, password):
            flash("Invalid password", "error")
            return redirect(url_for('login'))

        # Store user in session
        session['username'] = username
        flash("Login successful!", "success")
        return redirect(url_for('landing_page'))

    return render_template('login.html')

# Logout Route
@app.route('/logout')
def logout():
    session.pop('username', None)  # Clear the username from the session
    flash("You have been logged out", "success")  # Optional: Flash a message
    return redirect(url_for('landing_page'))  # Redirect to the landing page

# Protect routes that require authentication
@app.before_request
def require_login():
    allowed_routes = ['login', 'signup', 'landing_page', 'static']
    if request.endpoint not in allowed_routes and 'username' not in session:
        flash("You need to login first", "error")
        return redirect(url_for('login'))

# Existing routes for image prediction
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
            return render_template("result.html", prediction=prediction, certainty=certainty, image_path=file_path)
    return render_template("form.html")

@app.route("/team")
def team():
    return render_template("team.html")

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html")

if __name__ == "__main__":
    app.run(debug=True)