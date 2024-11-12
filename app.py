from flask import Flask, request, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import pandas as pd
import numpy as np
import pickle
import os

# Initialize the app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'yoursecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:Peezy1009@localhost:3306/breast_cancer_app'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Load the model
MODEL_PATH = 'model/model_saved.pkl'
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

# User model for the database
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    email = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False)

class PredictionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    result = db.Column(db.String(500), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Home route
@app.route('/')
def home():
    return render_template('home.html')  # Serve the home page

# Breast cancer detection page (requires login)
@app.route('/index')
@login_required
def index():
    return render_template('index.html')

# Register route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Account created successfully', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()

        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Login unsuccessful. Check email and password', 'danger')
    return render_template('login.html')

# Logout route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))  # Redirect to the home page after logout

# Dashboard route, showing prediction history
@app.route('/dashboard')
@login_required
def dashboard():
    predictions = PredictionResult.query.filter_by(user_id=current_user.id).all()
    return render_template('dashboard.html', predictions=predictions)

# Manual input route for single prediction
@app.route('/manual_input')
@login_required
def manual_input():
    return render_template('manual_input.html')

# File upload route for batch prediction
@app.route('/file_upload')
@login_required
def file_upload():
    return render_template('file_upload.html')

# Prediction via file upload
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        data = pd.read_excel(file_path)
        predictions = model.predict(data)
        data['Prediction'] = predictions

        results = []
        for index, prediction in enumerate(predictions):
            result = 'Malignant (Cancerous)' if prediction == 0 else 'Benign (Non-cancerous)'
            results.append(f'Row {index + 1}: The tissue is {result}')

        result_text = '\n'.join(results)

        # Save the result in the database
        prediction_result = PredictionResult(user_id=current_user.id, result=result_text)
        db.session.add(prediction_result)
        db.session.commit()

        return render_template('result.html', result=result_text)

# Single prediction via manual input
@app.route('/single_predict', methods=['POST'])
@login_required
def single_predict():
    input_data = [
        request.form.get('mean_radius'), request.form.get('mean_texture'), request.form.get('mean_perimeter'),
        request.form.get('mean_area'), request.form.get('mean_smoothness'), request.form.get('mean_compactness'),
        request.form.get('mean_concavity'), request.form.get('mean_concave_points'), request.form.get('mean_symmetry'),
        request.form.get('mean_fractal_dimension'), request.form.get('radius_error'), request.form.get('texture_error'),
        request.form.get('perimeter_error'), request.form.get('area_error'), request.form.get('smoothness_error'),
        request.form.get('compactness_error'), request.form.get('concavity_error'),
        request.form.get('concave_points_error'), request.form.get('symmetry_error'), request.form.get('fractal_dimension_error'),
        request.form.get('worst_radius'), request.form.get('worst_texture'), request.form.get('worst_perimeter'),
        request.form.get('worst_area'), request.form.get('worst_smoothness'), request.form.get('worst_compactness'),
        request.form.get('worst_concavity'), request.form.get('worst_concave_points'), request.form.get('worst_symmetry'),
        request.form.get('worst_fractal_dimension')
    ]

    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    result = 'Malignant (Cancerous)' if prediction[0] == 0 else 'Benign (Non-cancerous)'

    prediction_result = PredictionResult(user_id=current_user.id, result=result)
    db.session.add(prediction_result)
    db.session.commit()

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
