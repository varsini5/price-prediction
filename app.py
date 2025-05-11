from flask import Flask, render_template, request, redirect, url_for, flash, session
import sqlite3
import pandas as pd
import numpy as np
import os
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ---------------------- DATABASE SETUP ----------------------
DB_NAME = "users.db"

def get_db_connection():
    conn = sqlite3.connect(DB_NAME, timeout=10, check_same_thread=False)  # Adds timeout and allows multithreading
    conn.row_factory = sqlite3.Row
    return conn

def create_users_table():
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        """)
        conn.commit()

create_users_table()

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash("All fields are required!", "danger")
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(password)
        
        try:
            with get_db_connection() as conn:
                conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
                conn.commit()
            flash("Registration successful! Please login.", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username already exists!", "danger")
    
    return render_template("register.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        with get_db_connection() as conn:
            user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()

        if user and check_password_hash(user["password"], password):
            session['username'] = user['username']
            return redirect(url_for('upload'))
        else:
            flash("Invalid username or password!", "danger")

    return render_template("login.html")

@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            session['uploaded_file'] = filepath
            flash("File uploaded successfully!", "success")
            return redirect(url_for('predict'))
        else:
            flash("Invalid file format! Please upload a CSV file.", "danger")

    return render_template("upload.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'username' not in session:
        flash("Please login first!", "danger")
        return redirect(url_for('login'))

    if 'uploaded_file' not in session:
        flash("No dataset uploaded!", "danger")
        return redirect(url_for('upload'))

    file_path = session['uploaded_file']
    df = pd.read_csv(file_path)

    if 'Achieved' not in df.columns:
        flash("Dataset must contain an 'Achieved' column!", "danger")
        return redirect(url_for('upload'))

    categorical_cols = ['origin', 'destination', 'shipping_method']
    existing_categorical_cols = [col for col in categorical_cols if col in df.columns]

    if existing_categorical_cols:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded_features = encoder.fit_transform(df[existing_categorical_cols])
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(existing_categorical_cols))
        df = df.drop(columns=existing_categorical_cols)
        df = pd.concat([df, encoded_df], axis=1)

    X = df.drop(columns=['Achieved'])
    y = df['Achieved']
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'RandomForest': RandomForestRegressor(),
        'GradientBoosting': GradientBoostingRegressor(),
        'SVR': SVR()
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        df[f'{name}_Predicted'] = model.predict(X)

    predictions_file = os.path.join('static', 'predictions.csv')
    df.to_csv(predictions_file, index=False)
    session['predictions_file'] = predictions_file

    # Fix unwanted \n in table output
    clean_table = df.head().to_html(classes='table table-striped', index=False, escape=False).replace('\n', '')

    flash("Predictions saved successfully!", "success")
    return render_template('predict.html', tables=[clean_table])

@app.route('/graphs')
def graphs():
    if 'username' not in session:
        flash("Please login first!", "danger")
        return redirect(url_for('login'))
    
    if 'predictions_file' not in session:
        flash("No predictions found!", "danger")
        return redirect(url_for('predict'))

    df = pd.read_csv(session['predictions_file'])
    plt.figure(figsize=(8, 6))
    for model in ['RandomForest', 'GradientBoosting', 'SVR']:
        sns.scatterplot(x=df['Achieved'], y=df[f'{model}_Predicted'], label=model)
    
    plt.xlabel("Actual Shipping Price")
    plt.ylabel("Predicted Shipping Price")
    plt.title("Actual vs Predicted Shipping Prices")
    plt.legend()
    
    graph_path = os.path.join('static', 'graph.png')
    plt.savefig(graph_path)
    plt.close()
    
    return render_template('graphs.html', graph_path=graph_path)

@app.route('/conclusion')
def conclusion():
    return render_template('conclusion.html')

if __name__ == "__main__":
    app.run(debug=True)
