import csv
import os
import re
import json
import pandas as pd
import pdfplumber

# 🔒 SAFE GEMINI IMPORT (NO CRASH)
try:
    import google.generativeai as genai
except Exception:
    genai = None

from docx import Document
from PIL import Image
import io
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import IntegrityError
from datetime import date, datetime, timedelta

# ==========================================
# CONFIGURATION
# ==========================================
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'png', 'jpg', 'jpeg', 'webp'}

# ==========================================
# GEMINI CONFIGURATION (SAFE)
# ==========================================
initial_api_key = os.environ.get("GOOGLE_API_KEY")
model = None

if genai and initial_api_key:
    try:
        genai.configure(api_key=initial_api_key)
        model = genai.GenerativeModel("gemini-pro")
    except Exception as e:
        print("Gemini init failed:", e)
        model = None

# ==========================================
# FLASK APP
# ==========================================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ==========================================
# DATABASE CONFIGURATION (RENDER SAFE)
# ==========================================
database_url = os.environ.get("DATABASE_URL")
if not database_url:
    raise RuntimeError("DATABASE_URL not set. Configure PostgreSQL on Render.")

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_recycle': 280,
    'pool_pre_ping': True
}

db = SQLAlchemy(app)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Ensure tables exist
try:
    with app.app_context():
        db.create_all()
except Exception as e:
    print("DB init warning:", e)

# ==========================================
# DATABASE MODEL
# ==========================================
class Candidate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(150))
    name = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True, index=True)
    phone = db.Column(db.String(50), unique=True, index=True)
    college = db.Column(db.String(200))
    degree = db.Column(db.String(100))
    department = db.Column(db.String(100))
    state = db.Column(db.String(50))
    district = db.Column(db.String(50))
    year_passing = db.Column(db.String(20))
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)

# ==========================================
# EXTRACTION LOGIC (UNCHANGED)
# ==========================================
def extract_text_traditional(file_path):
    ext = file_path.rsplit('.', 1)[1].lower()
    text = ""
    try:
        if ext == 'pdf':
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += (page.extract_text() or "") + "\n"
        elif ext == 'docx':
            doc = Document(file_path)
            text = "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        print(f"Error reading document: {e}")
    return text

def extract_name(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in lines[:10]:
        if re.match(r'^[A-Z][A-Z\s\.]{2,}$', line): return line.title()
    for line in lines[:10]:
        if re.match(r'^[A-Z]\.?( )?[A-Z][a-zA-Z]+$', line): return line.title()
    for line in lines[:10]:
        if re.match(r'^[A-Z][a-zA-Z]+ [A-Z][a-zA-Z]+$', line): return line.strip()
    return "Unknown"

def extract_email(text):
    m = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return m.group(0) if m else "Not Specified"

def extract_phone(text):
    m = re.search(r'\b(?:\+?91)?\s*\d{10}\b', text)
    return m.group(0) if m else "Not Specified"

def extract_college(text):
    m = re.search(r'([A-Za-z ]+(University|Institute|College))', text, re.IGNORECASE)
    return m.group(0).strip() if m else "Not Specified"

def extract_degree(text):
    patterns = [
        r'b\.?tech', r'b\.?e', r'm\.?tech', r'm\.?e',
        r'bachelor(?: of)?', r'master(?: of)?', r'b\.?sc',
        r'm\.?sc', r'diploma', r'ph\.?d'
    ]
    for p in patterns:
        match = re.search(p, text, re.IGNORECASE)
        if match:
            return match.group(0).upper()
    return "Not Specified"

def extract_department(text):
    patterns = [
        "computer science", "information technology",
        "electronics", "mechanical", "civil",
        "artificial intelligence", "data science"
    ]
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return p.title()
    return "Not Specified"

def extract_state(text):
    states = ["Tamil Nadu", "Kerala", "Karnataka", "Andhra Pradesh", "Telangana"]
    for s in states:
        if re.search(r'\b' + s + r'\b', text, re.IGNORECASE):
            return s
    return "Not Specified"

def extract_district(text):
    districts = ["Chennai", "Coimbatore", "Madurai", "Salem"]
    for d in districts:
        if re.search(r'\b' + d + r'\b', text, re.IGNORECASE):
            return d
    return "Not Specified"

def extract_year_of_passing(text):
    years = re.findall(r'\b(20\d{2})\b', text)
    years = [int(y) for y in years if 2000 <= int(y) <= 2035]
    return str(max(years)) if years else "Not Specified"

# ==========================================
# GEMINI IMAGE EXTRACTION (SAFE)
# ==========================================
def extract_data_with_gemini(file_path):
    if not model:
        return {
            "Name": "Not Specified",
            "Phone": "Not Specified",
            "Email": "Not Specified",
            "College": "Not Specified",
            "Degree": "Not Specified",
            "Department": "Not Specified",
            "District": "Not Specified",
            "State": "Not Specified",
            "Passed Out": "Not Specified"
        }

    img = Image.open(file_path)
    prompt = """
    Extract resume details.
    Return ONLY JSON.
    Fields: Name, Phone, Email, College, Degree, Department, District, State, Passed Out.
    """
    try:
        response = model.generate_content([prompt, img])
        clean = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception as e:
        print("Gemini error:", e)
        return None

# ==========================================
# ROUTES (UNCHANGED)
# ==========================================
@app.route('/')
def index():
    return render_template('index.html', extracted_data=None)

@app.route('/set_api_key', methods=['POST'])
def set_api_key():
    if not genai:
        return jsonify({'success': False, 'message': 'Gemini SDK not installed'}), 500

    data = request.get_json()
    key = data.get('api_key')
    if not key:
        return jsonify({'success': False, 'message': 'No API key'}), 400

    genai.configure(api_key=key)
    global model
    model = genai.GenerativeModel("gemini-pro")
    return jsonify({'success': True})

@app.route('/init-db')
def init_db():
    with app.app_context():
        db.create_all()
    return "Database initialized!"

if __name__ == '__main__':
    app.run(debug=True)
