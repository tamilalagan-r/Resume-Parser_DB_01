import csv
import os
import re
import json
import traceback
import pandas as pd
import pdfplumber
import google.generativeai as genai
from docx import Document
from PIL import Image
import io
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import IntegrityError
from datetime import date, datetime, timedelta
from urllib.parse import quote_plus
from concurrent.futures import ThreadPoolExecutor
from threading import Semaphore

executor = ThreadPoolExecutor(max_workers=3)  # 3 files at a time
gemini_semaphore = Semaphore(2)               # 2 Gemini calls max

      
# ==========================================
# CONFIGURATION
# ==========================================
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'png', 'jpg', 'jpeg', 'webp'}

# API Configuration (can be set via environment variable or via the UI)
# Prefer setting GOOGLE_API_KEY in environment for production. The app also supports
# dynamically setting the key from the browser (modal input) which will configure
# the Gemini client for the running server process.
initial_api_key = os.environ.get("GOOGLE_API_KEY")
if initial_api_key:
    genai.configure(api_key=initial_api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
else:
    model = None

app = Flask(__name__) 
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
app.config['DEBUG'] = True
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1 GB
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ==========================================
# DATABASE CONFIGURATION
# ==========================================
database_url = os.environ.get('DATABASE_URL')
if not database_url:
    DB_USER = "root"
    DB_PASS = "MySql@1234" 
    DB_HOST = "127.0.0.1:3306"
    DB_NAME = "resume_db"
    encoded_pass = quote_plus(DB_PASS)
    database_url = f"mysql+pymysql://{DB_USER}:{encoded_pass}@{DB_HOST}/{DB_NAME}"

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'pool_recycle': 280, 'pool_pre_ping': True}

db = SQLAlchemy(app)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# Ensure DB tables exist on startup (best-effort). If creation fails, log and continue.
try:
    with app.app_context():
        db.create_all()
except Exception as e:
    print("Warning: could not create database tables on startup:", e)

# ==========================================
# DATABASE MODEL
# ==========================================
class Candidate(db.Model):
    __tablename__ = "candidate"

    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255))
    name = db.Column(db.String(100))
    email = db.Column(db.String(120), unique=True)
    phone = db.Column(db.String(20))
    college = db.Column(db.String(150))
    degree = db.Column(db.String(100))
    department = db.Column(db.String(100))
    state = db.Column(db.String(100))
    district = db.Column(db.String(100))
    year_passing = db.Column(db.Integer)
    upload_date = db.Column(db.DateTime, nullable=True)

@app.errorhandler(Exception)
def handle_exception(e):
    print("🔥 INTERNAL ERROR 🔥")
    traceback.print_exc()
    return "Internal Server Error - check Render logs", 500

# ==========================================
# EXTRACTION LOGIC
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
    patterns = [r'b\.?tech', r'b\.?e', r'm\.?tech', r'm\.?e', r'bachelor(?: of)?', r'master(?: of)?',
            r'b\.?sc', r'm\.?sc', r'b\.?a', r'm\.?a', r'b\.?com', r'm\.?com', r'b\.?ba', r'm\.?ba',
            r'b\.?ca', r'm\.?ca', r'b\.?ed', r'm\.?ed', r'b\.?pharm', r'm\.?pharm', r'b\.?arch', r'm\.?arch',
            r'b\.?ds', r'm\.?ds', r'mbbs', r'bams', r'bhms', r'b\.?voc', r'm\.?voc', r'diploma', r'pg diploma',
            r'ph\.?d', r'doctorate']
    for p in patterns:
        match = re.search(p, text, re.IGNORECASE)
        if match: return match.group(0).upper()
    return "Not Specified"

def extract_department(text):
    txt_lower = text.lower()
    edu_section = text
    keywords = ["education", "academic details", "qualification", "educational qualification"]
    for key in keywords:
        if key in txt_lower:
            start = txt_lower.index(key)
            edu_section = text[start:start + 1000]
            break          
    patterns = [
        r"electronics and communication", r"ece", r"computer science", r"cs", r"cse",
        r"electrical and electronics", r"eee", r"mechanical engineering", r"mech", r"civil engineering", r"civil",
        r"artificial intelligence and data science", r"ai&ds", r"data science", r"data analytics",
        r"artificial intelligence", r"ai", r"cyber security", r"information technology", r"it",
        r"physics", r"chemistry", r"biology", r"biotechnology", r"mathematics", r"statistics",
        r"accounting", r"finance", r"banking", r"bba", r"marketing", r"hr",
        r"english", r"history", r"computer applications", r"ca", r"pharmacy", r"law"
    ]
    for p in patterns:
        if re.search(p, edu_section, re.IGNORECASE): return p.title()
    return "Not Specified"

def extract_state(text):
    states = ["Tamil Nadu", "Tamilnadu", "Kerala", "Karnataka", "Andhra Pradesh", "Telangana", "Maharashtra", "Delhi"]
    for state in states:
        if re.search(r'\b' + re.escape(state) + r'\b', text, re.IGNORECASE): return state.title()
    return "Not Specified"

def extract_district(text):
    districts = [r"chennai", r"coimbatore", r"madurai", r"trichy", r"salem", r"tirunelveli", r"erode", r"vellore", r"thoothukudi", r"dindigul", r"thanjavur", r"tiruppur", r"virudhunagar", r"karur", r"nilgiris", r"krishnagiri", r"kanyakumari", r"kancheepuram", r"namakkal", r"sivagangai", r"cuddalore", r"pudukkottai", r"theni", r"ramanathapuram", r"thiruvarur", r"thiruvallur", r"tiruvannamalai", r"nagapattinam", r"viluppuram", r"perambalur", r"dharmapuri", r"ariyalur", r"tirupathur", r"tenkasi", r"chengalpattu", r"kallakurichi", r"ranipet", r"mayiladuthurai"]
    for dist in districts:
        if re.search(r'\b' + re.escape(dist) + r'\b', text, re.IGNORECASE): return dist.title()
    return "Not Specified"

def extract_year_of_passing(text):
    pattern_range = r'(20\d{2})\s*[\-\–]\s*(\d{2,4})'
    match_range = re.search(pattern_range, text)
    if match_range:
        end_year = match_range.group(2)
        if len(end_year) == 2: return "20" + end_year 
        return end_year
    matches = re.findall(r'\b(20\d{2})\b', text)
    valid_years = [int(y) for y in matches if 2000 <= int(y) <= 2030]
    if valid_years: return str(max(valid_years))
    return "Not Specified"


# -----------------------------
# Normalization helpers
# -----------------------------
def normalize_email(email):
    if not email: return None
    email = str(email).strip()
    if not email or email.lower() == 'not specified': return None
    return email.lower()


def normalize_phone(phone):
    if not phone: return None
    phone = str(phone).strip()
    if not phone or phone.lower() == 'not specified': return None
    digits = re.sub(r'\D', '', phone)
    if not digits: return None
    # prefer last 10 digits (handles country codes)
    if len(digits) >= 10:
        return digits[-10:]
    return digits


def parse_with_regex(filepath):
    raw_text = extract_text_traditional(filepath)
    raw_text = raw_text.replace("■", "").replace("●", "")
    return {
        "Name": extract_name(raw_text),
        "Email": extract_email(raw_text),
        "Phone": extract_phone(raw_text),
        "College": extract_college(raw_text),
        "Degree": extract_degree(raw_text),
        "Department": extract_department(raw_text),
        "Passed Out": extract_year_of_passing(raw_text),
        "State": extract_state(raw_text),
        "District": extract_district(raw_text)
    }

def extract_data_with_gemini(file_path):
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        if model is None:
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

        try:
            with gemini_semaphore:   # 🔒 LIMIT CALLS
                img = Image.open(file_path)
                prompt = """
                You are an expert Resume Parser.
                Return ONLY valid JSON.
                Fields: Name, Phone, Email, College, Degree, Department, District, State, Passed Out.
                """

                response = model.generate_content([prompt, img])
                clean = response.text.strip().replace("```json", "").replace("```", "")
                return json.loads(clean)

        except Exception as e:
            print("Gemini error:", e)
            return None

def process_and_save(filepath, filename):
    try:
        ext = filename.rsplit('.', 1)[1].lower()

        if ext in ['pdf', 'docx']:
            data = parse_with_regex(filepath)
        else:
            data = extract_data_with_gemini(filepath)

        if not data:
            return

        name_val = data.get('Name') or 'Unknown'
        email_val = normalize_email(data.get('Email'))
        phone_val = normalize_phone(data.get('Phone'))

        existing = None
        if email_val:
            existing = Candidate.query.filter_by(email=email_val).first()
        if not existing and phone_val:
            existing = Candidate.query.filter_by(phone=phone_val).first()

        if existing:
            existing.filename = filename
            existing.name = name_val
            existing.email = email_val
            existing.phone = phone_val
            existing.college = data.get('College')
            existing.degree = data.get('Degree')
            existing.department = data.get('Department')
            existing.year_passing = data.get('Passed Out')
            existing.state = data.get('State')
            existing.district = data.get('District')
            existing.upload_date = datetime.utcnow()
        else:
            db.session.add(Candidate(
                filename=filename,
                name=name_val,
                email=email_val,
                phone=phone_val,
                college=data.get('College'),
                degree=data.get('Degree'),
                department=data.get('Department'),
                year_passing=data.get('Passed Out'),
                state=data.get('State'),
                district=data.get('District'),
                upload_date=datetime.utcnow()
            ))

        db.session.commit()

    except Exception as e:
        db.session.rollback()
        print("Background error:", filename, e)

# ==========================================
# ROUTES
# ==========================================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html', extracted_data=None)

# 1. UPDATED UPLOAD ROUTE (Handle Multiple Files)
@app.route('/upload', methods=['POST'])
def upload_file():

    # -------- Folder path upload --------
    folder_path = request.form.get('folder_path')
    if folder_path:
        if not os.path.isdir(folder_path):
            return "Invalid folder path", 400

        for root, _, files in os.walk(folder_path):
            for fname in files:
                if allowed_file(fname):
                    src = os.path.join(root, fname)
                    dst = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(fname))
                    if not os.path.exists(dst):
                        with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
                            fdst.write(fsrc.read())
                    executor.submit(process_and_save, dst, fname)

        return redirect(url_for('dashboard'))

    # -------- File input upload --------
    if 'file' not in request.files:
        return redirect(request.url)

    files = request.files.getlist('file')

    for file in files:
        if not file or file.filename == '':
            continue
        if not allowed_file(file.filename):
            continue

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # background processing
        def safe_process(filepath, filename):
            with app.app_context():
                process_and_save(filepath, filename)
        
        
        executor.submit(safe_process, filepath, filename)

    return redirect(url_for('dashboard'))

    # If MULTIPLE files -> Process ALL and save to DB directly (Bulk Upload)
    if len(files) > 1:
        count = 0
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                ext = filename.rsplit('.', 1)[1].lower()
                data = None

                # Extract
                if ext in ['pdf', 'docx']:
                    data = parse_with_regex(filepath)
                elif ext in ['png', 'jpg', 'jpeg', 'webp']:
                    data = extract_data_with_gemini(filepath)

                # Save to DB immediately (deduplicate by email or phone)
                if data:
                    name_val = data.get('Name') or 'Unknown'
                    raw_email = data.get('Email')
                    raw_phone = data.get('Phone')
                    email_val = normalize_email(raw_email)
                    phone_val = normalize_phone(raw_phone)

                    existing = None
                    if email_val:
                        existing = Candidate.query.filter_by(email=email_val).first()
                    if not existing and phone_val:
                        existing = Candidate.query.filter_by(phone=phone_val).first()
                    if existing:
                        # update existing record
                        existing.filename = filename or existing.filename
                        existing.name = name_val or existing.name
                        existing.email = email_val or existing.email
                        existing.phone = phone_val or existing.phone
                        existing.college = data.get('College') or existing.college
                        existing.degree = data.get('Degree') or existing.degree
                        existing.department = data.get('Department') or existing.department
                        existing.year_passing = data.get('Passed Out') or existing.year_passing
                        existing.state = data.get('State') or existing.state
                        existing.district = data.get('District') or existing.district
                        existing.upload_date = datetime.utcnow()
                        db.session.add(existing)
                        count += 1
                    else:
                        new_candidate = Candidate(
                            filename=filename,
                            name=name_val,
                            email=email_val,
                            phone=phone_val,
                            college=data.get('College'),
                            degree=data.get('Degree'),
                            department=data.get('Department'),
                            year_passing=data.get('Passed Out'),
                            state=data.get('State'),
                            district=data.get('District')
                        )
                        db.session.add(new_candidate)
                        count += 1

        if count % 5 == 0:
            db.session.commit()
        return redirect(url_for('dashboard'))

    # If SINGLE file -> Keep original Review/Edit Logic
    elif len(files) == 1:
        file = files[0]
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            ext = filename.rsplit('.', 1)[1].lower()
            data = None

            if ext in ['pdf', 'docx']:
                data = parse_with_regex(filepath)
            elif ext in ['png', 'jpg', 'jpeg', 'webp']:
                data = extract_data_with_gemini(filepath)

            if data:
                data['filename'] = filename
                return render_template('index.html', extracted_data=data)

    return redirect(url_for('index'))

# 2. SAVE OR UPDATE CANDIDATE
@app.route('/save_candidate', methods=['POST'])
def save_candidate():
    try:
        candidate_id = request.form.get('candidate_id')
        
        filename = request.form.get('filename')
        name = request.form.get('name')
        raw_email = request.form.get('email')
        raw_phone = request.form.get('phone')
        college = request.form.get('college')
        degree = request.form.get('degree')
        department = request.form.get('department')
        year_passing = request.form.get('year_passing')
        state = request.form.get('state')
        district = request.form.get('district')

        # Normalize keys used for deduplication
        email = normalize_email(raw_email)
        phone = normalize_phone(raw_phone)

        candidate = None

        if candidate_id:
            candidate = Candidate.query.get(candidate_id)
        
        if not candidate:
            if email:
                candidate = Candidate.query.filter_by(email=email).first()
            if not candidate and phone:
                candidate = Candidate.query.filter_by(phone=phone).first()

        if candidate:
            candidate.filename = filename
            candidate.name = name
            candidate.email = email
            candidate.phone = phone
            candidate.college = college
            candidate.degree = degree
            candidate.department = department
            candidate.year_passing = year_passing
            candidate.state = state
            candidate.district = district
            candidate.upload_date = datetime.utcnow()
        else:
            candidate = Candidate(
                filename=filename, name=name, email=email, phone=phone,
                college=college, degree=degree, department=department,
                year_passing=year_passing, state=state, district=district
            )
            db.session.add(candidate)

        try:
            db.session.commit()
        except IntegrityError:
            # Handle race condition or unique constraint violation by merging into existing
            db.session.rollback()
            existing = None
            if email:
                existing = Candidate.query.filter_by(email=email).first()
            if not existing and phone:
                existing = Candidate.query.filter_by(phone=phone).first()
            if existing:
                existing.filename = filename or existing.filename
                existing.name = name or existing.name
                existing.email = email or existing.email
                existing.phone = phone or existing.phone
                existing.college = college or existing.college
                existing.degree = degree or existing.degree
                existing.department = department or existing.department
                existing.year_passing = year_passing or existing.year_passing
                existing.state = state or existing.state
                existing.district = district or existing.district
                existing.upload_date = datetime.utcnow()
                db.session.add(existing)
                db.session.commit()

        return redirect(url_for('dashboard'))

    except Exception as e:
        db.session.rollback()
        return f"Database Error: {e}"

@app.route('/dashboard')
def dashboard():
    search_query = request.args.get('search')
    query = Candidate.query

    if search_query:
        search_term = f"%{search_query}%"
        query = query.filter(
            db.or_(
                Candidate.name.ilike(search_term),
                Candidate.email.ilike(search_term),
                Candidate.phone.ilike(search_term)
            )
        )

    candidates = query.order_by(Candidate.upload_date.desc()).all()
    return render_template('dashboard.html', candidates=candidates, search_query=search_query)

# Endpoint to set the Google Generative AI API key from the frontend
@app.route('/set_api_key', methods=['POST'])
def set_api_key():
    try:
        data = request.get_json() or {}
        key = data.get('api_key')
        if not key:
            return jsonify({'success': False, 'message': 'No API key provided'}), 400
        # Configure the global client
        genai.configure(api_key=key)
        global model
        model = genai.GenerativeModel('gemini-2.5-flash')
        return jsonify({'success': True})
    except Exception as e:
        print('Error setting API key:', e)
        return jsonify({'success': False, 'message': str(e)}), 500

# 3. EDIT ROUTE
@app.route('/edit/<int:id>')
def edit_candidate(id):
    candidate = Candidate.query.get_or_404(id)
    data = {
        'id': candidate.id,
        'filename': candidate.filename,
        'Name': candidate.name,
        'Email': candidate.email,
        'Phone': candidate.phone,
        'College': candidate.college,
        'Degree': candidate.degree,
        'Department': candidate.department,
        'Passed Out': candidate.year_passing,
        'State': candidate.state,
        'District': candidate.district
    }
    return render_template('index.html', extracted_data=data)

# 4. BULK DELETE ROUTE
@app.route('/delete_bulk', methods=['POST'])
def delete_bulk():
    ids = request.form.getlist('selected_ids')
    if ids:
        try:
            Candidate.query.filter(Candidate.id.in_(ids)).delete(synchronize_session=False)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            print("Error deleting:", e)
    return redirect(url_for('dashboard'))

# 5. UPDATED EXPORT ROUTES (With Date Filters)
def get_filtered_query():
    query = Candidate.query
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    if start_date and end_date:
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d').replace(hour=23, minute=59, second=59)
            query = query.filter(Candidate.upload_date >= start, Candidate.upload_date <= end)
        except ValueError:
            pass # Handle invalid date format if necessary
            
    return query.all()

@app.route('/export/json')
def export_json():
    candidates = get_filtered_query()
    filename = "Resume_Data_Export.json"
    
    data = [{"Name": c.name, "Contact": c.phone, "Email": c.email, 
             "Degree": c.degree, "Department": c.department, "College": c.college, 
             "State": c.state, "District": c.district, "Passed Out": c.year_passing, 
             "File Name": c.filename} for c in candidates]
    
    json_str = json.dumps(data, indent=4)
    buf = io.BytesIO(json_str.encode('utf-8'))
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=filename, mimetype='application/json')

@app.route('/export/csv')
def export_csv():
    candidates = get_filtered_query()
    filename = "Resume_Data_Export.csv"
    
    data = [{"Name": c.name, "Contact": c.phone, "Email": c.email, 
             "Degree": c.degree, "Department": c.department, "College": c.college, 
             "State": c.state, "District": c.district, "Passed Out": c.year_passing, 
             "File Name": c.filename} for c in candidates]
    
    str_buf = io.StringIO()
    fieldnames = ["Name", "Contact", "Email", "Degree", "Department", "College", "State", "District", "Passed Out", "File Name"]
    writer = csv.DictWriter(str_buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)
    bytes_buf = io.BytesIO(str_buf.getvalue().encode('utf-8'))
    bytes_buf.seek(0)
    return send_file(bytes_buf, as_attachment=True, download_name=filename, mimetype='text/csv')

@app.route('/export/excel')
def export_excel():
    candidates = get_filtered_query()
    filename = "Resume_Data_Export.xlsx"
    
    data = [{"Name": c.name, "Contact": c.phone, "Email": c.email, 
             "Degree": c.degree, "Department": c.department, "College": c.college, 
             "State": c.state, "District": c.district, "Passed Out": c.year_passing, 
             "File Name": c.filename} for c in candidates]
    
    df = pd.DataFrame(data)
    bytes_buf = io.BytesIO()
    with pd.ExcelWriter(bytes_buf, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    bytes_buf.seek(0)
    return send_file(bytes_buf, as_attachment=True, download_name=filename, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

@app.route("/init-db")
def init_db():
    with app.app_context():
        db.create_all()
    return "Database tables created successfully"

if __name__ == '__main__':
    app.run(debug=True)
