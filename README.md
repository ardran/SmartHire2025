1. Introduction
The SmartHire Job Portal is a Django-based web application that
facilitates job postings, resume parsing, AI-powered exams, and AI
interviews. This document provides a step-by-step guide to install and
run the application on any device (Windows, macOS, or Linux).
2. Prerequisites Ensure your system meets the following requirements:
- Python 3.11.4
- Pip (Python package manager, included with Python)
- SQLite (Default for testing)
- Virtual Environment (Optional but recommended)
3. Installation Steps
3.1 Extract Project Files
- Unzip the provided project folder to your desired location.
3.2 Set Up a Virtual Environment (Recommended)
python -m venv venv
- Windows:
```bash
venv\Scripts\activate
```
- macOS/Linux:
```bash
source venv/bin/activate
```
3.3 Install Dependencies
```bash
pip install -r requirements.txt
```
*Installs Django, PyMuPDF, spaCy, OpenCV, dlib, and other required
packages.*
3.4 Set Up the Database
Option A: SQLite (Default for Testing)
- No additional setup required.
3.5 Apply Database Migrations
```bash
python manage.py makemigrations
python manage.py migrate
```
3.6 Create a Superuser (Admin Account)
```bash
python manage.py createsuperuser
```

Follow the prompts to set up an admin account.
3.7 Download AI Models
- spaCy English Model (For NLP-based resume parsing):
```bash
python -m spacy download en_core_web_sm
```
- dlib Face Landmark Model (For AI proctoring):
- Download `shape_predictor_68_face_landmarks.dat` from
[dlib.net](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz
2).
- Extract and place it in the project root folder.
---
4. Configuration
4.1 Environment Variables
Create a `.env` file in the project root with the following:
```env
SECRET_KEY=your_django_secret_key
DEBUG=True Set to False in production
DATABASE_URL=postgres://user:password@localhost:5432/smarthire_db
OPENAI_API_KEY=your_openai_key For AI-generated questions
GROQ_API_KEY=your_groq_key For answer validation
EMAIL_HOST_USER=your_email@gmail.com For email notifications
EMAIL_HOST_PASSWORD=your_email_password
```
4.2 Update `settings.py`
Ensure the following settings are configured:
```python
ALLOWED_HOSTS = ['*'] Replace with your domain in production
STATIC_URL = '/static/'
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
```
---
5. Running the Application
5.1 Start the Development Server
```bash
python manage.py runserver
```
- Access the application at:
[http://127.0.0.1:8000](http://127.0.0.1:8000)
5.2 Access Admin Panel
- Visit: [http://127.0.0.1:8000/admin](http://127.0.0.1:8000/admin)
- Log in with the superuser credentials.
6. Support
For assistance, contact:
smarthire064@gmail.com
