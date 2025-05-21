from django.shortcuts import render,redirect,HttpResponse
from django.http import JsonResponse,HttpResponseRedirect
from .models import *
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
import fitz  # PyMuPDF for PDF parsing
import spacy
import random

nlp = spacy.load("en_core_web_sm")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
            print("extracted text: ", text)
    return text

def extract_entities(text):
    doc = nlp(text)
    skills = []
    for ent in doc.ents:
        # Filter for relevant entities, e.g., skills, qualifications
        if ent.label_ in ("ORG", "PERSON", "GPE", "NORP", "SKILL", "WORK_OF_ART", "MISC",
                          "DATE", "TIME", "MONEY", "QUANTITY", "PERCENT", "LANGUAGE", 
                          "PRODUCT", "EVENT", "FAC", "DEGREE",):
            skills.append(ent.text.lower())
    return set(skills)
# Create your views here.

def index(request):
    return render(request, 'index.html')

def jobseekindex(request):
    if 'jobseeker_id' in request.session:
        jobseeker_id=request.session['jobseeker_id']
        user=JobseekerRegister.objects.get(id=jobseeker_id)
        return render(request, 'jobseekindex.html', {'user':user})
    else:
        return redirect('jobseeker-login')
def jobprovideindex(request):
    if 'jobprovider_id' in request.session:
        jobprovide_id=request.session['jobprovider_id']
        user=JobproviderRegister.objects.get(id=jobprovide_id)
        return render(request, 'jobprovideindex.html', {'user':user})
    return redirect('jobproviderlogin')

from django.core.mail import send_mail
def jobseekerreg(request):
    if request.method == 'POST':
        name=request.POST.get('name')
        email=request.POST.get('email')
        password=request.POST.get('password')
        number=request.POST.get('number')
        
        jobseek=JobseekerRegister(name=name, email=email, password=password, phone_number=number)
        jobseek.save()
        subject = 'Registration Successful'
        message = f'Hi {name},\n\n Your registration was successful! Welcome to our platform.'
        from_email = settings.EMAIL_HOST_USER
        recipient_list = [email]

        send_mail(subject, message, from_email, recipient_list, fail_silently=False)

        return redirect('jobseeker-login')
    return render(request, 'jobseekerreg.html')

def jobseekerlogin(request):
    if request.method == 'POST':
        email=request.POST.get('email')
        password=request.POST.get('password')
        try:
            jobseek=JobseekerRegister.objects.get(email=email, password=password)
            if jobseek:
                request.session['jobseeker_id'] = jobseek.id
                return redirect('jobseekerindex')
        except :
            alert="<script>alert('Invalid email or password'); window.location.href='/jobseeker-login/';</script>"
            return HttpResponse(alert)
    return render(request, 'jobseekerlogin.html')

def jobproviderreg(request):
    if request.method=='POST':
        name=request.POST.get('name')
        email=request.POST.get('email')
        password=request.POST.get('password')
        number=request.POST.get('number')
        cmpny_name=request.POST.get('company_name')
        
        jobprovider=JobproviderRegister(name=name, email=email, password=password, phone_number=number,company_name=cmpny_name)
        jobprovider.save()
        return redirect('jobproviderlogin')
    return render(request,'jobprovidereg.html')
    
def jobproviderlogin(request):
    if request.method=='POST':
        email=request.POST.get('email')
        password=request.POST.get('password')
        
        try:
            jobprovider=JobproviderRegister.objects.get(email=email, password=password)
            if jobprovider:
                request.session['jobprovider_id'] = jobprovider.id
                return redirect('jobprovideindex')
        except:
            alert="<script>alert('Invalid email or password'); window.location.href='/jobproviderlogin/';</script>"
            return HttpResponse(alert)
    return render(request, 'jobproviderlogin.html')


def jobseekprofile(request):
    if 'jobseeker_id' in request.session:
        jobseeker_id=request.session['jobseeker_id']
        user=JobseekerRegister.objects.get(id=jobseeker_id)
        msg = None
        msgi=None
        try:
            resumedtls=Resumes.objects.get(jobseeker=user)
            if resumedtls.resume_file is None:
                msg="no resume file"
            else:
                msgi=resumedtls
        except:
            msg = "No resume file uploaded."
            
        if request.method == "POST":
            resume_file = request.FILES.get('resume_file')
            resumes=Resumes(jobseeker=user, resume_file=resume_file)
            resumes.save()
            msg = "Resume uploaded successfully."
            return redirect('jobseekprofile')
            
        return render(request, 'jobseekprofile.html', {'user':user,'msg':msg,'msgi':msgi})
    else:
        return redirect('jobseeker-login')
    
def jobseekereditprofile(request):
    if 'jobseeker_id' in request.session:
        jobseeker_id=request.session['jobseeker_id']
        user=JobseekerRegister.objects.get(id=jobseeker_id)
        resumedtls=Resumes.objects.get(jobseeker=user)
        if request.method == 'POST':
            name=request.POST.get('name')
            email=request.POST.get('email')
            password=request.POST.get('password')
            number=request.POST.get('number')
            resum=request.FILES.get('resum')
            
            user.name=name
            user.email=email
            user.password=password
            user.phone_number=number
            if resum:
                resumedtls.resume_file=resum
            user.save()
            resumedtls.save()
            return redirect('jobseekprofile')
        return render(request, 'jobseekereditprofile.html', {'user':user, 'resumedtls':resumedtls})
    else:
        return redirect('jobseeker-login')
    
    
def jobproviderprofile(request):
    if 'jobprovider_id' in request.session:
        jobprovider_id = request.session['jobprovider_id']
        user = JobproviderRegister.objects.get(id=jobprovider_id)
        try:
            profile = JobProviderProfile.objects.get(jobprovider=user)
            return render(request, 'jobproviderprofile.html', {'user': user, 'profile': profile})
        except JobProviderProfile.DoesNotExist:
            if request.method == 'POST':
                company_logo = request.FILES.get('company_logo')
                description = request.POST.get('description')
                location = request.POST.get('location')
                profile = JobProviderProfile(jobprovider=user, company_logo=company_logo, company_description=description, address=location)
                profile.save()
                return redirect('jobproviderprofile')
            return render(request, 'jobproviderprofile.html', {'user': user, 'profile': None})

    else:
        return redirect('jobproviderlogin')
    
    
def jobprovidereditprofile(request):
    if 'jobprovider_id' in request.session:
        jobprovider_id = request.session['jobprovider_id']
        user = JobproviderRegister.objects.get(id=jobprovider_id)
        profile = JobProviderProfile.objects.get(jobprovider=user)
        if request.method == 'POST':
            company_logo = request.FILES.get('company_logo')
            description = request.POST.get('description')
            location = request.POST.get('location')
            
            user.name = request.POST.get('name')
            user.email = request.POST.get('email')
            user.password = request.POST.get('password')
            user.company_name = request.POST.get('company_name')
            
            if company_logo:
                profile.company_logo = company_logo
            profile.company_description = description
            
            profile.address = location
            
            user.save()
            profile.save()
            
            return redirect('jobproviderprofile')
        return render(request, 'jobprovidereditprofile.html', {'user': user, 'profile': profile})
    else:
        return redirect('jobproviderlogin')
    
    
def PostJob(request):
    if 'jobprovider_id' in request.session:
        jobprovider_id = request.session['jobprovider_id']
        user = JobproviderRegister.objects.get(id=jobprovider_id)
        if request.method == 'POST':
            title = request.POST.get('title')
            description = request.POST.get('description')
            
            job = Jobrequirements(jobprovider=user, job_title=title, job_description=description,status=True)
            job.save()
            
            return redirect('jobprovideindex')
        return render(request, 'postjob.html')
    else:
        return redirect('jobproviderlogin')
    
def viewJobs(request):
    if 'jobprovider_id' in request.session:
        jobprovider_id = request.session['jobprovider_id']
        user = JobproviderRegister.objects.get(id=jobprovider_id)
        jobs = Jobrequirements.objects.filter(jobprovider=user)
        return render(request, 'viewjobs.html', {'jobs': jobs})
    else:
        return redirect('jobproviderlogin')
    
def findJob(request):
    if 'jobseeker_id' in request.session:
        jobseeker_id = request.session['jobseeker_id']
        user = JobseekerRegister.objects.get(id=jobseeker_id)
        try:
            resumedtls = Resumes.objects.get(jobseeker=user)
        except :
            alert="<script>alert('no resume found!,please complete your profile );window.location.href='/jobseekprofile/';</script>"
            return HttpResponse(alert)
        jobs = Jobrequirements.objects.filter(status=True) 
        resume_path = resumedtls.resume_file.path
        resume_text = extract_text_from_pdf(resume_path)
        resume_entities = extract_entities(resume_text)
        print("R",resume_entities)

        matched_jobs = []
        for job in jobs:
            job_text = f"{job.job_title} {job.job_description}"
            job_entities = extract_entities(job_text)
            print(job_entities)
    
            if job_entities:
                matched_keywords = resume_entities.intersection(job_entities)
                # match_score = len(matched_keywords) / len(job_entities)
                match_score = len(resume_entities.intersection(job_entities)) / len(job_entities)
                print("Match score", match_score)
            else:
                match_score = 0  
                matched_keywords=set()

            if match_score >= 0: 
                # matched_jobs.append((job, match_score))
                matched_jobs.append({
                    "job": job,
                    "score": match_score,
                    "matched_keywords": matched_keywords  
                })
        matched_jobs.sort(key=lambda x: x["score"], reverse=True)
        # matched_jobs.sort(key=lambda x: x[1], reverse=True)

        return render(request, 'findjobs.html', {'matched_jobs': matched_jobs})
    
    return redirect('jobseeker-login')


def apply_job(request,resid):
    if 'jobseeker_id' in request.session:
        jobseeker_id = request.session['jobseeker_id']
        user = JobseekerRegister.objects.get(id=jobseeker_id)
        job = Jobrequirements.objects.get(id=resid)
        jobprovid=job.jobprovider
        applied_jobs = JobApplications.objects.filter(jobseeker=user, job_requirements=job)
        if applied_jobs:
            alert="<script>alert('Already applied this job'); window.location.href='/findJob/';</script>"
            return HttpResponse(alert)
        
        applied_job = JobApplications(jobseeker=user, job_requirements=job,jobprovider=jobprovid,selected_status=False)
        applied_job.save()
        
        return redirect('jobseekerindex')
    return redirect('jobseeker-login')

def viewappiledjobsprovider(request):
    if 'jobprovider_id' in request.session:
        jobprovider_id = request.session['jobprovider_id']
        jobs = JobApplications.objects.filter(jobprovider__id=jobprovider_id).select_related('jobseeker', 'job_requirements')
        return render(request, 'viewappiledjobsprovider.html', {'jobs': jobs})
    return redirect('jobproviderlogin')
def toggle_status(request, application_id):
    if 'jobprovider_id' in request.session:
        application = JobApplications.objects.get(id=application_id)
        application.selected_status = not application.selected_status
        application.save()
    return redirect('viewapplication') 

def jobseekApplications(request):
    if 'jobseeker_id' in request.session:
        jobseeker_id = request.session['jobseeker_id']
        applications = JobApplications.objects.filter(jobseeker__id=jobseeker_id)
        return render(request, 'jobseekapplications.html', {'applications': applications})
    return redirect('jobseeker-login')
    
    
def EditjobReq(request, jobid):
    if 'jobprovider_id' in request.session:
        jobprovider_id = request.session['jobprovider_id']
        user = JobproviderRegister.objects.get(id=jobprovider_id)
        job = Jobrequirements.objects.get(id=jobid)
        if request.method == 'POST':
            title = request.POST.get('title')
            description = request.POST.get('description')
            status=request.POST.get('status')
            
            job.job_title = title
            job.job_description = description
            job.status = status
            
            job.save()
            
            return redirect('viewjobs')
        return render(request, 'editjobreq.html', {'job': job})
    return redirect('jobproviderlogin')

def Deletejob(request, jobid):
    if 'jobprovider_id' in request.session:
        job = Jobrequirements.objects.get(id=jobid)
        job.delete()
        return redirect('viewjobs')
    return redirect('jobproviderlogin')




import openai
from django.shortcuts import render, redirect
from django.http import HttpResponse
from .models import JobproviderRegister, Jobrequirements, Exam, ExamQuestion

# Set up OpenAI API key
import openai
# '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'
openai.api_key = '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'

def generate_questions_from_chatgpt(exam_title):
    try:
        # Generate questions based on the exam title
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Specify the chat model
            messages=[
                {"role": "system", "content": "You are a helpful assistant that creates exam questions."},
                {"role": "user", "content": f"Generate 5 questions based on the exam title: {exam_title}. Each question should be clear, concise, and test the core knowledge of the subject."}
            ],
            max_tokens=150,  # Control the response length
            temperature=0.7  # Control randomness of the output
        )

        # Process the response to get the questions
        generated_questions = response['choices'][0]['message']['content'].strip().split('\n')
        return generated_questions
    except Exception as e:
        print("Error generating questions:", e)
        return []

def create_exam_and_questions(request):
    if 'jobprovider_id' in request.session:
        jobprovider_id = request.session['jobprovider_id']
        print("dssg",jobprovider_id)
        
        user = JobproviderRegister.objects.get(id=jobprovider_id)
        
        try:
            jobs = Jobrequirements.objects.filter(jobprovider=user).first()
        except Exception as e:
            print("oooo",jobprovider_id,e)
            alert = "<script>alert('No job found!');window.location.href='/jobprovideindex/';</script>"
            return HttpResponse(alert)
        
        if request.method == 'POST':
            title = request.POST.get('title')
            time_limit = request.POST.get('time_limit')
            questions = request.POST.getlist('questions')  
            max_marks_list = request.POST.getlist('max_marks')  
            
            # Create the exam entry
            exam = Exam(
                jobprovider=user,
                job_requirements=jobs,
                title=title,
                time_limit=time_limit
            )
            exam.save()

            # Generate additional questions using ChatGPT
            generated_questions = generate_questions_from_chatgpt(title)
            
            # Combine user-provided questions with generated questions
            all_questions = questions + generated_questions

            # Ensure that each question has a corresponding max_marks
            all_max_marks = max_marks_list + ['10'] * len(generated_questions)  # Default max marks for generated questions

            # Save all questions to the ExamQuestion model
            for question_text, max_marks in zip(all_questions, all_max_marks):
                if question_text and max_marks.isdigit(): 
                    ExamQuestion.objects.create(
                        exam=exam,
                        question_text=question_text,
                        max_marks=int(max_marks)
                    )

            return redirect('viewapplication')

        return render(request, 'create_exam_and_questions.html')
    
    return redirect('jobproviderlogin')


def list_exams(request):
    if 'jobprovider_id' in request.session:
        jobprovider_id = request.session['jobprovider_id']
        
        # Retrieve all exams for the job provider
        exams = Exam.objects.filter(jobprovider_id=jobprovider_id)
        
        return render(request, 'list_exams.html', {'exams': exams})
    

    
    return redirect('jobproviderlogin')

def view_exam_questions(request, exam_id):
    if 'jobprovider_id' in request.session:
        jobprovider_id = request.session['jobprovider_id']
        exam = Exam.objects.get(id=exam_id, jobprovider_id=jobprovider_id)
        questions = exam.questions.all()  # Access related questions

        return render(request, 'view_exam_questions.html', {'exam': exam, 'questions': questions})

    return redirect('jobproviderlogin')


def generate_random_token(length=8):
    return ''.join(random.choices('0123456789', k=length))






from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, HttpResponse
from .models import Exam, JobseekerRegister, ExamQuestion, Answers


# def take_exam(request, exam_id):
#     if 'jobseeker_id' in request.session:
#         jobseeker_id = request.session['jobseeker_id']
#         jobseeker = JobseekerRegister.objects.get(id=jobseeker_id)

#         # Generate a session-specific exam token
#         if not request.session.get('exam_token'):
#             request.session['exam_token'] = generate_random_token()

#         exam = get_object_or_404(Exam, id=exam_id)

#         # Check if the user has already taken the exam
#         if Answers.objects.filter(
#                 jobseeker=jobseeker,
#                 exam_question__exam=exam,
#                 exam_status=True
#             ).exists():
#             alert = "<script>alert('You have already attended this exam.'); window.location.replace('/jobseekindex/');</script>"
#             return HttpResponse(alert)

#         # Verify access via token
#         if 'token' in request.GET and request.GET['token'] != request.session['exam_token']:
#             return HttpResponse("Invalid access attempt.", status=403)

#         if request.method == 'POST':
#             responses = request.POST.getlist('responses')
#             question_ids = request.POST.getlist('question_ids')
#             warning_count = int(request.POST.get('warning_count', 0))

#             for question_id, response in zip(question_ids, responses):
#                 question = get_object_or_404(ExamQuestion, id=question_id)

#                 # Get or create the answer instance for each question
#                 answer, created = Answers.objects.get_or_create(
#                     jobseeker=jobseeker,
#                     exam_question=question,
#                     defaults={'answer': response}
#                 )

#                 # Update the answer if it already exists
#                 answer.answer = response

#                 # Log malpractice warnings
#                 if warning_count > 0:
#                     answer.exam_log = (answer.exam_log or '') + f"\nMalpractice detected: {warning_count} warning(s)."
#                     answer.exam_status = True
#                 else:
#                     answer.exam_status = True
#                     answer.exam_log = "No malpractice detected."

#                 answer.save()

#             # Redirect all users to exam_completed regardless of malpractice status
#             return JsonResponse({'redirect_url': redirect('exam_completed').url})

#         # Display exam questions
#         questions = exam.questions.all()
#         return render(request, 'take_exam.html', {'exam': exam, 'questions': questions, 'jobseeker': jobseeker})
#     return redirect('jobseeker-login')


from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, JsonResponse
from django.utils import timezone
from threading import Thread
import cv2
import dlib
import os
import time
from scipy.spatial import distance as dist
from imutils import face_utils
from ultralytics import YOLO
import numpy as np
from django.conf import settings
from django.contrib.auth.decorators import login_required
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from .models import JobseekerRegister, Exam, ExamQuestion, Answers

# Global flags and settings
monitoring_active = {}  # Dictionary to track monitoring status by jobseeker-exam pair
violation_history = {}  # Format: {jobseeker_exam_key: {'mobile': bool, 'multiple': bool, 'not_looking': bool}}

# Load face detection models
face_detector = dlib.get_frontal_face_detector()
predictor_path = os.path.join(settings.BASE_DIR, 'shape_predictor_68_face_landmarks.dat')
face_predictor = dlib.shape_predictor(predictor_path)

# Load YOLO model
yolo_model = YOLO('yolov8n.pt')  # Using the smaller model for performance

def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Compute the euclidean distance between the horizontal eye landmarks
    C = dist.euclidean(eye[0], eye[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def generate_random_token():
    import random
    import string
    return ''.join(random.choices(string.ascii_letters + string.digits, k=32))

def monitor_exam(jobseeker, exam):
    jobseeker_exam_key = f"{jobseeker.id}-{exam.id}"
    
    # Initialize violation history
    if jobseeker_exam_key not in violation_history:
        violation_history[jobseeker_exam_key] = {
            'mobile_detected': False,
            'multiple_persons_detected': False,
            'not_looking_at_screen': 0,
            'multiple_persons_count': 0,
            'mobile_detection_count': 0
        }
    
    cap = cv2.VideoCapture(0)
    
    # Give camera time to initialize
    time.sleep(1)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open camera for jobseeker {jobseeker.id}, exam {exam.id}")
        return
        
    print(f"INFO: Camera opened successfully for monitoring jobseeker {jobseeker.id}, exam {exam.id}")
    
    while cap.isOpened() and monitoring_active.get(jobseeker_exam_key, True):
        ret, frame = cap.read()
        if not ret:
            print(f"WARNING: Failed to read frame for jobseeker {jobseeker.id}, exam {exam.id}")
            time.sleep(0.5)
            continue

        # Process frame with YOLOv8 for person/phone detection
        results = yolo_model(frame)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()
        
        # Count people (class_id 0 is person in COCO dataset)
        num_people = sum(1 for cid in class_ids if int(cid) == 0)
        
        # Check for mobile phone (class_id 67 is cell phone in COCO dataset)
        mobile_detected = any(int(cid) == 67 for cid in class_ids)
        
        # Update violation history if violations are detected
        if mobile_detected:
            violation_history[jobseeker_exam_key]['mobile_detected'] = True
            violation_history[jobseeker_exam_key]['mobile_detection_count'] += 1
            print(f"VIOLATION: Mobile phone detected for jobseeker {jobseeker.id}, exam {exam.id}")
        
        if num_people > 1:
            violation_history[jobseeker_exam_key]['multiple_persons_detected'] = True
            violation_history[jobseeker_exam_key]['multiple_persons_count'] += 1
            print(f"VIOLATION: Multiple persons ({num_people}) detected for jobseeker {jobseeker.id}, exam {exam.id}")
        if not num_people:
            violation_history[jobseeker_exam_key]['not_looking_at_screen'] = True
            violation_history[jobseeker_exam_key]['not_looking_at_screen'] += 1
            print(f"VIOLATION:not looking ({num_people}) detected for jobseeker {jobseeker.id}, exam {exam.id}")
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_detector(gray, 0)
        
        # Eye monitoring
        eyes_open = False
        face_present = len(faces) > 0
        
        for face in faces:
            # Determine facial landmarks
            shape = face_predictor(gray, face)
            shape = face_utils.shape_to_np(shape)
            
            # Extract eye coordinates
            left_eye = shape[36:42]
            right_eye = shape[42:48]
            
            # Calculate eye aspect ratio
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            
            # Check if eyes are open (EAR > threshold)
            if ear > 0.25:  # EYE_AR_THRESH
                eyes_open = True
                break
        
        # Check if person is looking at screen
        if face_present and not eyes_open:
            violation_history[jobseeker_exam_key]['not_looking_at_screen'] += 1
            print(f"VIOLATION: Not looking at screen for jobseeker {jobseeker.id}, exam {exam.id}")
        
        # Small delay to reduce CPU usage
        time.sleep(0.2)

    # Clean up
    cap.release()
    print(f"INFO: Camera released for jobseeker {jobseeker.id}, exam {exam.id}")

def take_exam(request, exam_id):
    if 'jobseeker_id' in request.session:
        jobseeker_id = request.session['jobseeker_id']
        jobseeker = JobseekerRegister.objects.get(id=jobseeker_id)

        # Generate a session-specific exam token
        if not request.session.get('exam_token'):
            request.session['exam_token'] = generate_random_token()

        exam = get_object_or_404(Exam, id=exam_id)
        jobseeker_exam_key = f"{jobseeker.id}-{exam.id}"

        # Check if the user has already taken the exam
        if Answers.objects.filter(
                jobseeker=jobseeker,
                exam_question__exam=exam,
                exam_status=True
            ).exists():
            alert = "<script>alert('You have already attended this exam.'); window.location.replace('/jonseekindex/');</script>"
            return HttpResponse(alert)

        # Verify access via token
        if 'token' in request.GET and request.GET['token'] != request.session['exam_token']:
            return HttpResponse("Invalid access attempt.", status=403)

        # Start monitoring
        if request.method == 'GET' and not monitoring_active.get(jobseeker_exam_key, False):
            monitoring_active[jobseeker_exam_key] = True
            monitoring_thread = Thread(target=monitor_exam, args=(jobseeker, exam))
            monitoring_thread.daemon = True  # Thread stops when main process ends
            monitoring_thread.start()

        if request.method == 'POST':
            # Stop monitoring when exam is submitted
            monitoring_active[jobseeker_exam_key] = False
            
            responses = request.POST.getlist('responses')
            question_ids = request.POST.getlist('question_ids')
            warning_count = int(request.POST.get('warning_count', 0))

            # Get violation logs from the monitoring
            violations_log = ""
            
            if jobseeker_exam_key in violation_history:
                vh = violation_history[jobseeker_exam_key]
                if vh['mobile_detected']:
                    violations_log += f"Mobile phone detected {vh['mobile_detection_count']} times. "
                
                if vh['multiple_persons_detected']:
                    violations_log += f"Multiple persons detected {vh['multiple_persons_count']} times. "
                
                if vh['not_looking_at_screen'] > 0:
                    violations_log += f"Not looking at screen {vh['not_looking_at_screen']} times. "
            
            # Add frontend warnings
            if warning_count > 0:
                violations_log += f"UI detected warnings: {warning_count}"
            
            # If no violations were detected
            if not violations_log:
                violations_log = "No malpractice detected."

            for question_id, response in zip(question_ids, responses):
                question = get_object_or_404(ExamQuestion, id=question_id)

                # Get or create the answer instance for each question
                answer, created = Answers.objects.get_or_create(
                    jobseeker=jobseeker,
                    exam_question=question,
                    defaults={'answer': response}
                )

                # Update the answer if it already exists
                answer.answer = response
                answer.exam_status = True
                answer.exam_log = violations_log
                answer.save()

            # Redirect all users to exam_completed regardless of malpractice status
            return JsonResponse({'redirect_url': '/exam_completed/'})

        # Display exam questions
        questions = exam.questions.all()
        return render(request, 'take_exam.html', {'exam': exam, 'questions': questions, 'jobseeker': jobseeker})
    return redirect('jobseeker-login')


from groq import Groq
from reportlab.lib import colors
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
from django.http import HttpResponse
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate
from datetime import datetime
import json
from reportlab.lib.units import inch
from django.shortcuts import get_object_or_404

groq_client = Groq(api_key='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')

import json
from groq import Groq  # Correct import for the Groq library

import re

def validate_answer_with_groq(question_text, answer_text, max_marks, exam_log, question_type="text"):
    """Validate answer using Groq API and assign marks, adaptable to question type"""
    try:
        malpractice_detected = False
        malpractice_details = ""
        
        if exam_log and exam_log.lower() != "good" and "no violations" not in exam_log.lower():
            malpractice_detected = True
            malpractice_details = exam_log

        prompt = f"""
        As an expert examiner, evaluate this answer based on the question and its type:
        Question: {question_text}
        Answer: {answer_text}
        Maximum Marks: {max_marks}
        Question Type: {question_type} (e.g., text, multiple-choice, code, link-based, etc.)
        Malpractice: {malpractice_details}

        Instructions:
        1. Assign a score (out of {max_marks}) based on the answer's relevance and correctness for the given question type.
           - For text questions, expect a descriptive answer.
           - For link-based questions, a valid URL might be acceptable if it’s relevant.
           - For code questions, evaluate syntax and logic.
           - Reduce the score if malpractice is significant (e.g., repeated warnings).
        2. Provide brief feedback on the answer quality.
        3. If the answer is empty or completely irrelevant to the question type, give zero marks.
        4. Determine if it meets minimum eligibility (60% of {max_marks}).

        Return in JSON format:
        {{
            "score": <number>,
            "feedback": "<string>",
            "eligible": <boolean>
        }}
        """
        
        response = groq_client.chat.completions.create(
            model="mistral-saba-24b",
            messages=[
                {"role": "system", "content": "You are an expert examiner evaluating exam answers."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        raw_content = response.choices[0].message.content.strip()
        print("Raw API Response:", raw_content)
        
        if not raw_content:
            raise ValueError("Empty response from API")
        
        # Extract JSON from the response (removing preamble text)
        json_match = re.search(r'```json\s*(.*?)\s*```', raw_content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = raw_content  # Fallback if no ```json``` markers
        
        try:
            result = json.loads(json_str)
        except json.JSONDecodeError as e:
            print("Failed to parse JSON:", str(e), "Raw content:", raw_content)
            score = 0 if not answer_text.strip() else max_marks // 2
            feedback = raw_content or "Unable to evaluate answer properly"
            eligible = (score >= 0.6 * max_marks)
            result = {"score": score, "feedback": feedback, "eligible": eligible}
        
        if not isinstance(result, dict) or "score" not in result or "feedback" not in result or "eligible" not in result:
            raise ValueError("Invalid response format from API")
        
        # Normalize score
        score = min(result["score"], max_marks)
        if not answer_text.strip():
            score = 0
            result["feedback"] = "No answer provided. " + result["feedback"]
        result["score"] = score
        result["eligible"] = bool(result["eligible"])
        
        return result
    except Exception as e:
        print("Error in validation:", str(e))
        return {
            "score": 0,
            "feedback": f"Error in validation: {str(e)}",
            "eligible": False
        }

def generate_exam_performance_pdf(request, exam_id, jobseeker_id):
    exam = get_object_or_404(Exam, id=exam_id)
    jobseeker = get_object_or_404(JobseekerRegister, id=jobseeker_id)
    answers = Answers.objects.filter(jobseeker=jobseeker, exam_question__exam=exam)
    
    total_marks_possible = sum(q.max_marks for q in exam.questions.all())
    total_marks_scored = 0
    validated_results = []
    
    for answer in answers:
        question_type = getattr(answer.exam_question, 'question_type', 'text')
        
        validation = validate_answer_with_groq(
            answer.exam_question.question_text,
            answer.answer,
            answer.exam_question.max_marks,
            answer.exam_log,
            question_type=question_type
        )
        
        # Update answer with feedback and score
        answer.save()
        
        total_marks_scored += validation["score"]
        validated_results.append({
            "question": answer.exam_question.question_text,
            "answer": answer.answer,
            "score": validation["score"],
            "max_marks": answer.exam_question.max_marks,
            "feedback": validation["feedback"],
            "eligible": validation["eligible"]
        })
    
    percentage_score = (total_marks_scored / total_marks_possible * 100) if total_marks_possible > 0 else 0
    
    # PDF setup
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="exam_report_{jobseeker.name}_{exam.title}.pdf"'
    doc = SimpleDocTemplate(response, pagesize=letter)
    styles = getSampleStyleSheet()
    table_style = ParagraphStyle('TableCell', parent=styles['Normal'], fontSize=9, leading=11)
    
    elements = []
    elements.append(Paragraph("Exam Performance Report", styles['Heading1']))
    elements.append(Spacer(1, 0.25*inch))
    
    info_data = [
        ["Candidate:", jobseeker.name],
        ["Exam:", exam.title],
        ["Date:", datetime.now().strftime("%d/%m/%Y")],
        ["Total Score:", f"{total_marks_scored}/{total_marks_possible} ({percentage_score:.1f}%)"]
    ]
    elements.append(Table(info_data, colWidths=[2*inch, 3.5*inch]))
    elements.append(Spacer(1, 0.25*inch))
    
    elements.append(Paragraph("Detailed Performance", styles['Heading2']))
    question_data = [["Question", "Answer", "Ai Score", "Feedback"]]
    
    for result in validated_results:
        question_paragraph = Paragraph(
            result["question"][:100] + "..." if len(result["question"]) > 100 else result["question"], 
            table_style
        )
        answer_paragraph = Paragraph(
            result["answer"][:100] + "..." if len(result["answer"]) > 100 else result["answer"], 
            table_style
        )
        feedback_paragraph = Paragraph(result["feedback"], table_style)
        
        question_data.append([
            question_paragraph,
            answer_paragraph,
            f"{result['score']}/{result['max_marks']}",
            feedback_paragraph
        ])
    
    t = Table(question_data, colWidths=[1.5*inch, 1.5*inch, 0.75*inch, 2.75*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    
    elements.append(t)
    elements.append(Spacer(1, 0.25*inch))
    
    elements.append(Paragraph("Proctoring Summary", styles['Heading2']))
    exam_logs = set(a.exam_log.split("| Feedback: ")[0].strip() for a in answers if a.exam_log)
    proctoring_text = "<br/>".join(exam_logs) if exam_logs else "No violations detected"
    elements.append(Paragraph(proctoring_text, styles['Normal']))
    
    eligible = percentage_score >= 60 and all("No violations" in a.exam_log.split("| Feedback: ")[0].strip() or not a.exam_log for a in answers)
    elements.append(Spacer(1, 0.25*inch))
    elements.append(Paragraph(f"Eligibility Status: {'Eligible' if eligible else 'Not Eligible'}", styles['Heading3']))
    
    doc.build(elements)
    return response

from django.shortcuts import render, get_object_or_404
from .models import Exam, Answers, JobseekerRegister, ExamQuestion

def exam_attendees(request, exam_id):
    # Get the exam details
    exam = get_object_or_404(Exam, id=exam_id)
    
    # Get unique jobseekers who attended this exam
    attendees = JobseekerRegister.objects.filter(
        answers__exam_question__exam=exam, 
        answers__exam_status=True
    ).distinct()
    
    # Get detailed information for each attendee
    attendee_details = []
    for attendee in attendees:
        # Get all answers by this jobseeker for this exam
        answers = Answers.objects.filter(
            jobseeker=attendee,
            exam_question__exam=exam,
            exam_status=True
        )
        
        # Calculate total marks scored and total possible marks
        total_scored = sum(answer.marks_scored or 0 for answer in answers)
        total_possible = sum(answer.exam_question.max_marks for answer in answers)
        
        # Calculate percentage
        percentage = (total_scored / total_possible * 100) if total_possible > 0 else 0
        
        # Add to details list
        attendee_details.append({
            'jobseeker': attendee,
            'total_scored': total_scored,
            'total_possible': total_possible,
            'percentage': round(percentage, 2),
            'passed': percentage >= 60,  # Assuming 60% is passing grade
        })
    
    context = {
        'exam': exam,
        'attendee_details': attendee_details
    }
    
    return render(request, 'exam_attendees.html', context)
# from django.http import HttpResponse
# from reportlab.lib.pagesizes import letter
# from reportlab.lib import colors
# from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
# from reportlab.lib.units import inch
# from datetime import datetime
# import os
# from django.conf import settings
# from django.shortcuts import get_object_or_404

# def generate_exam_performance_pdf(request, exam_id, jobseeker_id):
#     # Get required data
#     exam = get_object_or_404(Exam, id=exam_id)
#     jobseeker = get_object_or_404(JobseekerRegister, id=jobseeker_id)
#     job_requirements = exam.job_requirements
    
#     # Calculate total score and other metrics
#     exam_questions = ExamQuestion.objects.filter(exam=exam)
#     answers = Answers.objects.filter(jobseeker=jobseeker, exam_question__in=exam_questions)
    
#     total_marks_possible = sum(q.max_marks for q in exam_questions)
#     total_marks_scored = sum(a.marks_scored for a in answers if a.marks_scored is not None)
    
#     if total_marks_possible > 0:
#         percentage_score = (total_marks_scored / total_marks_possible) * 100
#     else:
#         percentage_score = 0
    
#     # Check for malpractice
#     malpractice_detected = any('malpractice detected' in (a.exam_log or '').lower() for a in answers)
#     suspicious_behavior = any(('suspicious' in (a.exam_log or '').lower() or 
#                                'warning' in (a.exam_log or '').lower()) 
#                                for a in answers)
    
#     # Initialize PDF
#     response = HttpResponse(content_type='application/pdf')
#     response['Content-Disposition'] = f'attachment; filename="candidate_report_{jobseeker.name}_{exam.title}.pdf"'
    
#     doc = SimpleDocTemplate(response, pagesize=letter)
#     styles = getSampleStyleSheet()
    
#     # Custom styles
#     title_style = ParagraphStyle(
#         'Title',
#         parent=styles['Heading1'],
#         fontSize=16,
#         spaceAfter=12,
#         alignment=1  # Center alignment
#     )
    
#     heading_style = ParagraphStyle(
#         'Heading2',
#         parent=styles['Heading2'],
#         fontSize=14,
#         spaceAfter=10,
#     )
    
#     normal_style = styles['Normal']
    
#     # Content elements
#     elements = []
    
#     # Header
#     elements.append(Paragraph("Candidate Performance Report", title_style))
#     elements.append(Paragraph("SmartHire AI-Powered Recruitment System", styles['Italic']))
#     elements.append(Spacer(1, 0.25*inch))
    
#     # Candidate info
#     data = [
#         ["Candidate Name:", jobseeker.name],
#         ["Application ID:", f"SH{jobseeker.id:06d}"],
#         ["Job Applied For:", job_requirements.job_title],
#         ["Report Date:", datetime.now().strftime("%d/%m/%Y")]
#     ]
    
#     t = Table(data, colWidths=[2*inch, 3.5*inch])
#     t.setStyle(TableStyle([
#         ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
#         ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
#         ('FONTSIZE', (0, 0), (-1, -1), 10),
#         ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
#     ]))
#     elements.append(t)
#     elements.append(Spacer(1, 0.25*inch))
    
#     # Test Performance
#     elements.append(Paragraph("1. Test Performance Overview", heading_style))
    
#     coding_efficiency = "N/A"
#     code_execution_time = "N/A"
    
#     # If this was a coding test, you would calculate these values
#     # For now, using placeholder values based on score
#     if percentage_score > 80:
#         coding_efficiency = "90%"
#         code_execution_time = "1.2 sec"
#     elif percentage_score > 60:
#         coding_efficiency = "75%"
#         code_execution_time = "1.8 sec"
#     else:
#         coding_efficiency = "60%"
#         code_execution_time = "2.5 sec"
    
#     # Table header
#     test_data = [
#         ["Parameter", "Score", "Remarks"],
#         ["Aptitude Score", f"{percentage_score:.1f}/100", "Pass" if percentage_score >= 60 else "Fail"],
#         ["Coding Score", f"{total_marks_scored}/{total_marks_possible}", "Pass" if percentage_score >= 60 else "Fail"],
#         ["Code Efficiency (%)", coding_efficiency, "Efficient" if percentage_score > 70 else "Average"],
#         ["Code Execution Time (sec)", code_execution_time, "Fast" if percentage_score > 70 else "Average"]
#     ]
    
#     t = Table(test_data, colWidths=[2*inch, 1.5*inch, 2*inch])
#     t.setStyle(TableStyle([
#         ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
#         ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
#         ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
#         ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
#         ('FONTSIZE', (0, 0), (-1, 0), 10),
#         ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
#         ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
#     ]))
#     elements.append(t)
#     elements.append(Spacer(1, 0.25*inch))
    
#     # AI-Driven Virtual Interview Performance
#     elements.append(Paragraph("2. AI-Driven Virtual Interview Performance", heading_style))
    
#     # Calculate these based on your requirements
#     # For now, using values derived from the score
#     accuracy = min(85, percentage_score + 5)
#     tech_knowledge = min(8, percentage_score/12.5)
#     problem_solving = min(7, percentage_score/14.3)
#     communication = min(9, 7 + (percentage_score/50))  # Base 7 + up to 2 points
    
#     interview_data = [
#         ["Parameter", "Score", "Remarks"],
#         ["Accuracy of Responses (%)", f"{accuracy:.1f}%", "Correct" if accuracy > 70 else "Needs Improvement"],
#         ["Depth of Technical Knowledge", f"{tech_knowledge:.1f}/10", "Strong" if tech_knowledge > 7 else "Average"],
#         ["Problem-Solving Skills", f"{problem_solving:.1f}/10", "Good" if problem_solving > 6 else "Average"],
#         ["Communication Clarity", f"{communication:.1f}/10", "Clear" if communication > 7 else "Average"]
#     ]
    
#     t = Table(interview_data, colWidths=[2*inch, 1.5*inch, 2*inch])
#     t.setStyle(TableStyle([
#         ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
#         ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
#         ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
#         ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
#         ('FONTSIZE', (0, 0), (-1, 0), 10),
#         ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
#         ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
#     ]))
#     elements.append(t)
#     elements.append(Spacer(1, 0.25*inch))
    
#     # Malpractice & Proctoring Results
#     elements.append(Paragraph("3. Malpractice & Proctoring Results", heading_style))
    
#     # Get proctoring data from answers
#     proctoring_data = [
#         ["Parameter", "Score", "Remarks"],
#         ["Impersonation Check", "Pass", "Verified"],
#         ["Unauthorized Materials Detected", "Yes" if suspicious_behavior else "No", "Detected" if suspicious_behavior else "None"],
#         ["Background Distractions", "High" if malpractice_detected else "Low", "Significant" if malpractice_detected else "Acceptable"]
#     ]
    
#     t = Table(proctoring_data, colWidths=[2*inch, 1.5*inch, 2*inch])
#     t.setStyle(TableStyle([
#         ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
#         ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
#         ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
#         ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
#         ('FONTSIZE', (0, 0), (-1, 0), 10),
#         ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
#         ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
#     ]))
#     elements.append(t)
#     elements.append(Spacer(1, 0.25*inch))
    
#     # Final Evaluation & Recommendation
#     elements.append(Paragraph("4. Final Evaluation & Recommendation", heading_style))
    
#     elements.append(Paragraph(f"Overall Score: {percentage_score:.1f}/100", normal_style))
#     elements.append(Spacer(1, 0.1*inch))
#     elements.append(Paragraph("Candidate Performance Summary:", normal_style))
    
#     # Determine final recommendation
#     if malpractice_detected:
#         recommendation = "Rejected (Malpractice)"
#         tech_proficiency = "Undetermined"
#         behavioral = "Questionable"
#     elif percentage_score < 60:
#         recommendation = "Rejected"
#         tech_proficiency = "Weak"
#         behavioral = "Below Average"
#     elif percentage_score < 75:
#         recommendation = "Shortlisted"
#         tech_proficiency = "Average"
#         behavioral = "Good"
#     else:
#         recommendation = "Selected"
#         tech_proficiency = "Strong"
#         behavioral = "Excellent"
    
#     summary_data = [
#         ["✔ Technical Proficiency:", tech_proficiency],
#         ["✔ Behavioral Suitability:", behavioral],
#         ["✔ Final Recommendation:", recommendation]
#     ]
    
#     t = Table(summary_data, colWidths=[2*inch, 3.5*inch])
#     t.setStyle(TableStyle([
#         ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
#         ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
#         ('FONTSIZE', (0, 0), (-1, -1), 10),
#         ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
#     ]))
#     elements.append(t)
#     elements.append(Spacer(1, 0.25*inch))
    
#     # Recruiter Notes & Suggestions
#     elements.append(Paragraph("5. Recruiter Notes & Suggestions", heading_style))
    
#     # Generate appropriate notes based on performance
#     if percentage_score > 80:
#         strengths = "Strong problem-solving skills, excellent communication clarity."
#         improvements = "Consider further assessment for team fit."
#         next_steps = "Proceed with offer letter issuance."
#     elif percentage_score > 60:
#         strengths = "Good technical foundation, adequate problem-solving ability."
#         improvements = "Needs to work on decision-making in high-pressure situations."
#         next_steps = "Consider for further interview rounds."
#     else:
#         strengths = "Shows basic understanding of concepts."
#         improvements = "Needs significant improvement in technical skills and problem-solving."
#         next_steps = "Not recommended for this position at this time."
    
#     # If malpractice was detected, override
#     if malpractice_detected:
#         strengths = "Unable to properly assess due to proctoring violations."
#         improvements = "Ethical considerations in test-taking environment."
#         next_steps = "Reject application due to malpractice concerns."
    
#     notes_data = [
#         ["• Strengths:", strengths],
#         ["• Areas for Improvement:", improvements],
#         ["• Next Steps:", next_steps]
#     ]
    
#     t = Table(notes_data, colWidths=[2*inch, 3.5*inch])
#     t.setStyle(TableStyle([
#         ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
#         ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
#         ('FONTSIZE', (0, 0), (-1, -1), 10),
#         ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
#     ]))
#     elements.append(t)
#     elements.append(Spacer(1, 0.5*inch))
    
#     # Authorized By
#     elements.append(Paragraph("Authorized By:", normal_style))
#     elements.append(Spacer(1, 0.1*inch))
#     elements.append(Paragraph(f"[Recruiter's Name]", normal_style))
#     elements.append(Paragraph(f"[{exam.jobprovider.company_name}]", normal_style))
    
#     # Build PDF
#     doc.build(elements)
#     return response

from django.core.mail import send_mail
from django.urls import reverse
import secrets
from django.conf import settings
def send_exam_links_page(request):
    if 'jobprovider_id' in request.session:
        jobprovider_id = request.session['jobprovider_id']
        
        # Fetch all applications and exams related to the job provider
        applications = JobApplications.objects.filter(jobprovider_id=jobprovider_id).select_related('job_requirements', 'jobseeker')
        exams = Exam.objects.filter(jobprovider_id=jobprovider_id)
        
        context = {
            'applications': applications,
            'exams': exams,
        }
        
        return render(request, 'send_exam_links.html', context)
    return redirect('jobprovidelogin')


from django.contrib import messages
from django.core.exceptions import MultipleObjectsReturned
from django.http import JsonResponse
from django.core.mail import send_mail
from django.shortcuts import get_object_or_404
from django.urls import reverse
from django.conf import settings

def send_exam_email(request, job_id, jobseeker_id):
    jobseeker = get_object_or_404(JobseekerRegister, id=jobseeker_id)
    exam_id = request.GET.get('exam_id')
    exam = get_object_or_404(Exam, id=exam_id)

    # Generate exam link
    exam_link = request.build_absolute_uri(reverse('take_exam', args=[exam.id]))

    try:
        send_mail(
            subject="Your Exam Link",
            message=(
                f"Hello {jobseeker.name},\n\n"
                "Please ensure you are logged in to the website before clicking this link; otherwise, you will be unable to access the exam.\n\n"
                f"Here is your exam link: {exam_link}\n\n"
                "Good luck!"
            ),
            from_email=settings.EMAIL_HOST_USER,
            recipient_list=[jobseeker.email],
            fail_silently=False,
        )
        # Return JSON response with success
        return JsonResponse({'status': 'success', 'message': f"Exam link sent successfully to {jobseeker.email}."})
    except Exception as e:
        # Return JSON response with error
        return JsonResponse({'status': 'error', 'message': f"Failed to send exam link to {jobseeker.email}. Error: {e}"})



def exam_results(request, exam_id):
    exam = get_object_or_404(Exam, id=exam_id)
    
    # Get jobseeker_id from query parameters
    jobseeker_id = request.GET.get('jobseeker')
    if not jobseeker_id:
        # Optionally handle the case where no jobseeker is specified
        return redirect('exam_attendees', exam_id=exam_id)  # or some error page
    
    # Filter answers by both exam and specific jobseeker
    answers = Answers.objects.filter(
        exam_question__exam=exam,
        exam_status=True,
        jobseeker__id=jobseeker_id
    )
    
    if request.method == "POST":
        # Flag to track if any marks were updated
        marks_updated = False
        
        # Process form submission for adding/editing marks
        for answer in answers:
            marks_field = f'marks_{answer.id}'
            new_marks = request.POST.get(marks_field)
            
            if new_marks is not None and new_marks.strip():
                try:
                    answer.marks_scored = int(new_marks)
                    answer.save()
                    marks_updated = True
                except ValueError:
                    # Handle invalid input (non-integer)
                    pass
        
        # Only redirect after processing all answers
        if marks_updated:
            return redirect('exam_attendees', exam_id=exam_id)  # Redirect back to attendees list
            
    return render(request, 'exam_results.html', {'exam': exam, 'answers': answers})
def DeleteExam(request, exmid):
    if 'jobprovider_id' in request.session:
        job = Exam.objects.get(id=exmid)
        job.delete()
        return redirect('list_exams')
    return redirect('jobproviderlogin')

def exam_completed(request):
    return render(request, 'exam_completed.html')

def exam_terminated(request):
    return HttpResponse('Exam terminated due to malpractice')
def logout(request):
    if 'jobprovider_id' in request.session:
        del request.session['jobprovider_id']
        return redirect('jobproviderlogin')
    elif 'jobseeker_id' in request.session:
        del request.session['jobseeker_id']
        return redirect('jobseeker-login')
    
  
  
  
def adminindex(request):
    if 'ademail' in request.session:
        jobsek = JobseekerRegister.objects.all()  # Get all job seekers
        jobpro = JobProviderProfile.objects.all()  # Get all job providers
        return render(request, 'adminindex.html', {'jobsek': jobsek, 'jobpro': jobpro})
    else:
        alert = "<script>alert('Please login as admin'); window.location.href='/admin_login/';</script>"
        return HttpResponse(alert)
def admin_log(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        if email == 'admin@gmail.com' and password == 'admin':
            request.session['ademail'] = email
            return redirect('admin_panel')
        else:
            alert="<script>alert('Invalid email or password'); window.location.href='/admin_log/';</script>"
            return HttpResponse(alert)
    return render(request, 'admin_login.html')


def admin_jobseekers(request):
    if 'ademail' not in request.session:
        alert = "<script>alert('Please login as admin'); window.location.href='/admin_log/';</script>"
        return HttpResponse(alert)

    jobseekers = JobseekerRegister.objects.all()
    return render(request, 'admin_jobseekers.html', {'jobseekers': jobseekers})

def edit_jobseeker(request, pk):
    jobseeker = get_object_or_404(JobseekerRegister, pk=pk)
    if request.method == 'POST':
        jobseeker.name = request.POST.get('name')
        jobseeker.email = request.POST.get('email')
        jobseeker.phone_number = request.POST.get('phone_number')
        jobseeker.save()
        messages.success(request, 'Job seeker updated successfully!')
        return redirect('admin_jobseekers')
    return render(request, 'edit_jobseeker.html', {'jobseeker': jobseeker})

def delete_jobseeker(request, pk):
    jobseeker = get_object_or_404(JobseekerRegister, pk=pk)
    jobseeker.delete()
    messages.success(request, 'Job seeker deleted successfully!')
    return redirect('admin_jobseekers')


def admin_jobproviders(request):
    if 'ademail' not in request.session:
        alert = "<script>alert('Please login as admin'); window.location.href='/admin_log/';</script>"
        return HttpResponse(alert)

    jobproviders = JobproviderRegister.objects.all()
    return render(request, 'admin_jobproviders.html', {'jobproviders': jobproviders})

def edit_jobprovider(request, pk):
    jobprovider = get_object_or_404(JobproviderRegister, pk=pk)
    profile = JobProviderProfile.objects.filter(jobprovider=jobprovider).first()

    if request.method == 'POST':
        jobprovider.name = request.POST.get('name')
        jobprovider.email = request.POST.get('email')
        jobprovider.phone_number = request.POST.get('phone_number')
        jobprovider.company_name = request.POST.get('company_name')
        jobprovider.save()

        # Update or create profile
        if profile:
            profile.company_description = request.POST.get('company_description')
            profile.address = request.POST.get('address')
            if request.FILES.get('company_logo'):
                profile.company_logo = request.FILES.get('company_logo')
            profile.save()
        else:
            # Create a new profile if it does not exist
            profile = JobProviderProfile(
                jobprovider=jobprovider,
                company_description=request.POST.get('company_description'),
                address=request.POST.get('address'),
                company_logo=request.FILES.get('company_logo')
            )
            profile.save()

        messages.success(request, 'Job provider and profile updated successfully!')
        return redirect('admin_jobproviders')

    return render(request, 'edit_jobprovider.html', {'jobprovider': jobprovider, 'profile': profile})

def delete_jobprovider(request, pk):
    jobprovider = get_object_or_404(JobproviderRegister, pk=pk)
    jobprovider.delete()
    messages.success(request, 'Job provider deleted successfully!')
    return redirect('admin_jobproviders')


def adlogout(request):
    if 'ademail' in request.session:
        del request.session['ademail']
        return redirect('admin_log')
    else:
        alert = "<script>alert('Please login as admin'); window.location.href='/admin_log/';</script>"
        return HttpResponse(alert)
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from groq import Groq
import json

client = Groq(api_key="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")

def chatbot(request):
    # Start with an initial question
    first_question = "Tell me about yourself."
    # Initialize session to store responses and question count
    request.session['question_count'] = 0
    request.session['responses'] = []
    return render(request, 'chatbot.html', {'first_question': first_question})

# @csrf_exempt
# def process_question(request):
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             user_response = data.get('response', '').strip()

#             if not user_response:
#                 return JsonResponse({
#                     'status': 'error',
#                     'message': 'Please provide a response.'
#                 })

#             # Store response and increment question count
#             question_count = request.session.get('question_count', 0)
#             responses = request.session.get('responses', [])
#             responses.append(user_response)
#             question_count += 1
#             request.session['question_count'] = question_count
#             request.session['responses'] = responses

#             # Check if 6 questions have been asked
#             if question_count >= 6:
#                 # Generate pass/fail report
#                 prompt = (
#                     f"User responses: {json.dumps(responses)}. "
#                     "Based on these 6 responses, determine if the user passes or fails the interview. "
#                     "Respond only with 'Pass' or 'Fail'."
#                 )
#                 response = client.chat.completions.create(
#                     model="llama3-70b-8192",
#                     messages=[
#                         {"role": "system", "content": (
#                             "You are a strict interviewer. Evaluate responses and decide pass/fail based on clarity, relevance, and professionalism."
#                         )},
#                         {"role": "user", "content": prompt}
#                     ]
#                 )
#                 result = response.choices[0].message.content.strip()
#                 # Clear session after completion
#                 request.session.flush()
#                 return JsonResponse({
#                     'status': 'success',
#                     'response_message': result
#                 })

#             # Generate a follow-up question based on the latest response
#             prompt = (
#                 f"Previous response: {user_response}. "
#                 "Ask a concise, relevant follow-up question related to this response for an interview."
#             )
#             response = client.chat.completions.create(
#                 model="llama3-70b-8192",
#                 messages=[
#                     {"role": "system", "content": (
#                         "You are a strict interviewer. Generate a concise follow-up question based on the user's response."
#                     )},
#                     {"role": "user", "content": prompt}
#                 ]
#             )
#             next_question = response.choices[0].message.content.strip()

#             return JsonResponse({
#                 'status': 'success',
#                 'response_message': next_question
#             })
#         except Exception as e:
#             return JsonResponse({'status': 'error', 'message': str(e)})
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from django.core.mail import send_mail
from django.conf import settings

@csrf_exempt
def process_question(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_response = data.get('response', '').strip()

            if not user_response:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Please provide a response.'
                })

            # Store response and increment question count
            question_count = request.session.get('question_count', 0)
            responses = request.session.get('responses', [])
            responses.append(user_response)
            question_count += 1
            request.session['question_count'] = question_count
            request.session['responses'] = responses

            # Check if 6 questions have been asked
            if question_count >= 6:
                # Generate pass/fail report
                prompt = (
                    f"User responses: {json.dumps(responses)}. "
                    "Based on these 6 responses, determine if the user passes or fails the interview. "
                    "Respond only with 'Pass' or 'Fail'."
                )
                response = client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[
                        {"role": "system", "content": (
                            "You are a very lenient interviewer. Pass the user if they provided any meaningful effort in at least 3 out of 6 responses. "
                            "Fail only if most responses (4 or more) are blank, gibberish, or completely unrelated to an interview context."
                        )},
                        {"role": "user", "content": prompt}
                    ]
                )
                result = response.choices[0].message.content.strip()

                # Get user email from session
                try:
                    user_id = request.session.get('jobseeker_id')
                    user_e = JobseekerRegister.objects.get(id=user_id)
                    user_email = user_e.email
                except JobseekerRegister.DoesNotExist:
                    print("Jobseeker")
                    return redirect('jobseeker-login')
                if user_email:
                    try:
                        # Send email with the result
                        subject = "Interview Result"
                        message = f"Dear User,\n\nYour interview result is: {result}\n\nThank you for participating!"
                        send_mail(
                            subject,
                            message,
                            settings.EMAIL_HOST_USER,  # From email (configured in settings)
                            [user_email],              # To email
                            fail_silently=False,
                        )
                    except Exception as email_error:
                        # Log the error if needed, but don’t block the response
                        print(f"Failed to send email: {str(email_error)}")
                else:
                    print("No user email found in session")

                # Clear session after completion
                if 'question_count' in request.session:
                    del request.session['question_count']
                if 'responses' in request.session:
                    del request.session['responses']
                return JsonResponse({
                    'status': 'success',
                    'response_message': result
                })

            # Generate a follow-up question based on the latest response
            prompt = (
                f"Previous response: {user_response}. "
                "Ask a concise, relevant follow-up question related to this response for an interview."
            )
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": (
                        "You are a strict interviewer. Generate a concise follow-up question based on the user's response."
                    )},
                    {"role": "user", "content": prompt}
                ]
            )
            next_question = response.choices[0].message.content.strip()

            return JsonResponse({
                'status': 'success',
                'response_message': next_question
            })
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
from django.shortcuts import get_object_or_404, redirect
from django.core.mail import send_mail
from django.contrib import messages
from .models import JobseekerRegister

@csrf_exempt
def send_interview_link(request, jobseeker_id):
    if request.method == 'POST':
        try:
            jobseeker = get_object_or_404(JobseekerRegister, id=jobseeker_id)
            interview_link = "http://127.0.0.1:8000/chat/"
            subject = "Your Interview Link"
            message = f"Dear {jobseeker.name},\n\nYou have been invited to an interview. Please click the link below to start:\n{interview_link}\n\nBest regards,\nYour Hiring Team"
            from_email = "thanikkalsooraj@gmail.com"
            recipient_list = [jobseeker.email]

            send_mail(subject, message, from_email, recipient_list, fail_silently=False)

            return JsonResponse({
                'status': 'success',
                'message': 'Interview link sent successfully!'
            })
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            })
    return JsonResponse({
        'status': 'error',
        'message': 'Invalid request method'
    })