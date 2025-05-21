from django.db import models

# Create your models here.
class JobseekerRegister(models.Model):
    name = models.CharField(max_length=50,null=True, blank=True)
    email = models.EmailField(max_length=50,null=True, blank=True ,unique=True)
    password = models.CharField(max_length=50,null=True, blank=True)
    phone_number = models.CharField(max_length=15, null=True, blank=True)
    
class JobproviderRegister(models.Model):
    name = models.CharField(max_length=50,null=True, blank=True)
    email = models.EmailField(max_length=50,null=True, blank=True, unique=True)
    password = models.CharField(max_length=50,null=True, blank=True)
    company_name = models.CharField(max_length=50,null=True, blank=True)
    phone_number = models.CharField(max_length=15, null=True, blank=True)
    
    
class Resumes(models.Model):
    jobseeker = models.ForeignKey(JobseekerRegister, on_delete=models.CASCADE)
    resume_file = models.FileField(upload_to='resumes/')
    
class Jobrequirements(models.Model):
    jobprovider = models.ForeignKey(JobproviderRegister, on_delete=models.CASCADE)
    job_title = models.CharField(max_length=50,null=True, blank=True)
    job_description = models.TextField(null=True, blank=True)
    status=models.BooleanField(default=False, null=True)
    
class JobProviderProfile(models.Model):
    jobprovider = models.ForeignKey(JobproviderRegister, on_delete=models.CASCADE)
    company_description = models.TextField(null=True, blank=True)
    company_logo = models.ImageField(upload_to='company_logos/', null=True, blank=True)
    address=models.TextField(null=True, blank=True)
    
    
class JobApplications(models.Model):
    jobseeker = models.ForeignKey(JobseekerRegister, on_delete=models.CASCADE)
    jobprovider = models.ForeignKey(JobproviderRegister, on_delete=models.CASCADE)
    job_requirements = models.ForeignKey(Jobrequirements, on_delete=models.CASCADE)
    selected_status=models.BooleanField(default=False,null=True, blank=True)
    
class Exam(models.Model):
    jobprovider = models.ForeignKey(JobproviderRegister, on_delete=models.CASCADE)
    job_requirements = models.ForeignKey(Jobrequirements, on_delete=models.CASCADE)
    title = models.CharField(max_length=100)
    time_limit = models.IntegerField(help_text="Time limit in minutes")
    created_at = models.DateTimeField(auto_now_add=True)
class ExamQuestion(models.Model):
    exam = models.ForeignKey(Exam, related_name='questions', on_delete=models.CASCADE)
    question_text = models.TextField()
    max_marks = models.IntegerField(help_text="Maximum marks for this question")
    scored=models.IntegerField(default=0,null=True,blank=True)
    
class Answers(models.Model):
    jobseeker = models.ForeignKey(JobseekerRegister, on_delete=models.CASCADE)
    exam_question = models.ForeignKey(ExamQuestion, on_delete=models.CASCADE)
    answer = models.TextField(null=True, blank=True)
    marks_scored = models.IntegerField(default=0,null=True, blank=True)
    exam_status= models.BooleanField(default=False,null=True,blank=True)
    exam_log = models.TextField(blank=True,null=True,default='good')
    
    
    
    