from django.contrib import admin

# Register your models here.
from .models import *


admin.site.register(Resumes)
admin.site.register(JobseekerRegister)
admin.site.register(JobproviderRegister)
admin.site.register(JobProviderProfile)
admin.site.register(Jobrequirements)
admin.site.register(JobApplications)
admin.site.register(Exam)
admin.site.register(ExamQuestion)
admin.site.register(Answers)