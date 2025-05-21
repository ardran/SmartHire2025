from django.urls import path
from .views import *


urlpatterns = [
    path('',index,name='index'),
    path('jonseekindex/',jobseekindex,name='jobseekerindex'),
    path('jobprovideindex/',jobprovideindex,name='jobprovideindex'),
    path('jobseeker-register/',jobseekerreg,name='jobseeker-register'),
    path('jobseeker-login/',jobseekerlogin,name='jobseeker-login'),
    path('jobproviderlogin/',jobproviderlogin,name='jobproviderlogin'),
    path('jobproviderreg/',jobproviderreg,name='jobproviderreg'),
    path('jobseekprofile/',jobseekprofile,name='jobseekprofile'),
    path('jobseekereditprofile/',jobseekereditprofile,name='jobseekereditprofile'),
    path('jobproviderprofile/',jobproviderprofile,name='jobproviderprofile'),
    path('jobprovidereditprofile/',jobprovidereditprofile,name='jobprovidereditprofile'),
    path('PostJob/',PostJob,name='PostJob'),
    path('viewjobs/',viewJobs,name='viewjobs'),
    path('viewapplication/',viewappiledjobsprovider,name='viewapplication'),
    path('jobseekApplications/',jobseekApplications,name='jobseekApplications'),
    path('toggle-status/<int:application_id>/',toggle_status, name='toggle_status'),
    path('EditjobReq/<int:jobid>/',EditjobReq, name='editjob'),
    path('Deletejob/<int:jobid>/',Deletejob, name='deletejob'),
    path('apply_job/<int:resid>/',apply_job,name='apply_job'),
    path('list-exams/', list_exams, name='list_exams'),
    path('viewexam/<int:exam_id>/',view_exam_questions, name='view_exam_questions'),
    path('createexamandquestions/', create_exam_and_questions, name='createexamandquestions'),
    path('take-exam/<int:exam_id>/', take_exam, name='take_exam'),
    path('send_exam_links/', send_exam_links_page, name='send_exam_links_page'),
    path('send_exam_email/<int:job_id>/<int:jobseeker_id>/', send_exam_email, name='send_exam_email'),
    path('delete-exam/<int:exmid>/',DeleteExam, name='delete_exam'),
    path('exam_completed/', exam_completed, name='exam_completed'),
    path('exam/<int:exam_id>/results/', exam_results, name='exam_results'),
    path('exam_terminated/', exam_terminated, name='exam_terminated'),
    path('findJob/',findJob,name='findjob'),
    path('logout/',logout,name='logout'),
    
    path('adminindex/',adminindex,name='admin_panel'),
    path('admin_log/',admin_log,name='admin_log'),
    path('jobseekers/', admin_jobseekers, name='admin_jobseekers'),
    path('jobseekersedit/<int:pk>/', edit_jobseeker, name='edit_jobseeker'),
    path('jobseekersdelete/<int:pk>/', delete_jobseeker, name='delete_jobseeker'),
    
    path('jobproviders/', admin_jobproviders, name='admin_jobproviders'),
    path('jobprovidersedit/<int:pk>/', edit_jobprovider, name='edit_jobprovider'),
    path('jobprovidersdelete/<int:pk>/', delete_jobprovider, name='delete_jobprovider'),
    path('admin_logout/',adlogout, name='admin_logout'),
    path('generate_exam_performance_pdf/<exam_id>/<jobseeker_id>/',generate_exam_performance_pdf,name='generate_exam_performance_pdf'),

    path('exam_attendees/<int:exam_id>/', exam_attendees, name='exam_attendees'),
    
    path('send-interview-link/<int:jobseeker_id>/', send_interview_link, name='send_interview_link'),
    
    path('process_question/', process_question, name='process_question'),
    path('chat/', chatbot, name='chat'),
]
