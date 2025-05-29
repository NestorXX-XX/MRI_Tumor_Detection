from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import UserProfile

# Create your views here.

@login_required
def profile(request):
    return render(request, 'users/profile.html')

@login_required
def admin_dashboard(request):
    if not request.user.userprofile.is_admin:
        messages.error(request, 'You do not have permission to access this page.')
        return redirect('home')
    users = UserProfile.objects.all()
    return render(request, 'users/admin_dashboard.html', {'users': users})
