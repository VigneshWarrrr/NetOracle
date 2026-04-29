from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.mixins import UserPassesTestMixin, LoginRequiredMixin
from django.views import View
from django.contrib.auth.models import User
from django.contrib import messages
from .models import UserActivity

class AdminOnlyMixin(UserPassesTestMixin):
    def test_func(self):
        return self.request.user.is_superuser or self.request.user.is_staff

class UserManagementDashboardView(LoginRequiredMixin, AdminOnlyMixin, View):
    template_name = 'management/dashboard.html'

    def get(self, request):
        users = User.objects.all().order_by('-date_joined')
        activities = UserActivity.objects.all().order_by('-timestamp')[:20]
        return render(request, self.template_name, {
            'users': users,
            'activities': activities
        })

class UserCreateView(LoginRequiredMixin, AdminOnlyMixin, View):
    def post(self, request):
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        is_staff = request.POST.get('is_staff') == 'on'

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists.")
        else:
            user = User.objects.create_user(username=username, email=email, password=password)
            user.is_staff = is_staff
            user.save()
            UserActivity.objects.create(user=request.user, action="Created User", details=f"Created {username}")
            messages.success(request, f"User {username} created successfully.")
        
        return redirect('management:dashboard')

class UserDeleteView(LoginRequiredMixin, AdminOnlyMixin, View):
    def post(self, request, user_id):
        user = get_object_or_404(User, id=user_id)
        if user == request.user:
            messages.error(request, "You cannot delete yourself.")
        else:
            username = user.username
            user.delete()
            UserActivity.objects.create(user=request.user, action="Deleted User", details=f"Deleted {username}")
            messages.success(request, f"User {username} deleted successfully.")
        
        return redirect('management:dashboard')
