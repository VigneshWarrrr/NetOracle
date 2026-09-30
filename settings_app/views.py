from django.shortcuts import render, redirect
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.forms import PasswordChangeForm
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views import View
from django.contrib import messages
from users.models import UserProfile

class SettingsDashboardView(LoginRequiredMixin, View):
    template_name = 'settings_app/dashboard.html'

    def get(self, request):
        password_form = PasswordChangeForm(request.user)
        return render(request, self.template_name, {
            'password_form': password_form,
            'current_theme': request.user.profile.theme
        })

    def post(self, request):
        if 'change_password' in request.POST:
            password_form = PasswordChangeForm(request.user, request.POST)
            if password_form.is_valid():
                user = password_form.save()
                update_session_auth_hash(request, user)
                messages.success(request, 'Your password was successfully updated!')
                return redirect('settings_app:dashboard')
            else:
                messages.error(request, 'Please correct the error below.')
        
        elif 'update_theme' in request.POST:
            theme = request.POST.get('theme')
            if theme in dict(UserProfile.THEME_CHOICES):  # light / dark / system; anything else is ignored
                profile = request.user.profile
                profile.theme = theme
                profile.save()
                messages.success(request, f'Theme updated to {theme.capitalize()}!')
                return redirect('settings_app:dashboard')

        password_form = PasswordChangeForm(request.user)
        return render(request, self.template_name, {
            'password_form': password_form,
            'current_theme': request.user.profile.theme
        })
