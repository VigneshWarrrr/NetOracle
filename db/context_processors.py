def theme_processor(request):
    if request.user.is_authenticated:
        if hasattr(request.user, 'profile'):
            return {'current_theme': request.user.profile.theme}
    return {'current_theme': 'dark'}
