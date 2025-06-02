from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import User
from .models import UserProfile

class CustomUserAdmin(UserAdmin):
    # Override the password validation
    def save_model(self, request, obj, form, change):
        if not change:  # Only for new users
            obj.set_password(form.cleaned_data['password'])
        obj.save()

# Unregister the default UserAdmin
admin.site.unregister(User)

# Register our custom UserAdmin
admin.site.register(User, CustomUserAdmin)

# Register UserProfile
admin.site.register(UserProfile)
