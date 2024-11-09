from django.contrib.auth.models import AbstractUser, Group, Permission
from django.db import models

class CustomUser(AbstractUser):
    wallet_address = models.CharField(max_length=100, null=True, blank=True)
    is_doctor = models.BooleanField(default=False)

    # Add related_name to avoid the clash
    groups = models.ManyToManyField(
        Group,
        related_name="customuser_set",  # Avoid name clash
        blank=True,
        help_text='The groups this user belongs to.',
        verbose_name='groups',
    )
    user_permissions = models.ManyToManyField(
        Permission,
        related_name="customuser_set",  # Avoid name clash
        blank=True,
        help_text='Specific permissions for this user.',
        verbose_name='user permissions',
    )

    def __str__(self):
        return self.username
