"""
Management command: create_default_users
=========================================
Creates pre-defined users in the database automatically.
Safe to run multiple times — skips users that already exist.

Usage:
    python manage.py create_default_users

Add/edit users in the USERS list below before deploying.
"""

from django.core.management.base import BaseCommand
from django.contrib.auth.models import User


# ─────────────────────────────────────────────────
#  ✏️  EDIT THIS LIST to add/change default users
# ─────────────────────────────────────────────────
USERS = [
    {
        "username":   "admin",
        "password":   "Admin@1234",
        "email":      "admin@healthcare.ai",
        "first_name": "Admin",
        "last_name":  "User",
        "is_staff":   True,
        "is_superuser": True,
    },
    {
        "username":   "doctor",
        "password":   "Doctor@1234",
        "email":      "doctor@healthcare.ai",
        "first_name": "Dr. Arjun",
        "last_name":  "Sharma",
        "is_staff":   False,
        "is_superuser": False,
    },
    {
        "username":   "analyst",
        "password":   "Analyst@1234",
        "email":      "analyst@healthcare.ai",
        "first_name": "Data",
        "last_name":  "Analyst",
        "is_staff":   False,
        "is_superuser": False,
    },
    {
        "username":   "researcher",
        "password":   "Research@1234",
        "email":      "researcher@healthcare.ai",
        "first_name": "Clinical",
        "last_name":  "Researcher",
        "is_staff":   False,
        "is_superuser": False,
    },
]
# ─────────────────────────────────────────────────


class Command(BaseCommand):
    help = "Create default users for the Healthcare AI platform (safe to re-run)"

    def handle(self, *args, **options):
        self.stdout.write("\n" + "="*55)
        self.stdout.write("  Healthcare AI — Default User Setup")
        self.stdout.write("="*55)

        created_count = 0
        skipped_count = 0

        for u in USERS:
            if User.objects.filter(username=u["username"]).exists():
                self.stdout.write(
                    self.style.WARNING(f"  ⚠  SKIP   '{u['username']}' already exists")
                )
                skipped_count += 1
            else:
                user = User.objects.create_user(
                    username=u["username"],
                    password=u["password"],
                    email=u["email"],
                    first_name=u["first_name"],
                    last_name=u["last_name"],
                )
                user.is_staff = u.get("is_staff", False)
                user.is_superuser = u.get("is_superuser", False)
                user.save()

                role = "SUPERUSER" if u.get("is_superuser") else ("STAFF" if u.get("is_staff") else "USER")
                self.stdout.write(
                    self.style.SUCCESS(f"  ✅ CREATED '{u['username']}' [{role}]  pass: {u['password']}")
                )
                created_count += 1

        self.stdout.write("="*55)
        self.stdout.write(
            self.style.SUCCESS(f"  Done! Created: {created_count}  |  Skipped: {skipped_count}")
        )
        self.stdout.write("="*55 + "\n")

        if created_count > 0:
            self.stdout.write("  📋 Login credentials summary:")
            self.stdout.write("  " + "-"*51)
            for u in USERS:
                self.stdout.write(f"  Username : {u['username']}")
                self.stdout.write(f"  Password : {u['password']}")
                self.stdout.write(f"  Role     : {'Admin' if u.get('is_superuser') else 'Staff' if u.get('is_staff') else 'User'}")
                self.stdout.write("  " + "-"*51)
            self.stdout.write("")
