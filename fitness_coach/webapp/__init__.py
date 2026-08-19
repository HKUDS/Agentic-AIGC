"""Mobile-first web sign-up: verification by email or phone, survey, plan."""

from .auth import AuthService, detect_channel, normalize_email, normalize_phone

__all__ = ["AuthService", "detect_channel", "normalize_email", "normalize_phone"]
