try:
    from libsensorio import core
except ImportError:
    raise ImportError("Run setup.py to build library before importing.")

__all__ = ["libsensorio"]
