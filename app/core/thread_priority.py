"""Thread priority control for Apple Silicon Super/Efficiency core scheduling.

macOS uses QoS (Quality of Service) classes to decide which cores run a thread:
  QOS_CLASS_USER_INTERACTIVE (0x21) -> Super cores preferred (highest perf)
  QOS_CLASS_USER_INITIATED   (0x19) -> Super cores preferred
  QOS_CLASS_DEFAULT          (0x15) -> Any core
  QOS_CLASS_UTILITY          (0x11) -> Efficiency cores preferred
  QOS_CLASS_BACKGROUND       (0x09) -> Efficiency cores only

For inference, we want Super cores to maximize throughput.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import threading
from app.core.logging import get_logger

logger = get_logger(__name__)

# macOS QoS class constants (sys/qos.h)
QOS_CLASS_USER_INTERACTIVE = 0x21
QOS_CLASS_USER_INITIATED = 0x19
QOS_CLASS_DEFAULT = 0x15
QOS_CLASS_UTILITY = 0x11
QOS_CLASS_BACKGROUND = 0x09

_libc = None


def _get_libc():
    global _libc
    if _libc is None:
        path = ctypes.util.find_library("c")
        if path:
            _libc = ctypes.CDLL(path, use_errno=True)
    return _libc


def set_thread_qos(qos_class: int = QOS_CLASS_USER_INTERACTIVE) -> bool:
    """Set the current thread's QoS class.

    Args:
        qos_class: One of the QOS_CLASS_* constants.
                   Default is USER_INTERACTIVE for Super cores.

    Returns:
        True if successful, False otherwise.
    """
    libc = _get_libc()
    if libc is None:
        return False

    try:
        func = libc.pthread_set_qos_class_self_np
        func.restype = ctypes.c_int
        func.argtypes = [ctypes.c_uint, ctypes.c_int]
        ret = func(qos_class, 0)
        return ret == 0
    except Exception:
        return False


def set_super_core_priority() -> bool:
    """Set current thread to prefer Super cores (highest performance)."""
    ok = set_thread_qos(QOS_CLASS_USER_INTERACTIVE)
    if ok:
        logger.info("Thread QoS set to USER_INTERACTIVE (Super cores preferred)")
    else:
        logger.warning("Failed to set Super core priority")
    return ok


def set_efficiency_core_priority() -> bool:
    """Set current thread to prefer Efficiency cores (power saving)."""
    ok = set_thread_qos(QOS_CLASS_UTILITY)
    if ok:
        logger.info("Thread QoS set to UTILITY (Efficiency cores preferred)")
    return ok


def apply_to_current_process() -> bool:
    """Apply Super core priority to the main thread.

    Call this during server startup to ensure the inference thread
    runs on Super cores.
    """
    return set_super_core_priority()


class SuperCoreThread(threading.Thread):
    """A thread that runs on Super cores via QoS class."""

    def run(self):
        set_thread_qos(QOS_CLASS_USER_INTERACTIVE)
        super().run()
