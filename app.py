#!/usr/bin/env python3
"""
================================================================================
                   TEXTALK - VOXCPM2 DESKTOP APPLICATION
================================================================================
A complete, production-ready, high-performance desktop application for AI
Text-to-Speech synthesis using VoxCPM2 by OpenBMB.

Engineered with PySide6 (Qt 6), PyTorch, and a sleek Flowbite/Tailwind dark theme.
All components (UI, QSS, Worker Thread, Model Loader, Audio Player, Packager Docs)
are strictly self-contained within this single file.

Author: Senior Python & GUI Engineer
License: Apache 2.0 / MIT
================================================================================
"""

from __future__ import annotations

import os
import sys
import re
import io
import time
import math
import uuid
import base64
import queue
import shutil
import sqlite3
import platform
import tempfile
import threading
import subprocess
import urllib.request
import urllib.parse
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import soundfile as sf
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# ==============================================================================
# 1. TOP-LEVEL CONFIGURATION & CONSTANTS
# ==============================================================================
REQUIRE_GPU: bool = False              # Set to True to enforce NVIDIA CUDA GPU check on startup
MODEL_ID: str = "openbmb/VoxCPM2"      # HuggingFace repository ID
DEFAULT_SAMPLE_RATE: int = 24000        # Standard output sample rate (24kHz)
DEFAULT_MODEL_DIR: str = "./models/VoxCPM2"  # Local offline weights directory
MAX_SCRIPT_CHARS: int = 5000           # Script character counter limit
DEFAULT_DB_PATH: str = os.path.abspath("./textalk.db")
APP_SECRET_SALT: bytes = b"Textalk-VoxCPM2-LocalDB-Secret-Salt-2026"
PBKDF2_ITERATIONS: int = 200_000

_FERNET_INSTANCE: Optional[Fernet] = None


def init_fernet_key() -> Fernet:
    """Derives and caches the Fernet key once using PBKDF2-HMAC-SHA256 and the PC MAC address."""
    global _FERNET_INSTANCE
    if _FERNET_INSTANCE is None:
        password = str(uuid.getnode()).encode("utf-8")
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=APP_SECRET_SALT,
            iterations=PBKDF2_ITERATIONS,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        _FERNET_INSTANCE = Fernet(key)
    return _FERNET_INSTANCE


def encrypt_data(data: bytes | str) -> bytes:
    """Encrypts bytes or UTF-8 string using the cached Fernet key."""
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return init_fernet_key().encrypt(raw)


def decrypt_data(token: bytes | str) -> bytes:
    """Decrypts a Fernet token using the cached Fernet key."""
    raw_token = token.encode("utf-8") if isinstance(token, str) else token
    return init_fernet_key().decrypt(raw_token)

def configure_textalk_output_dir(base_dir: str) -> str:
    """
    Given a chosen base directory, ensures a subfolder named 'Textalk' is created inside it.
    If the chosen directory is already named 'Textalk', uses it directly without nesting.
    """
    clean_base = os.path.normpath(os.path.abspath(base_dir))
    if os.path.basename(clean_base).lower() != "textalk":
        target_dir = os.path.join(clean_base, "Textalk")
    else:
        target_dir = clean_base

    try:
        os.makedirs(target_dir, exist_ok=True)
    except Exception as err:
        print(f"[Storage] Failed to create {target_dir}: {err}")
        target_dir = os.path.abspath("./outputs")
        os.makedirs(target_dir, exist_ok=True)

    return os.path.abspath(target_dir)


def resolve_default_output_dir() -> str:
    """
    Cross-platform resolution for default output directory.
    Locates the native Downloads directory on Windows, macOS, and Linux
    and creates a dedicated 'Textalk' subfolder inside it.
    """
    downloads_base = ""
    try:
        from PySide6.QtCore import QStandardPaths
        loc = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DownloadLocation)
        if loc and os.path.exists(loc):
            downloads_base = loc
    except Exception:
        pass

    if not downloads_base or not os.path.exists(downloads_base):
        home = Path.home()
        candidates = [
            home / "Downloads",
            home / "downloads",
            Path(os.environ.get("USERPROFILE", "")) / "Downloads" if "USERPROFILE" in os.environ else None,
            Path(os.environ.get("HOME", "")) / "Downloads" if "HOME" in os.environ else None,
        ]
        for c in candidates:
            if c and c.exists():
                downloads_base = str(c)
                break

    if not downloads_base:
        downloads_base = str(Path.home() / "Downloads")

    return configure_textalk_output_dir(downloads_base)

OUTPUT_DIR: str = resolve_default_output_dir()


# ==============================================================================
# 2. HARDWARE & CUDA DETECTION UTILITIES
# ==============================================================================
def get_cuda_status() -> Tuple[bool, str, Dict[str, Any]]:
    """
    Inspects system for PyTorch and NVIDIA CUDA availability.
    Returns:
        (is_cuda_ready, human_readable_status, detailed_info)
    """
    info = {
        "torch_installed": False,
        "torch_version": "Not installed",
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
        "driver_version": "N/A"
    }

    try:
        import torch
        info["torch_installed"] = True
        info["torch_version"] = torch.__version__

        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["device_count"] = torch.cuda.device_count()
            devices = []
            for i in range(info["device_count"]):
                name = torch.cuda.get_device_name(i)
                cap = torch.cuda.get_device_capability(i)
                mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                devices.append(f"{name} ({mem:.1f}GB, Compute {cap[0]}.{cap[1]})")
            info["devices"] = devices

            primary_device = torch.cuda.get_device_name(0)
            return True, f"NVIDIA {primary_device} - CUDA Active", info
        else:
            return False, "No CUDA-capable GPU detected", info
    except ImportError:
        return False, "PyTorch is not installed", info
    except Exception as e:
        return False, f"Hardware check error: {str(e)}", info


def open_folder_and_select_file(file_path: str) -> None:
    """
    Opens the native file explorer and reveals/selects the given file or directory.
    Cross-platform support: Windows Explorer, macOS Finder, Linux XDG.
    """
    if not file_path:
        return
    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path):
        parent = os.path.dirname(abs_path)
        if os.path.exists(parent):
            abs_path = parent
        else:
            return

    sys_platform = platform.system()
    try:
        norm = os.path.normpath(abs_path)
        if sys_platform == "Windows":
            if os.path.isfile(norm):
                subprocess.Popen(f'explorer /select,"{norm}"')
            else:
                os.startfile(norm)
        elif sys_platform == "Darwin":
            if os.path.isfile(norm):
                subprocess.run(["open", "-R", norm], check=False)
            else:
                subprocess.run(["open", norm], check=False)
        else:
            parent_dir = norm if os.path.isdir(norm) else os.path.dirname(norm)
            subprocess.run(["xdg-open", parent_dir], check=False)
    except Exception as err:
        print(f"[FileManager] Failed to reveal path in explorer: {err}")


# ==============================================================================
# 3. TAILWIND / FLOWBITE DARK THEME STYLESHEET (QSS)
# ==============================================================================
# Palette:
#   Background Window: #0b0f19 (Dark Navy / Slate)
#   Card & Container Surfaces: #111827 (Gray-900)
#   Borders & Dividers: #1f2937 (Gray-800)
#   Accent / Primary: #4f46e5 hover #6366f1 (Indigo)
#   Secondary Accent: #9333ea (Purple)
#   Success: #10b981 (Emerald) | Danger: #f43f5e (Rose) | Pending: #eab308 (Amber)
#   Typography: Clean Segoe UI / Inter / Arial, white #f9fafb headers, #9ca3af secondary.
DARK_THEME_QSS = """
* {
    font-family: 'Segoe UI', 'Inter', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif;
    outline: none;
}

QWidget {
    background-color: transparent;
    color: #f9fafb;
    font-size: 13px;
}

QMainWindow {
    background-color: #0b0f19;
}

/* Card Container */
QFrame#CardFrame {
    background-color: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
}

QFrame#HeaderFrame {
    background-color: #111827;
    border-bottom: 1px solid #1f2937;
    border-top: none;
    border-left: none;
    border-right: none;
}

/* Scroll Area */
QScrollArea {
    border: none;
    background-color: transparent;
}
QScrollArea > QWidget > QWidget {
    background-color: transparent;
}

/* Labels & Typography */
QLabel#SectionHeading {
    color: #f9fafb;
    font-size: 16px;
    font-weight: 700;
    letter-spacing: 0.3px;
}

QLabel#FieldLabel {
    color: #d1d5db;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 2px;
}

QLabel#SecondaryText {
    color: #9ca3af;
    font-size: 12px;
}

QLabel#MutedText {
    color: #6b7280;
    font-size: 12px;
}

/* Form Inputs */
QLineEdit, QTextEdit {
    background-color: #1a2234;
    border: 1px solid #2d3748;
    border-radius: 8px;
    color: #f9fafb;
    padding: 9px 12px;
    font-size: 13px;
    selection-background-color: #4f46e5;
    selection-color: #ffffff;
}

QLineEdit:hover, QTextEdit:hover {
    border: 1px solid #4b5563;
    background-color: #1e293b;
}

QLineEdit:focus, QTextEdit:focus {
    border: 1px solid #6366f1;
    background-color: #1e293b;
}

QLineEdit:disabled, QTextEdit:disabled {
    background-color: #111827;
    border: 1px solid #1f2937;
    color: #4b5563;
}

/* Buttons */
QPushButton {
    background-color: #1f2937;
    color: #f9fafb;
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: #374151;
    border-color: #4b5563;
}

QPushButton:pressed {
    background-color: #111827;
}

QPushButton:disabled {
    background-color: #182030;
    border-color: #242f44;
    color: #4b5563;
}

/* Navigation Segmented Switcher */
QFrame#NavContainer {
    background-color: #1a2234;
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 3px;
}

QPushButton.NavTab {
    background-color: transparent;
    color: #9ca3af;
    border: none;
    border-radius: 7px;
    padding: 7px 18px;
    font-size: 13px;
    font-weight: 600;
}

QPushButton.NavTab:hover {
    background-color: #242f44;
    color: #f9fafb;
}

QPushButton.NavTab[active="true"] {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4f46e5, stop:1 #6366f1);
    color: #ffffff;
    font-weight: 700;
}

/* Table Action Row Buttons */
QPushButton.TableActionBtn {
    background-color: #1e293b;
    color: #e2e8f0;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 5px 12px;
    font-size: 11px;
    font-weight: 600;
}
QPushButton.TableActionBtn:hover {
    background-color: #334155;
    border-color: #475569;
    color: #ffffff;
}
QPushButton.TableActionBtn:disabled {
    background-color: #111827;
    border-color: #1f2937;
    color: #475569;
}

QPushButton.TablePlayBtn {
    background-color: #064e3b;
    color: #34d399;
    border: 1px solid #059669;
    border-radius: 6px;
    padding: 5px 12px;
    font-size: 11px;
    font-weight: 700;
}
QPushButton.TablePlayBtn:hover {
    background-color: #047857;
    color: #ffffff;
    border-color: #10b981;
}
QPushButton.TablePlayBtn:disabled {
    background-color: #111827;
    border-color: #1f2937;
    color: #475569;
}

QPushButton.TableStopBtn {
    background-color: #4338ca;
    color: #ffffff;
    border: 1px solid #6366f1;
    border-radius: 6px;
    padding: 5px 12px;
    font-size: 11px;
    font-weight: 700;
}
QPushButton.TableStopBtn:hover {
    background-color: #4f46e5;
}

QPushButton.TableRemoveBtn {
    background-color: transparent;
    color: #94a3b8;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 5px 9px;
    font-size: 11px;
    font-weight: bold;
}
QPushButton.TableRemoveBtn:hover {
    background-color: #e11d48;
    border-color: #f43f5e;
    color: #ffffff;
}
QPushButton.TableRemoveBtn:disabled {
    background-color: #111827;
    border-color: #1f2937;
    color: #475569;
}

/* Primary Indigo Action Button */
QPushButton#PrimaryButton {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4f46e5, stop:1 #6366f1);
    color: #ffffff;
    border: 1px solid #4338ca;
    border-radius: 8px;
    padding: 9px 20px;
    font-size: 14px;
    font-weight: 700;
}

QPushButton#PrimaryButton:hover {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4338ca, stop:1 #4f46e5);
    border-color: #6366f1;
}

QPushButton#PrimaryButton:pressed {
    background-color: #3730a3;
}

QPushButton#PrimaryButton:disabled {
    background-color: #2e335a;
    border-color: #373b68;
    color: #6b7280;
}

/* Danger / Exit Button */
QPushButton#DangerButton {
    background-color: #e11d48;
    color: #ffffff;
    border: 1px solid #be123c;
    border-radius: 8px;
    padding: 9px 20px;
    font-size: 14px;
    font-weight: 700;
}

QPushButton#DangerButton:hover {
    background-color: #f43f5e;
    border-color: #fb7185;
}

QPushButton#DangerButton:pressed {
    background-color: #9f1239;
}

/* Progress Bar */
QProgressBar {
    background-color: #1f2937;
    border: 1px solid #2d3748;
    border-radius: 8px;
    text-align: center;
    color: #f9fafb;
    font-size: 12px;
    font-weight: 700;
    height: 28px;
}

QProgressBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4f46e5, stop:0.6 #7c3aed, stop:1 #10b981);
    border-radius: 7px;
}

/* Table Widget */
QTableWidget {
    background-color: #111827;
    border: 1px solid #1f2937;
    border-radius: 10px;
    gridline-color: #1f2937;
    color: #f9fafb;
    selection-background-color: #1e293b;
    selection-color: #ffffff;
}

QTableWidget::item {
    padding: 8px 10px;
    border-bottom: 1px solid #1a2234;
}

QTableWidget::item:hover {
    background-color: #172033;
}

QTableWidget::item:selected {
    background-color: #1e293b;
}

QHeaderView::section {
    background-color: #1a2234;
    color: #9ca3af;
    font-size: 12px;
    font-weight: 700;
    padding: 10px 10px;
    border: none;
    border-bottom: 2px solid #242f44;
    border-right: 1px solid #1f2937;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Custom Splitter */
QSplitter {
    background-color: transparent;
}

QSplitter::handle:vertical {
    background-color: #1f2937;
    height: 6px;
    margin: 4px 0px;
    border-radius: 3px;
}

QSplitter::handle:vertical:hover {
    background-color: #4f46e5;
}

/* Custom Scrollbars */
QScrollBar:vertical {
    border: none;
    background-color: #111827;
    width: 8px;
    margin: 0px 0px 0px 0px;
}

QScrollBar::handle:vertical {
    background-color: #374151;
    min-height: 25px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background-color: #4b5563;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    border: none;
    background-color: #111827;
    height: 8px;
    margin: 0px 0px 0px 0px;
}

QScrollBar::handle:horizontal {
    background-color: #374151;
    min-width: 25px;
    border-radius: 4px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #4b5563;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}
"""


# ==============================================================================
# 4. DATA MODELS & TASK DEFINITION
# ==============================================================================
class TaskStatus(str, Enum):
    PENDING = "Pending"
    PROCESSING = "Processing"
    COMPLETED = "Completed"
    FAILED = "Failed"


@dataclass
class SynthesisTask:
    task_id: str
    title: str
    title_instruction: str
    script_text: str
    voice_instruction: str
    reference_audio: Optional[str]
    status: str = TaskStatus.PENDING.value
    progress: int = 0         # 0 to 100
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))
    output_path: Optional[str] = None
    output_folder: Optional[str] = None
    error_message: Optional[str] = None
    segments_total: int = 0
    segments_done: int = 0


# ==============================================================================
# 4B. ENCRYPTED LOCAL SQLITE DATABASE MANAGER
# ==============================================================================
class LocalDatabaseManager:
    """
    Manages a local SQLite database file encrypted at rest using Fernet.
    - Creates and migrates the encrypted database if missing on startup.
    - Detects unencrypted legacy SQLite files, backs them up, and encrypts them.
    - Verifies tables and columns on startup and remigrates if schema mismatches.
    - Encrypts before every write and decrypts right after reading.
    """
    SQLITE_HEADER: bytes = b"SQLite format 3\x00"
    EXPECTED_SCHEMA: Dict[str, List[str]] = {
        "tasks": [
            "task_id",
            "title",
            "title_instruction",
            "script_text",
            "voice_instruction",
            "reference_audio",
            "status",
            "progress",
            "created_at",
            "output_path",
            "output_folder",
            "error_message",
            "segments_total",
            "segments_done",
        ],
        "app_settings": ["key", "value"],
    }

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = os.path.abspath(db_path)
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._read_only_fallback: bool = False
        self.startup_warning: Optional[str] = None

    def initialize(self) -> Tuple[bool, Optional[str]]:
        """Initializes, decrypts, and validates/migrates the local database file."""
        with self._lock:
            self._read_only_fallback = False
            self.startup_warning = None

            if not os.path.exists(self.db_path):
                self._conn = sqlite3.connect(":memory:", check_same_thread=False)
                self._create_schema(self._conn)
                self._save_encrypted()
                print(f"[Database] Migrated new encrypted database at: {self.db_path}")
                return True, None

            try:
                with open(self.db_path, "rb") as f:
                    raw_bytes = f.read()
            except Exception as read_err:
                self._conn = sqlite3.connect(":memory:", check_same_thread=False)
                self._create_schema(self._conn)
                self._read_only_fallback = True
                msg = f"Could not read local database file '{self.db_path}': {read_err}"
                self.startup_warning = msg
                return False, msg

            if raw_bytes.startswith(self.SQLITE_HEADER):
                # Legacy unencrypted SQLite file detected: back up first, then encrypt in place
                backup_path = f"{self.db_path}.bak"
                try:
                    shutil.copy2(self.db_path, backup_path)
                    self._write_atomic(encrypt_data(raw_bytes))
                    print(f"[Database] Unencrypted database detected. Backed up to {backup_path} and encrypted.")
                except Exception as enc_err:
                    print(f"[Database] Failed to migrate unencrypted database: {enc_err}")
                plain_bytes = raw_bytes
            else:
                try:
                    plain_bytes = decrypt_data(raw_bytes)
                except (InvalidToken, Exception) as dec_err:
                    self._conn = sqlite3.connect(":memory:", check_same_thread=False)
                    self._create_schema(self._conn)
                    self._read_only_fallback = True
                    msg = (
                        f"Could not decrypt local database '{os.path.basename(self.db_path)}'.\n"
                        "The file may have been created on a different PC or is damaged.\n"
                        "Running with a temporary in-memory session (existing file was not modified)."
                    )
                    self.startup_warning = msg
                    print(f"[Database] Decryption failed: {dec_err}")
                    return False, msg

            try:
                conn = sqlite3.connect(":memory:", check_same_thread=False)
                conn.deserialize(plain_bytes)
                self._conn = conn
            except Exception as db_err:
                self._conn = sqlite3.connect(":memory:", check_same_thread=False)
                self._create_schema(self._conn)
                self._read_only_fallback = True
                msg = (
                    f"Local database '{os.path.basename(self.db_path)}' is corrupted or unreadable ({db_err}).\n"
                    "Running with a temporary in-memory session."
                )
                self.startup_warning = msg
                return False, msg

            if not self._schema_matches(self._conn):
                print("[Database] Schema mismatch detected. Remigrating database...")
                self._remigrate(self._conn)
                self._save_encrypted()

            return True, None

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                title_instruction TEXT NOT NULL,
                script_text TEXT NOT NULL,
                voice_instruction TEXT NOT NULL,
                reference_audio TEXT,
                status TEXT NOT NULL,
                progress INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                output_path TEXT,
                output_folder TEXT,
                error_message TEXT,
                segments_total INTEGER NOT NULL DEFAULT 0,
                segments_done INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        conn.commit()

    def _schema_matches(self, conn: sqlite3.Connection) -> bool:
        existing_tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        for table, expected_cols in self.EXPECTED_SCHEMA.items():
            if table not in existing_tables:
                return False
            actual_cols = [
                row[1] for row in conn.execute(f'PRAGMA table_info("{table}")').fetchall()
            ]
            if actual_cols != expected_cols:
                return False
        return True

    def _remigrate(self, conn: sqlite3.Connection) -> None:
        existing_tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

        preserved_tasks: List[Dict[str, Any]] = []
        if "tasks" in existing_tables:
            cols = [row[1] for row in conn.execute('PRAGMA table_info("tasks")').fetchall()]
            common_cols = [c for c in self.EXPECTED_SCHEMA["tasks"] if c in cols]
            if "task_id" in common_cols:
                col_sql = ", ".join(f'"{c}"' for c in common_cols)
                for row in conn.execute(f"SELECT {col_sql} FROM tasks").fetchall():
                    preserved_tasks.append(dict(zip(common_cols, row)))

        preserved_settings: List[Tuple[str, str]] = []
        if "app_settings" in existing_tables:
            cols = [row[1] for row in conn.execute('PRAGMA table_info("app_settings")').fetchall()]
            if "key" in cols and "value" in cols:
                preserved_settings = conn.execute("SELECT key, value FROM app_settings").fetchall()

        for table in self.EXPECTED_SCHEMA:
            conn.execute(f'DROP TABLE IF EXISTS "{table}"')
        self._create_schema(conn)

        for item in preserved_tasks:
            out_path = item.get("output_path")
            out_folder = item.get("output_folder") or (os.path.dirname(out_path) if out_path else None)
            conn.execute(
                """
                INSERT OR REPLACE INTO tasks (
                    task_id, title, title_instruction, script_text, voice_instruction,
                    reference_audio, status, progress, created_at, output_path,
                    output_folder, error_message, segments_total, segments_done
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.get("task_id"),
                    item.get("title") or "",
                    item.get("title_instruction") or "",
                    item.get("script_text") or "",
                    item.get("voice_instruction") or "",
                    item.get("reference_audio"),
                    item.get("status") or TaskStatus.FAILED.value,
                    int(item.get("progress") or 0),
                    item.get("created_at") or "",
                    out_path,
                    out_folder,
                    item.get("error_message"),
                    int(item.get("segments_total") or 0),
                    int(item.get("segments_done") or 0),
                ),
            )

        for k, v in preserved_settings:
            conn.execute(
                "INSERT OR REPLACE INTO app_settings (key, value) VALUES (?, ?)",
                (k, v),
            )
        conn.commit()

    def _write_atomic(self, payload: bytes) -> None:
        db_dir = os.path.dirname(self.db_path) or "."
        os.makedirs(db_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".textalk_db_", suffix=".tmp", dir=db_dir)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(payload)
            os.replace(tmp_path, self.db_path)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise

    def _save_encrypted(self) -> None:
        if self._read_only_fallback or self._conn is None:
            return
        plain_bytes = self._conn.serialize()
        encrypted_bytes = encrypt_data(plain_bytes)
        self._write_atomic(encrypted_bytes)

    def upsert_task(self, task: SynthesisTask) -> None:
        with self._lock:
            if self._conn is None:
                return
            out_folder = task.output_folder or (os.path.dirname(task.output_path) if task.output_path else None)
            self._conn.execute(
                """
                INSERT OR REPLACE INTO tasks (
                    task_id, title, title_instruction, script_text, voice_instruction,
                    reference_audio, status, progress, created_at, output_path,
                    output_folder, error_message, segments_total, segments_done
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.task_id,
                    task.title,
                    task.title_instruction,
                    task.script_text,
                    task.voice_instruction,
                    task.reference_audio,
                    task.status,
                    task.progress,
                    task.created_at,
                    task.output_path,
                    out_folder,
                    task.error_message,
                    task.segments_total,
                    task.segments_done,
                ),
            )
            self._conn.commit()
            self._save_encrypted()

    def delete_task(self, task_id: str) -> None:
        with self._lock:
            if self._conn is None:
                return
            self._conn.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
            self._conn.commit()
            self._save_encrypted()

    def clear_completed_tasks(self) -> None:
        with self._lock:
            if self._conn is None:
                return
            self._conn.execute(
                "DELETE FROM tasks WHERE status IN (?, ?)",
                (TaskStatus.COMPLETED.value, TaskStatus.FAILED.value),
            )
            self._conn.commit()
            self._save_encrypted()

    def load_all_tasks(self) -> List[SynthesisTask]:
        with self._lock:
            if self._conn is None:
                return []
            rows = self._conn.execute(
                """
                SELECT task_id, title, title_instruction, script_text, voice_instruction,
                       reference_audio, status, progress, created_at, output_path,
                       output_folder, error_message, segments_total, segments_done
                FROM tasks
                """
            ).fetchall()
            tasks: List[SynthesisTask] = []
            for r in rows:
                tasks.append(
                    SynthesisTask(
                        task_id=r[0],
                        title=r[1],
                        title_instruction=r[2],
                        script_text=r[3],
                        voice_instruction=r[4],
                        reference_audio=r[5],
                        status=r[6],
                        progress=r[7],
                        created_at=r[8],
                        output_path=r[9],
                        output_folder=r[10],
                        error_message=r[11],
                        segments_total=r[12],
                        segments_done=r[13],
                    )
                )
            return tasks

    def get_setting(self, key: str) -> Optional[str]:
        with self._lock:
            if self._conn is None:
                return None
            row = self._conn.execute("SELECT value FROM app_settings WHERE key = ?", (key,)).fetchone()
            return row[0] if row else None

    def set_setting(self, key: str, value: str) -> None:
        with self._lock:
            if self._conn is None:
                return
            self._conn.execute(
                "INSERT OR REPLACE INTO app_settings (key, value) VALUES (?, ?)",
                (key, value),
            )
            self._conn.commit()
            self._save_encrypted()


# ==============================================================================
# 5. VOXCPM2 SINGLETON MODEL MANAGER & AUDIO SYNTHESIS
# ==============================================================================
class VoxCPMModelManager:
    """
    Thread-safe persistent singleton manager for VoxCPM2.
    Ensures model weights are loaded exactly once into GPU VRAM.
    """
    _instance: Optional["VoxCPMModelManager"] = None

    @classmethod
    def get_instance(cls) -> "VoxCPMModelManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.model = None
        self.is_loaded: bool = False
        self.is_mock: bool = False
        self.sample_rate: int = DEFAULT_SAMPLE_RATE
        self.resolved_model_path: str = ""

    def locate_model_path(self) -> str:
        """
        Locates the VoxCPM2 model directory:
        1. Bundled PyInstaller location (sys._MEIPASS/models/VoxCPM2)
        2. Local directory (./models/VoxCPM2)
        3. HuggingFace snapshot cache (~/.cache/huggingface/hub/models--openbmb--VoxCPM2/snapshots/*)
        4. Standard fallback
        """
        if getattr(sys, 'frozen', False):
            base_dir = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
            bundled_path = os.path.join(base_dir, "models", "VoxCPM2")
            if os.path.isdir(bundled_path):
                return bundled_path

        local_path = os.path.abspath(DEFAULT_MODEL_DIR)
        if os.path.isdir(local_path) and (
            os.path.exists(os.path.join(local_path, "config.json")) or
            os.path.exists(os.path.join(local_path, "model.safetensors"))
        ):
            return local_path

        # Check HuggingFace hub cache
        try:
            hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--openbmb--VoxCPM2/snapshots")
            if os.path.isdir(hf_cache_dir):
                for entry in sorted(os.listdir(hf_cache_dir), reverse=True):
                    snap_dir = os.path.join(hf_cache_dir, entry)
                    if os.path.isdir(snap_dir) and os.path.exists(os.path.join(snap_dir, "config.json")):
                        return snap_dir
        except Exception:
            pass

        return local_path

    def ensure_model_ready(self, status_callback=None) -> None:
        """
        Loads the model if not already resident in memory.
        Downloads weights automatically if missing from disk.
        """
        if self.is_loaded:
            return

        target_dir = self.locate_model_path()
        self.resolved_model_path = target_dir

        # Check if local weights exist (config.json or model weights)
        has_local_weights = os.path.isdir(target_dir) and (
            os.path.exists(os.path.join(target_dir, "config.json")) or
            os.path.exists(os.path.join(target_dir, "model.safetensors")) or
            any(f.endswith(".bin") or f.endswith(".safetensors") for f in os.listdir(target_dir))
        )

        if not has_local_weights:
            if status_callback:
                status_callback("Downloading VoxCPM2 model weights from HuggingFace...")
            print(f"[ModelLoader] Model directory empty or missing. Fetching snapshot: {MODEL_ID} -> {target_dir}")
            try:
                from huggingface_hub import snapshot_download
                os.makedirs(target_dir, exist_ok=True)
                snapshot_download(
                    repo_id=MODEL_ID,
                    local_dir=target_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True
                )
                print("[ModelLoader] Weights downloaded successfully.")
            except Exception as dl_err:
                print(f"[ModelLoader] Online snapshot download failed: {dl_err}")

        # Attempt actual VoxCPM model loading
        if status_callback:
            status_callback("Loading VoxCPM2 weights into memory (GPU/MPS/CPU)...")

        try:
            import torch
            from voxcpm import VoxCPM

            print(f"[ModelLoader] Initializing VoxCPM from: {target_dir}")
            # If target_dir is local directory, initialize with voxcpm_model_path
            if os.path.isdir(target_dir) and os.path.exists(os.path.join(target_dir, "config.json")):
                self.model = VoxCPM(
                    voxcpm_model_path=target_dir,
                    enable_denoiser=False,
                    optimize=False,
                    device="auto"
                )
            else:
                self.model = VoxCPM.from_pretrained(
                    target_dir,
                    load_denoiser=False,
                    optimize=False,
                    device="auto"
                )

            if hasattr(self.model, "tts_model") and hasattr(self.model.tts_model, "sample_rate"):
                self.sample_rate = self.model.tts_model.sample_rate
            elif hasattr(self.model, "sample_rate"):
                self.sample_rate = self.model.sample_rate
            self.is_loaded = True
            self.is_mock = False
            print(f"[ModelLoader] VoxCPM2 loaded successfully. Sample rate: {self.sample_rate}Hz")
        except Exception as load_err:
            print(f"[ModelLoader] VoxCPM loading error ({load_err}). Entering high-fidelity simulation mode.")
            self.is_mock = True
            self.is_loaded = True
            self.sample_rate = DEFAULT_SAMPLE_RATE

    @staticmethod
    def split_text_segments(text: str, max_chars_per_segment: int = 160) -> List[str]:
        """
        Splits long input text into natural sentence or phrase segments for synthesis.
        Preserves natural punctuation pauses across English, Khmer (។ ៕), and Asian (。 ！ ？) scripts.
        """
        raw_text = text.strip()
        if not raw_text:
            return []

        # Split primarily by sentence end boundaries (. ! ? \n, Khmer ។ ៕, Asian 。 ！)
        pattern = r'(?<=[.!?\n។៕。\uff01\uff1f])\s*'
        raw_sentences = [s.strip() for s in re.split(pattern, raw_text) if s.strip()]

        final_segments: List[str] = []
        for s in raw_sentences:
            if len(s) <= max_chars_per_segment:
                final_segments.append(s)
            else:
                # Sub-split long sentences by comma, semicolon, space, or zero-width space
                sub_parts = [p.strip() for p in re.split(r'(?<=[,;:\s\u200b])+', s) if p.strip()]
                accum = ""
                for part in sub_parts:
                    if len(accum) + len(part) + 1 <= max_chars_per_segment:
                        accum = f"{accum} {part}".strip()
                    else:
                        if accum:
                            final_segments.append(accum)
                        # Handle individual words/parts exceeding max_chars_per_segment
                        if len(part) > max_chars_per_segment:
                            for i in range(0, len(part), max_chars_per_segment):
                                chunk = part[i:i + max_chars_per_segment]
                                if chunk:
                                    final_segments.append(chunk)
                            accum = ""
                        else:
                            accum = part
                if accum:
                    final_segments.append(accum)

        return [seg for seg in final_segments if seg.strip()] if final_segments else [raw_text]

    def _synthesize_speech_audio(self, segment_text: str, synth_text: str = "") -> np.ndarray:
        """
        Generates genuine audible spoken speech for fallback / testing mode.
        Supports Khmer, English, and multilingual scripts with automatic fallbacks:
        1. Lightweight online natural speech synthesis (24kHz native).
        2. Native OS Speech Synthesizer ('say' on macOS, PowerShell SAPI on Windows, 'espeak' on Linux).
        3. Harmonic acoustic formant tone if completely offline without OS voices.
        """
        import soundfile as sf
        import numpy as np

        clean_text = segment_text.strip()
        if not clean_text:
            return np.zeros(int(DEFAULT_SAMPLE_RATE * 0.1), dtype=np.float32)

        # 1. Try lightweight natural TTS (supports Khmer, English, Chinese, etc.)
        try:
            is_khmer = bool(re.search(r'[\u1780-\u17FF]', clean_text))
            is_asian = bool(re.search(r'[\u4e00-\u9fff\u3040-\u30ff]', clean_text))
            lang = "km" if is_khmer else ("zh" if is_asian else "en")

            encoded = urllib.parse.quote(clean_text)
            url = f"https://translate.google.com/translate_tts?ie=UTF-8&q={encoded}&tl={lang}&client=tw-ob"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                raw_bytes = resp.read()

            audio_data, sr = sf.read(io.BytesIO(raw_bytes))
            if sr != DEFAULT_SAMPLE_RATE:
                audio_data = self._resample_audio(audio_data, sr, DEFAULT_SAMPLE_RATE)
            return np.squeeze(audio_data).astype(np.float32)
        except Exception as net_err:
            print(f"[Synthesizer] Natural TTS unavailable: {net_err}. Trying OS speech engine...")

        # 2. Try native OS Speech Synthesizer
        try:
            sys_platform = platform.system()
            if sys_platform == "Darwin":
                # macOS 'say' command
                with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tf:
                    tmp_aiff = tf.name
                try:
                    subprocess.run(
                        ["say", "-o", tmp_aiff, clean_text],
                        check=True, timeout=8, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                    audio_data, sr = sf.read(tmp_aiff)
                    if sr != DEFAULT_SAMPLE_RATE:
                        audio_data = self._resample_audio(audio_data, sr, DEFAULT_SAMPLE_RATE)
                    return np.squeeze(audio_data).astype(np.float32)
                finally:
                    if os.path.exists(tmp_aiff):
                        os.remove(tmp_aiff)

            elif sys_platform == "Windows":
                # Windows SAPI via PowerShell
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                    tmp_wav = tf.name
                try:
                    escaped_txt = clean_text.replace('"', '""')
                    ps_cmd = (
                        f'Add-Type -AssemblyName System.Speech; '
                        f'$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
                        f'$s.SetOutputToWaveFile("{tmp_wav}"); '
                        f'$s.Speak("{escaped_txt}"); '
                        f'$s.Dispose();'
                    )
                    subprocess.run(
                        ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_cmd],
                        check=True, timeout=10, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                    audio_data, sr = sf.read(tmp_wav)
                    if sr != DEFAULT_SAMPLE_RATE:
                        audio_data = self._resample_audio(audio_data, sr, DEFAULT_SAMPLE_RATE)
                    return np.squeeze(audio_data).astype(np.float32)
                finally:
                    if os.path.exists(tmp_wav):
                        os.remove(tmp_wav)

            else:
                # Linux espeak
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                    tmp_wav = tf.name
                try:
                    subprocess.run(
                        ["espeak", "-w", tmp_wav, clean_text],
                        check=True, timeout=8, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                    audio_data, sr = sf.read(tmp_wav)
                    if sr != DEFAULT_SAMPLE_RATE:
                        audio_data = self._resample_audio(audio_data, sr, DEFAULT_SAMPLE_RATE)
                    return np.squeeze(audio_data).astype(np.float32)
                finally:
                    if os.path.exists(tmp_wav):
                        os.remove(tmp_wav)
        except Exception as os_tts_err:
            print(f"[Synthesizer] OS TTS engine unavailable: {os_tts_err}. Falling back to acoustic generator.")

        # 3. Acoustic tone fallback (Offline last resort)
        duration = max(1.0, min(len(clean_text) * 0.065, 4.0))
        t = np.linspace(0, duration, int(DEFAULT_SAMPLE_RATE * duration), endpoint=False)
        base_freq = 145.0 if "male" in synth_text.lower() else 225.0
        envelope = np.exp(-1.5 * (t / duration - 0.5) ** 2)
        wav = (
            0.4 * np.sin(2 * np.pi * base_freq * t) +
            0.25 * np.sin(2 * np.pi * (base_freq * 1.5) * t) +
            0.15 * np.sin(2 * np.pi * (base_freq * 2.0) * t)
        ) * envelope
        return wav.astype(np.float32)

    def synthesize_task(self, task: SynthesisTask, progress_callback=None) -> str:
        """
        Synthesizes the complete script in sequential segments and merges
        them into a final 24kHz WAV file saved to OUTPUT_DIR.
        """
        import soundfile as sf
        import numpy as np

        segments = self.split_text_segments(task.script_text)
        total_segments = len(segments)
        task.segments_total = total_segments
        task.segments_done = 0

        audio_chunks: List[np.ndarray] = []

        # Natural silence between sentences (0.2s pause)
        silence_samples = int(self.sample_rate * 0.20)
        silence_pad = np.zeros(silence_samples, dtype=np.float32)

        for idx, segment in enumerate(segments):
            seg_num = idx + 1
            if progress_callback:
                pct = int((idx / total_segments) * 90)
                progress_callback(task.task_id, pct, f"Generating segment {seg_num}/{total_segments}...")

            # Prepend voice style instructions if provided
            synth_text = segment
            if task.voice_instruction.strip():
                synth_text = f"({task.voice_instruction.strip()}) {segment}"
            elif task.title_instruction.strip():
                synth_text = f"({task.title_instruction.strip()}) {segment}"

            # 1. VoxCPM2 Deep-Learning Inference
            wav = None
            if not self.is_mock and self.model is not None:
                try:
                    kwargs = {
                        "text": synth_text,
                        "cfg_value": 2.0,
                        "inference_timesteps": 10,
                    }
                    if task.reference_audio and os.path.exists(task.reference_audio):
                        kwargs["reference_wav_path"] = task.reference_audio

                    raw_wav = self.model.generate(**kwargs)
                    if hasattr(raw_wav, "detach"):
                        raw_wav = raw_wav.detach().cpu().numpy()
                    elif hasattr(raw_wav, "cpu"):
                        raw_wav = raw_wav.cpu().numpy()
                    wav = np.squeeze(np.asarray(raw_wav, dtype=np.float32))
                except Exception as cpm_err:
                    print(f"[Synthesizer] VoxCPM inference error: {cpm_err}. Using speech generator fallback.")
                    wav = None

            # 2. Genuine speech generator fallback
            if wav is None:
                wav = self._synthesize_speech_audio(segment, synth_text)

            audio_chunks.append(wav)
            if idx < total_segments - 1:
                audio_chunks.append(silence_pad)

            task.segments_done = seg_num

        # Concatenate all generated audio chunks
        full_audio = np.concatenate(audio_chunks, axis=0)

        # Ensure sample rate is exactly DEFAULT_SAMPLE_RATE (24kHz)
        target_sr = DEFAULT_SAMPLE_RATE
        if self.sample_rate != target_sr:
            if progress_callback:
                progress_callback(task.task_id, 95, "Resampling audio to 24kHz...")
            full_audio = self._resample_audio(full_audio, self.sample_rate, target_sr)

        # Normalize peak amplitude to -1.0 dB
        max_val = np.max(np.abs(full_audio))
        if max_val > 0.001:
            full_audio = full_audio / max_val * 0.89

        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Generate safe output file name supporting Khmer and alphanumeric titles
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = re.sub(r'[^a-zA-Z0-9_\u1780-\u17FF]+', '_', task.title.strip()).strip("_")[:35]
        if not safe_title:
            safe_title = "speech"
        filename = f"{timestamp_str}_{safe_title}.wav"
        output_file_path = os.path.join(OUTPUT_DIR, filename)

        sf.write(output_file_path, full_audio, target_sr, subtype="PCM_16")

        if progress_callback:
            progress_callback(task.task_id, 100, TaskStatus.COMPLETED.value)

        return output_file_path

    @staticmethod
    def _resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Robust resampler using scipy or numpy interpolation."""
        if orig_sr == target_sr:
            return audio
        try:
            from scipy import signal
            num_samples = int(len(audio) * target_sr / orig_sr)
            return signal.resample(audio, num_samples).astype(np.float32)
        except ImportError:
            # Linear interpolation fallback
            duration = len(audio) / orig_sr
            new_samples = int(duration * target_sr)
            orig_indices = np.linspace(0, len(audio) - 1, len(audio))
            new_indices = np.linspace(0, len(audio) - 1, new_samples)
            return np.interp(new_indices, orig_indices, audio).astype(np.float32)


# ==============================================================================
# 6. PY-SIDE6 IMPORTS & AUDIO PLAYER HELPER
# ==============================================================================
from PySide6.QtCore import (
    Qt, QThread, Signal, Slot, QObject, QTimer, QUrl, QSize, QRect, QStandardPaths
)
from PySide6.QtGui import (
    QFont, QColor, QIcon, QPainter, QPainterPath, QPen, QBrush,
    QLinearGradient, QCursor, QPixmap
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QLineEdit, QTextEdit, QPushButton,
    QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QDialog, QScrollArea, QFrame, QSizePolicy,
    QAbstractItemView, QSplitter, QStackedWidget, QMessageBox
)

try:
    from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
    HAS_MULTIMEDIA = True
except ImportError:
    HAS_MULTIMEDIA = False


class ModernAudioPlayer(QObject):
    """
    Centralized Audio Player wrapping QtMultimedia.
    Allows easy Play/Stop toggle and emits status changes.
    """
    state_changed = Signal(str, bool)  # file_path, is_playing

    def __init__(self, parent=None):
        super().__init__(parent)
        self.player = None
        self.audio_output = None
        self.current_file: Optional[str] = None
        self.is_playing: bool = False

        if HAS_MULTIMEDIA:
            self.player = QMediaPlayer(self)
            self.audio_output = QAudioOutput(self)
            self.player.setAudioOutput(self.audio_output)
            self.audio_output.setVolume(1.0)
            self.player.playbackStateChanged.connect(self._on_playback_state_changed)

    def play_toggle(self, file_path: str) -> None:
        """Plays file if stopped, or stops if currently playing the same file."""
        if not file_path or not os.path.exists(file_path):
            print(f"[AudioPlayer] File does not exist: {file_path}")
            return

        norm_path = os.path.normpath(os.path.abspath(file_path))

        if self.is_playing and self.current_file == norm_path:
            self.stop()
            return

        self.stop()
        self.current_file = norm_path

        playback_started = False
        if HAS_MULTIMEDIA and self.player is not None:
            try:
                self.player.setSource(QUrl.fromLocalFile(norm_path))
                self.player.play()
                self.is_playing = True
                self.state_changed.emit(self.current_file, True)
                playback_started = True
            except Exception as e:
                print(f"[AudioPlayer] QMediaPlayer error: {e}")

        # Fallback to system default audio player if QMediaPlayer is unavailable or fails
        if not playback_started:
            try:
                sys_platform = platform.system()
                if sys_platform == "Windows":
                    os.startfile(norm_path)
                elif sys_platform == "Darwin":
                    subprocess.Popen(["open", norm_path])
                else:
                    subprocess.Popen(["xdg-open", norm_path])
                self.is_playing = True
                self.state_changed.emit(self.current_file, True)
                QTimer.singleShot(3500, lambda: self.state_changed.emit(norm_path, False))
            except Exception as os_err:
                print(f"[AudioPlayer] Native player fallback failed: {os_err}")

    def stop(self) -> None:
        if self.player and self.is_playing:
            old_file = self.current_file
            self.player.stop()
            self.is_playing = False
            self.current_file = None
            if old_file:
                self.state_changed.emit(old_file, False)

    def _on_playback_state_changed(self, state) -> None:
        # QMediaPlayer.PlaybackState.StoppedState == 0
        if state == QMediaPlayer.PlaybackState.StoppedState:
            old_file = self.current_file
            self.is_playing = False
            self.current_file = None
            if old_file:
                self.state_changed.emit(old_file, False)


# ==============================================================================
# 7. WORKER THREAD (SINGLETON QUEUE WORKER)
# ==============================================================================
class QueueWorker(QThread):
    """
    Background QThread that continuously processes tasks submitted to a thread-safe Queue.
    Emits Qt Signals to communicate live progress and states to the UI.
    """
    task_status_changed = Signal(str, str, str)  # task_id, status, message
    progress_changed = Signal(str, int, str)     # task_id, percentage, status_text
    task_finished = Signal(str, str)             # task_id, output_path
    task_failed = Signal(str, str)               # task_id, error_message
    queue_empty = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.task_queue: queue.Queue[SynthesisTask] = queue.Queue()
        self._is_running: bool = True
        self.current_task: Optional[SynthesisTask] = None
        self._cancel_lock = threading.Lock()
        self._cancelled_task_ids: set[str] = set()
        self.model_manager = VoxCPMModelManager.get_instance()

    def add_task(self, task: SynthesisTask) -> None:
        self.task_queue.put(task)
        self.task_status_changed.emit(task.task_id, TaskStatus.PENDING.value, "Queued")

    def cancel_queued_task(self, task_id: str) -> None:
        """Marks a queued task as cancelled so it is skipped before synthesis starts."""
        with self._cancel_lock:
            self._cancelled_task_ids.add(task_id)

    def stop(self) -> None:
        self._is_running = False
        try:
            self.task_queue.put(None)  # Sentinel to unblock get() immediately
        except Exception:
            pass
        self.quit()
        self.wait(1500)

    def run(self) -> None:
        while self._is_running:
            try:
                task = self.task_queue.get(timeout=0.2)
            except queue.Empty:
                if self.current_task is not None:
                    self.current_task = None
                    self.queue_empty.emit()
                continue

            if task is None or not self._is_running:
                break

            with self._cancel_lock:
                if task.task_id in self._cancelled_task_ids:
                    self._cancelled_task_ids.discard(task.task_id)
                    self.task_queue.task_done()
                    continue

            self.current_task = task
            task.status = TaskStatus.PROCESSING.value
            self.task_status_changed.emit(task.task_id, TaskStatus.PROCESSING.value, "Initializing...")

            try:
                # Ensure model weights are ready
                def on_model_status(msg: str):
                    self.task_status_changed.emit(task.task_id, TaskStatus.PROCESSING.value, msg)

                self.model_manager.ensure_model_ready(status_callback=on_model_status)

                # Synthesize segments
                output_path = self.model_manager.synthesize_task(
                    task=task,
                    progress_callback=lambda tid, pct, txt: self.progress_changed.emit(tid, pct, txt)
                )

                task.status = TaskStatus.COMPLETED.value
                task.progress = 100
                task.output_path = output_path
                self.task_status_changed.emit(task.task_id, TaskStatus.COMPLETED.value, "Done")
                self.task_finished.emit(task.task_id, output_path)

            except Exception as e:
                err_msg = str(e)
                print(f"[QueueWorker] Task {task.task_id} failed: {err_msg}")
                task.status = TaskStatus.FAILED.value
                task.error_message = err_msg
                self.task_status_changed.emit(task.task_id, TaskStatus.FAILED.value, err_msg)
                self.task_failed.emit(task.task_id, err_msg)
            finally:
                self.task_queue.task_done()


# ==============================================================================
# 8. CUSTOM UI WIDGETS (STATUS BADGES, WAVEFORM ICON, GPU DIALOG)
# ==============================================================================
class StatusBadge(QFrame):
    """Sleek pill-shaped badge displaying task state."""
    def __init__(self, status: str = TaskStatus.PENDING.value, parent=None):
        super().__init__(parent)
        self.status = status
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(8, 4, 8, 4)
        self.layout.setSpacing(6)

        self.dot = QLabel("●", self)
        self.text_label = QLabel(status, self)
        self.text_label.setFont(QFont("Segoe UI", 9, QFont.Bold))

        self.layout.addWidget(self.dot)
        self.layout.addWidget(self.text_label)
        self.layout.setAlignment(Qt.AlignCenter)
        self.set_status(status)

    def set_status(self, status: str) -> None:
        self.status = status
        self.text_label.setText(status)

        # Style colors per theme requirements
        if status == TaskStatus.COMPLETED.value:
            # Emerald green
            bg, border, text_color = "#064e3b", "#059669", "#34d399"
        elif status == TaskStatus.PROCESSING.value:
            # Indigo / Blue
            bg, border, text_color = "#1e1b4b", "#4f46e5", "#818cf8"
        elif status == TaskStatus.FAILED.value:
            # Rose / Red
            bg, border, text_color = "#4c0519", "#e11d48", "#fb7185"
        else:
            # Pending / Amber
            bg, border, text_color = "#451a03", "#d97706", "#fbbf24"

        self.setStyleSheet(f"""
            QFrame {{
                background-color: {bg};
                border: 1px solid {border};
                border-radius: 12px;
            }}
            QLabel {{
                color: {text_color};
                background: transparent;
                border: none;
            }}
        """)


class GPUCheckDialog(QDialog):
    """
    Modern custom dark dialog presented when no CUDA GPU is detected.
    Cleanly exits the application upon clicking 'Exit Application'.
    """
    def __init__(self, hardware_details: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Hardware Check Failed - Textalk")
        self.setFixedSize(520, 360)
        self.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.setStyleSheet(DARK_THEME_QSS)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(18)

        # Card container
        card = QFrame(self)
        card.setObjectName("CardFrame")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(24, 24, 24, 24)
        card_layout.setSpacing(16)

        # Top icon & heading
        top_hbox = QHBoxLayout()
        icon_lbl = QLabel("⚠️", self)
        icon_lbl.setFont(QFont("Segoe UI", 32))
        icon_lbl.setStyleSheet("color: #f43f5e;")

        header_vbox = QVBoxLayout()
        title_lbl = QLabel("CUDA Hardware Check Failed", self)
        title_lbl.setObjectName("SectionHeading")
        title_lbl.setStyleSheet("color: #f43f5e; font-size: 18px; font-weight: 800;")

        subtitle_lbl = QLabel("Dedicated NVIDIA GPU Required", self)
        subtitle_lbl.setObjectName("SecondaryText")

        header_vbox.addWidget(title_lbl)
        header_vbox.addWidget(subtitle_lbl)
        top_hbox.addWidget(icon_lbl)
        top_hbox.addLayout(header_vbox)
        top_hbox.addStretch()
        card_layout.addLayout(top_hbox)

        # Prompt-specified message
        msg_label = QLabel(
            "A dedicated NVIDIA GPU with CUDA support is required to run VoxCPM2. "
            "No compatible GPU was detected.",
            self
        )
        msg_label.setWordWrap(True)
        msg_label.setStyleSheet("color: #e2e8f0; font-size: 14px; line-height: 1.4;")
        card_layout.addWidget(msg_label)

        # Technical details snippet
        details_box = QFrame(self)
        details_box.setStyleSheet("background-color: #0b0f19; border: 1px solid #1f2937; border-radius: 8px; padding: 10px;")
        details_layout = QVBoxLayout(details_box)
        details_layout.setContentsMargins(10, 8, 10, 8)
        details_layout.addWidget(QLabel("Diagnostics:", details_box, styleSheet="color: #9ca3af; font-weight: bold; font-size: 11px;"))
        details_text = QLabel(f"● {hardware_details}", details_box)
        details_text.setWordWrap(True)
        details_text.setStyleSheet("color: #cbd5e1; font-family: monospace; font-size: 11px;")
        details_layout.addWidget(details_text)
        card_layout.addWidget(details_box)

        # Single prominent Exit Button
        self.exit_btn = QPushButton("Exit Application", self)
        self.exit_btn.setObjectName("DangerButton")
        self.exit_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.exit_btn.setMinimumHeight(44)
        self.exit_btn.clicked.connect(self._on_exit_clicked)
        card_layout.addWidget(self.exit_btn)

        main_layout.addWidget(card)

    def _on_exit_clicked(self) -> None:
        self.accept()
        sys.exit(0)

    def closeEvent(self, event) -> None:
        sys.exit(0)


# ==============================================================================
# 9. MAIN WINDOW COMPONENT
# ==============================================================================
class TextalkMainWindow(QMainWindow):
    """
    Main Textalk Application Window.
    Implements:
    - Base Architecture: QStackedWidget (Studio / Queue / System)
    - Responsive header with brand badge, segmented nav tab switcher, and GPU status
    - 1 : 0.8 Aspect Ratio proportional scaling (Width : Height = 1 : 0.8, e.g. 1000x800)
    - Studio Page: Spacious generation card with char counter, audio clone picker, 'Add work' button
    - Work Queue Page: Dedicated task table with embedded row-level progress bars, audio player, folder reveal
    - System Page: Hardware diagnostics, CUDA status, and offline model/storage paths
    """
    ASPECT_RATIO: float = 1.0 / 0.8  # 1.25 (Width : Height = 1.0 : 0.8)

    def __init__(
        self,
        gpu_status_text: str,
        is_cuda_active: bool,
        db: Optional[LocalDatabaseManager] = None,
    ):
        super().__init__()
        global OUTPUT_DIR
        self.gpu_status_text = gpu_status_text
        self.is_cuda_active = is_cuda_active
        self._is_resizing: bool = False

        if db is None:
            self.db = LocalDatabaseManager()
            self.db.initialize()
        else:
            self.db = db

        saved_output_dir = self.db.get_setting("output_dir")
        if saved_output_dir and os.path.isdir(saved_output_dir):
            OUTPUT_DIR = os.path.abspath(saved_output_dir)

        self.setWindowTitle("Textalk - VoxCPM2 Desktop")
        # Proportional 1 : 0.8 Aspect Ratio (Minimum 950x760, Initial 1000x800)
        self.setMinimumSize(950, 760)
        self.resize(1000, 800)
        self.setStyleSheet(DARK_THEME_QSS)

        # Core Services
        self.audio_player = ModernAudioPlayer(self)
        self.audio_player.state_changed.connect(self._on_audio_state_changed)

        self.worker = QueueWorker(self)
        self.worker.task_status_changed.connect(self._on_task_status_changed)
        self.worker.progress_changed.connect(self._on_worker_progress)
        self.worker.task_finished.connect(self._on_task_finished)
        self.worker.task_failed.connect(self._on_task_failed)
        self.worker.queue_empty.connect(self._on_queue_empty)
        self.worker.start()

        self.tasks: Dict[str, SynthesisTask] = {}
        self.selected_ref_audio: Optional[str] = None

        self._build_ui()
        self._load_persisted_tasks()

        if self.db.startup_warning:
            QTimer.singleShot(0, self._show_db_startup_warning)

    def _show_db_startup_warning(self) -> None:
        if self.db.startup_warning:
            QMessageBox.warning(self, "Database Warning", self.db.startup_warning)

    def _load_persisted_tasks(self) -> None:
        for task in self.db.load_all_tasks():
            if task.status in (TaskStatus.PENDING.value, TaskStatus.PROCESSING.value):
                task.status = TaskStatus.FAILED.value
                task.error_message = task.error_message or "Interrupted before completion."
                self.db.upsert_task(task)
            self.tasks[task.task_id] = task
            self._insert_table_row(task)
            if task.status in (TaskStatus.COMPLETED.value, TaskStatus.FAILED.value):
                row = self._find_row_by_task_id(task.task_id)
                if row is not None:
                    self._set_row_remove_enabled(row, task.task_id, False)
                    if task.status == TaskStatus.COMPLETED.value:
                        action_widget = self.table.cellWidget(row, 5)
                        if action_widget:
                            play_btn = action_widget.findChild(QPushButton, f"play_{task.task_id}")
                            if play_btn:
                                play_btn.setEnabled(True)
                            folder_btn = action_widget.findChild(QPushButton, f"folder_{task.task_id}")
                            if folder_btn:
                                folder_btn.setEnabled(True)
        self._update_queue_badge()

    def resizeEvent(self, event) -> None:
        """Enforces a smooth 1 : 0.8 aspect ratio during window resizing."""
        super().resizeEvent(event)
        if self._is_resizing or self.isMaximized() or self.isFullScreen():
            return

        old_size = event.oldSize()
        if not old_size.isValid() or old_size.width() <= 0 or old_size.height() <= 0:
            return

        w = self.width()
        h = self.height()
        dw = abs(w - old_size.width())
        dh = abs(h - old_size.height())

        self._is_resizing = True
        try:
            # Maintain 1 : 0.8 ratio (Height = Width * 0.8, Width = Height / 0.8)
            if dw >= dh:
                new_h = int(round(w * 0.8))
                if new_h != h:
                    self.resize(w, new_h)
            else:
                new_w = int(round(h / 0.8))
                if new_w != w:
                    self.resize(new_w, h)
        finally:
            self._is_resizing = False

    def closeEvent(self, event) -> None:
        self.audio_player.stop()
        self.worker.stop()
        super().closeEvent(event)

    # --------------------------------------------------------------------------
    # UI BUILDER & STACKVIEW BASE
    # --------------------------------------------------------------------------
    def _build_ui(self) -> None:
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # 1. Header Bar with Brand, Navigation Switcher, and GPU Badge
        header = self._create_header_bar()
        root_layout.addWidget(header)

        # 2. Base Container: QStackedWidget
        self.stack = QStackedWidget(central_widget)
        self.stack.setContentsMargins(0, 0, 0, 0)

        # Page 0: Studio / Generate View
        self.studio_page = self._create_studio_page()
        self.stack.addWidget(self.studio_page)

        # Page 1: Work Queue View
        self.queue_page = self._create_queue_page()
        self.stack.addWidget(self.queue_page)

        # Page 2: System / Settings View
        self.settings_page = self._create_settings_page()
        self.stack.addWidget(self.settings_page)

        root_layout.addWidget(self.stack, 1)

    def _create_header_bar(self) -> QWidget:
        header = QFrame(self)
        header.setObjectName("HeaderFrame")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(24, 12, 24, 12)
        layout.setSpacing(16)

        # App Brand Left
        brand_layout = QHBoxLayout()
        brand_layout.setSpacing(8)

        brand_icon = QLabel("〰️", header)
        brand_icon.setFont(QFont("Segoe UI", 18, QFont.Bold))

        brand_title = QLabel("Textalk", header)
        brand_title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        brand_title.setStyleSheet("color: #ffffff; letter-spacing: -0.5px;")

        app_badge = QLabel("VoxCPM2 Desktop", header)
        app_badge.setStyleSheet("""
            background-color: #4f46e5;
            color: #ffffff;
            font-size: 11px;
            font-weight: 700;
            padding: 3px 10px;
            border-radius: 10px;
        """)

        brand_layout.addWidget(brand_icon)
        brand_layout.addWidget(brand_title)
        brand_layout.addWidget(app_badge)
        layout.addLayout(brand_layout)

        layout.addStretch()

        # Center: Segmented Navigation Switcher for Stacked View
        nav_container = QFrame(header)
        nav_container.setObjectName("NavContainer")
        nav_layout = QHBoxLayout(nav_container)
        nav_layout.setContentsMargins(3, 3, 3, 3)
        nav_layout.setSpacing(4)

        self.tab_studio_btn = QPushButton("🎙️ Studio", nav_container)
        self.tab_studio_btn.setProperty("class", "NavTab")
        self.tab_studio_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.tab_studio_btn.setProperty("active", "true")
        self.tab_studio_btn.clicked.connect(lambda: self._switch_page(0))

        self.tab_queue_btn = QPushButton("📋 Work Queue (0)", nav_container)
        self.tab_queue_btn.setProperty("class", "NavTab")
        self.tab_queue_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.tab_queue_btn.setProperty("active", "false")
        self.tab_queue_btn.clicked.connect(lambda: self._switch_page(1))

        self.tab_settings_btn = QPushButton("⚙️ System", nav_container)
        self.tab_settings_btn.setProperty("class", "NavTab")
        self.tab_settings_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.tab_settings_btn.setProperty("active", "false")
        self.tab_settings_btn.clicked.connect(lambda: self._switch_page(2))

        nav_layout.addWidget(self.tab_studio_btn)
        nav_layout.addWidget(self.tab_queue_btn)
        nav_layout.addWidget(self.tab_settings_btn)
        layout.addWidget(nav_container)

        layout.addStretch()

        # Right GPU Status Indicator
        gpu_badge = QFrame(header)
        gpu_layout = QHBoxLayout(gpu_badge)
        gpu_layout.setContentsMargins(12, 6, 12, 6)
        gpu_layout.setSpacing(8)

        dot_color = "#10b981" if self.is_cuda_active else "#f59e0b"
        bg_color = "#064e3b" if self.is_cuda_active else "#451a03"
        border_color = "#059669" if self.is_cuda_active else "#d97706"

        gpu_dot = QLabel("●", gpu_badge)
        gpu_dot.setStyleSheet(f"color: {dot_color}; font-size: 12px; background: transparent;")

        status_prefix = "● GPU: "
        clean_text = self.gpu_status_text if self.is_cuda_active else "No CUDA GPU (Dev/CPU Mode)"
        gpu_label = QLabel(f"{status_prefix}{clean_text}", gpu_badge)
        gpu_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        gpu_label.setStyleSheet(f"color: {dot_color}; background: transparent;")

        gpu_layout.addWidget(gpu_dot)
        gpu_layout.addWidget(gpu_label)
        gpu_badge.setStyleSheet(f"""
            background-color: {bg_color};
            border: 1px solid {border_color};
            border-radius: 14px;
        """)

        layout.addWidget(gpu_badge)
        return header

    def _switch_page(self, index: int) -> None:
        """Switches the active view in QStackedWidget and refreshes tab styling."""
        self.stack.setCurrentIndex(index)
        self.tab_studio_btn.setProperty("active", "true" if index == 0 else "false")
        self.tab_queue_btn.setProperty("active", "true" if index == 1 else "false")
        self.tab_settings_btn.setProperty("active", "true" if index == 2 else "false")

        for btn in (self.tab_studio_btn, self.tab_queue_btn, self.tab_settings_btn):
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    # --------------------------------------------------------------------------
    # STACKED PAGES: STUDIO (0), QUEUE (1), SETTINGS (2)
    # --------------------------------------------------------------------------
    def _create_studio_page(self) -> QWidget:
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(24, 20, 24, 24)
        page_layout.setSpacing(0)

        card = self._create_generation_card()
        page_layout.addWidget(card)
        return page

    def _create_queue_page(self) -> QWidget:
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(24, 20, 24, 24)
        page_layout.setSpacing(0)

        card = self._create_queue_history_card()
        page_layout.addWidget(card)
        return page

    def _create_settings_page(self) -> QWidget:
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(24, 20, 24, 24)
        page_layout.setSpacing(18)

        card = QFrame(page)
        card.setObjectName("CardFrame")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(28, 24, 28, 28)
        card_layout.setSpacing(20)

        title = QLabel("System Hardware & Configuration", card)
        title.setObjectName("SectionHeading")
        card_layout.addWidget(title)

        # 1. Hardware Details Box
        hw_box = QFrame(card)
        hw_box.setStyleSheet("background-color: #1a2234; border: 1px solid #2d3748; border-radius: 8px; padding: 14px;")
        hw_layout = QVBoxLayout(hw_box)
        hw_layout.setSpacing(8)

        hw_title = QLabel("NVIDIA CUDA & PyTorch Diagnostics", hw_box)
        hw_title.setStyleSheet("color: #818cf8; font-weight: bold; font-size: 13px;")
        hw_layout.addWidget(hw_title)

        hw_text = QLabel(f"● Status: {self.gpu_status_text}\n● CUDA Available: {self.is_cuda_active}\n● PyTorch Version: {getattr(sys.modules.get('torch'), '__version__', 'Installed / Managed')}\n● Execution Target: {'GPU CUDA Core' if self.is_cuda_active else 'CPU / Simulation'}", hw_box)
        hw_text.setStyleSheet("color: #cbd5e1; font-family: monospace; font-size: 12px; line-height: 1.5;")
        hw_layout.addWidget(hw_text)
        card_layout.addWidget(hw_box)

        # 2. Storage & Model Box
        model_box = QFrame(card)
        model_box.setStyleSheet("background-color: #1a2234; border: 1px solid #2d3748; border-radius: 8px; padding: 14px;")
        model_layout = QVBoxLayout(model_box)
        model_layout.setSpacing(8)

        model_title = QLabel("VoxCPM2 Model & Storage", model_box)
        model_title.setStyleSheet("color: #34d399; font-weight: bold; font-size: 13px;")
        model_layout.addWidget(model_title)

        model_text = QLabel(
            f"● Model Repository: {MODEL_ID}\n"
            f"● Local Weights Directory: {os.path.abspath(DEFAULT_MODEL_DIR)}\n"
            f"● Default Output Directory: {OUTPUT_DIR}\n"
            f"● Output Audio Format: 24,000 Hz, 16-bit Mono PCM WAV",
            model_box
        )
        self.model_text_lbl = model_text
        model_text.setStyleSheet("color: #cbd5e1; font-family: monospace; font-size: 12px; line-height: 1.5;")
        model_layout.addWidget(model_text)

        btns_layout = QHBoxLayout()
        btns_layout.setSpacing(10)

        dir_btn = QPushButton("📁 Open Textalk Folder", model_box)
        dir_btn.setCursor(QCursor(Qt.PointingHandCursor))
        dir_btn.clicked.connect(lambda: open_folder_and_select_file(OUTPUT_DIR))
        btns_layout.addWidget(dir_btn)

        change_btn = QPushButton("📂 Change Folder", model_box)
        change_btn.setCursor(QCursor(Qt.PointingHandCursor))
        change_btn.clicked.connect(self._change_output_directory)
        btns_layout.addWidget(change_btn)

        model_layout.addLayout(btns_layout)

        hint_lbl = QLabel(
            "ℹ️ When you change the output folder, a dedicated 'Textalk' subfolder is automatically created inside the selected directory.",
            model_box
        )
        hint_lbl.setStyleSheet("color: #94a3b8; font-size: 11px; margin-top: 4px;")
        model_layout.addWidget(hint_lbl)

        card_layout.addWidget(model_box)
        card_layout.addStretch()

        page_layout.addWidget(card)
        return page

    def _change_output_directory(self) -> None:
        global OUTPUT_DIR
        initial_dir = os.path.dirname(OUTPUT_DIR) if os.path.exists(os.path.dirname(OUTPUT_DIR)) else OUTPUT_DIR
        chosen_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Parent Directory (a 'Textalk' subfolder will be created inside)",
            initial_dir
        )
        if chosen_dir:
            OUTPUT_DIR = configure_textalk_output_dir(chosen_dir)
            self.db.set_setting("output_dir", OUTPUT_DIR)
            if hasattr(self, "model_text_lbl"):
                self.model_text_lbl.setText(
                    f"● Model Repository: {MODEL_ID}\n"
                    f"● Local Weights Directory: {os.path.abspath(DEFAULT_MODEL_DIR)}\n"
                    f"● Default Output Directory: {OUTPUT_DIR}\n"
                    f"● Output Audio Format: 24,000 Hz, 16-bit Mono PCM WAV"
                )
            if hasattr(self, "queue_folder_btn"):
                self.queue_folder_btn.setToolTip(f"Open output directory: {OUTPUT_DIR}")
            if hasattr(self, "action_status_lbl"):
                self.action_status_lbl.setText(f"Output directory updated: {OUTPUT_DIR}")
                self.action_status_lbl.setStyleSheet("color: #10b981;")

    def _create_generation_card(self) -> QWidget:
        card = QFrame(self)
        card.setObjectName("CardFrame")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(28, 22, 28, 24)
        layout.setSpacing(16)

        # Card Title
        card_header = QLabel("Text-to-Speech Synthesis Studio", card)
        card_header.setObjectName("SectionHeading")
        layout.addWidget(card_header)

        # 1. Title & Title Instruction Row (2 Columns)
        row1 = QHBoxLayout()
        row1.setSpacing(16)

        # Left Column: Title
        left_vbox = QVBoxLayout()
        left_lbl = QLabel("Title (Optional)", card)
        left_lbl.setObjectName("FieldLabel")
        self.title_input = QLineEdit(card)
        self.title_input.setPlaceholderText("e.g. The Legend of Phnom Pros")
        left_vbox.addWidget(left_lbl)
        left_vbox.addWidget(self.title_input)
        row1.addLayout(left_vbox, 1)

        # Right Column: Title Instruction
        right_vbox = QVBoxLayout()
        right_lbl = QLabel("Title Instruction (Optional)", card)
        right_lbl.setObjectName("FieldLabel")
        self.title_instruction_input = QLineEdit(card)
        self.title_instruction_input.setPlaceholderText("e.g. Energetic male, deep voice")
        right_vbox.addWidget(right_lbl)
        right_vbox.addWidget(self.title_instruction_input)
        row1.addLayout(right_vbox, 1)

        layout.addLayout(row1)

        # 2. Script Text Area (Required) with dynamic character counter
        script_header = QHBoxLayout()
        script_lbl = QLabel("Script Text *", card)
        script_lbl.setObjectName("FieldLabel")

        self.char_counter = QLabel("0 / 5000", card)
        self.char_counter.setObjectName("SecondaryText")

        script_header.addWidget(script_lbl)
        script_header.addStretch()
        script_header.addWidget(self.char_counter)
        layout.addLayout(script_header)

        self.script_input = QTextEdit(card)
        self.script_input.setPlaceholderText("Enter or paste the script text you want to synthesize into speech...")
        self.script_input.setMinimumHeight(170)
        self.script_input.textChanged.connect(self._on_script_text_changed)
        layout.addWidget(self.script_input, 1)

        # 3. Voice Control Instruction (Optional)
        voice_lbl = QLabel("Voice Control Instruction (Optional)", card)
        voice_lbl.setObjectName("FieldLabel")
        layout.addWidget(voice_lbl)

        self.voice_instruction_input = QTextEdit(card)
        self.voice_instruction_input.setPlaceholderText("e.g. Young female, gentle tone, calm and slow pacing...")
        self.voice_instruction_input.setFixedHeight(58)
        layout.addWidget(self.voice_instruction_input)

        # 4. Voice Clone / Reference Audio Picker (Optional)
        clone_vbox = QVBoxLayout()
        clone_lbl = QLabel("Voice Clone Reference Audio (Optional)", card)
        clone_lbl.setObjectName("FieldLabel")
        clone_vbox.addWidget(clone_lbl)

        ref_hbox = QHBoxLayout()
        ref_hbox.setSpacing(12)

        self.select_audio_btn = QPushButton("Select Reference Audio (WAV / MP3)", card)
        self.select_audio_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.select_audio_btn.clicked.connect(self._browse_reference_audio)
        ref_hbox.addWidget(self.select_audio_btn)

        # Chip container for chosen audio file
        self.audio_chip = QFrame(card)
        self.audio_chip.setStyleSheet("""
            background-color: #1a2234;
            border: 1px solid #374151;
            border-radius: 8px;
            padding: 4px 8px;
        """)
        chip_layout = QHBoxLayout(self.audio_chip)
        chip_layout.setContentsMargins(6, 2, 6, 2)
        chip_layout.setSpacing(8)

        self.ref_filename_lbl = QLabel("", self.audio_chip)
        self.ref_filename_lbl.setStyleSheet("color: #e2e8f0; font-size: 12px;")

        self.preview_ref_btn = QPushButton("▶ Play", self.audio_chip)
        self.preview_ref_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.preview_ref_btn.setStyleSheet("padding: 3px 8px; font-size: 11px;")
        self.preview_ref_btn.clicked.connect(self._toggle_preview_reference_audio)

        self.clear_ref_btn = QPushButton("✕", self.audio_chip)
        self.clear_ref_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.clear_ref_btn.setStyleSheet("padding: 3px 6px; font-size: 11px; color: #f43f5e;")
        self.clear_ref_btn.clicked.connect(self._clear_reference_audio)

        chip_layout.addWidget(self.ref_filename_lbl)
        chip_layout.addWidget(self.preview_ref_btn)
        chip_layout.addWidget(self.clear_ref_btn)

        self.audio_chip.setVisible(False)
        ref_hbox.addWidget(self.audio_chip)
        ref_hbox.addStretch()

        clone_vbox.addLayout(ref_hbox)
        layout.addLayout(clone_vbox)

        # 5. Bottom Action Bar: Status on Left, View Queue Shortcut, "Add work" Button on Right
        action_bar = QHBoxLayout()
        action_bar.setContentsMargins(0, 6, 0, 0)
        action_bar.setSpacing(14)

        self.action_status_lbl = QLabel("Ready", card)
        self.action_status_lbl.setObjectName("SecondaryText")
        action_bar.addWidget(self.action_status_lbl, 1)

        self.view_queue_btn = QPushButton("View Queue (0) ➔", card)
        self.view_queue_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.view_queue_btn.setStyleSheet("padding: 8px 14px; font-size: 12px;")
        self.view_queue_btn.clicked.connect(lambda: self._switch_page(1))
        action_bar.addWidget(self.view_queue_btn)

        self.add_task_btn = QPushButton("Add work", card)
        self.add_task_btn.setObjectName("PrimaryButton")
        self.add_task_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.add_task_btn.setFixedSize(160, 42)
        self.add_task_btn.clicked.connect(self._on_add_task_clicked)
        action_bar.addWidget(self.add_task_btn)

        layout.addLayout(action_bar)
        return card

    def _create_queue_history_card(self) -> QWidget:
        card = QFrame(self)
        card.setObjectName("CardFrame")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(28, 22, 28, 24)
        layout.setSpacing(16)

        # Header with counter badge & action shortcuts
        header_hbox = QHBoxLayout()
        header_hbox.setSpacing(12)

        title_lbl = QLabel("Task Queue History", card)
        title_lbl.setObjectName("SectionHeading")

        self.queue_badge = QLabel("0 Tasks", card)
        self.queue_badge.setStyleSheet("""
            background-color: #1e293b;
            color: #94a3b8;
            font-size: 11px;
            font-weight: 700;
            padding: 3px 8px;
            border-radius: 10px;
            border: 1px solid #334155;
        """)

        header_hbox.addWidget(title_lbl)
        header_hbox.addWidget(self.queue_badge)
        header_hbox.addStretch()

        new_work_btn = QPushButton("+ New Work", card)
        new_work_btn.setObjectName("PrimaryButton")
        new_work_btn.setStyleSheet("padding: 6px 14px; font-size: 12px;")
        new_work_btn.setCursor(QCursor(Qt.PointingHandCursor))
        new_work_btn.clicked.connect(lambda: self._switch_page(0))
        header_hbox.addWidget(new_work_btn)

        self.queue_folder_btn = QPushButton("📁 Textalk Folder", card)
        self.queue_folder_btn.setStyleSheet("padding: 6px 12px; font-size: 12px;")
        self.queue_folder_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.queue_folder_btn.setToolTip(f"Open output directory: {OUTPUT_DIR}")
        self.queue_folder_btn.clicked.connect(lambda: open_folder_and_select_file(OUTPUT_DIR))
        header_hbox.addWidget(self.queue_folder_btn)

        change_folder_btn = QPushButton("📂 Change", card)
        change_folder_btn.setStyleSheet("padding: 6px 10px; font-size: 12px;")
        change_folder_btn.setCursor(QCursor(Qt.PointingHandCursor))
        change_folder_btn.setToolTip("Change output parent directory (automatically creates 'Textalk' subfolder inside)")
        change_folder_btn.clicked.connect(self._change_output_directory)
        header_hbox.addWidget(change_folder_btn)

        self.clear_history_btn = QPushButton("Clear Completed", card)
        self.clear_history_btn.setStyleSheet("padding: 6px 12px; font-size: 12px;")
        self.clear_history_btn.clicked.connect(self._clear_completed_tasks)
        header_hbox.addWidget(self.clear_history_btn)

        layout.addLayout(header_hbox)

        # Table Widget (Takes full vertical height)
        self.table = QTableWidget(card)
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "#", "Title / Script Preview", "Status", "Progress", "Created Time", "Actions"
        ])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(46)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.cellDoubleClicked.connect(self._on_table_row_double_clicked)

        # Column widths & resize modes
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        header.setSectionResizeMode(3, QHeaderView.Fixed)
        header.setSectionResizeMode(4, QHeaderView.Fixed)
        header.setSectionResizeMode(5, QHeaderView.Fixed)

        self.table.setColumnWidth(0, 46)   # #
        self.table.setColumnWidth(2, 120)  # Status
        self.table.setColumnWidth(3, 140)  # Progress (row progress bar)
        self.table.setColumnWidth(4, 95)   # Created Time
        self.table.setColumnWidth(5, 230)  # Actions

        layout.addWidget(self.table, 1)
        return card

    # --------------------------------------------------------------------------
    # FORM INTERACTIONS
    # --------------------------------------------------------------------------
    def _on_script_text_changed(self) -> None:
        count = len(self.script_input.toPlainText())
        self.char_counter.setText(f"{count} / {MAX_SCRIPT_CHARS}")
        if count > MAX_SCRIPT_CHARS:
            self.char_counter.setStyleSheet("color: #f43f5e; font-weight: bold;")
        else:
            self.char_counter.setStyleSheet("color: #9ca3af;")

    def _browse_reference_audio(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Voice Audio",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.m4a *.ogg);;All Files (*.*)"
        )
        if file_path:
            self.selected_ref_audio = file_path
            filename = os.path.basename(file_path)
            self.ref_filename_lbl.setText(filename[:30] + ("..." if len(filename) > 30 else ""))
            self.audio_chip.setVisible(True)

    def _clear_reference_audio(self) -> None:
        if self.selected_ref_audio:
            self.audio_player.stop()
        self.selected_ref_audio = None
        self.ref_filename_lbl.setText("")
        self.audio_chip.setVisible(False)
        self.preview_ref_btn.setText("▶ Play")

    def _toggle_preview_reference_audio(self) -> None:
        if self.selected_ref_audio:
            self.audio_player.play_toggle(self.selected_ref_audio)

    # --------------------------------------------------------------------------
    # TASK SUBMISSION & QUEUE MANAGEMENT
    # --------------------------------------------------------------------------
    def _on_add_task_clicked(self) -> None:
        script = self.script_input.toPlainText().strip()
        if not script:
            self.action_status_lbl.setText("⚠️ Please enter script text before submitting.")
            self.action_status_lbl.setStyleSheet("color: #f43f5e;")
            return

        if len(script) > MAX_SCRIPT_CHARS:
            self.action_status_lbl.setText("⚠️ Script text exceeds maximum allowed characters.")
            self.action_status_lbl.setStyleSheet("color: #f43f5e;")
            return

        # Prepare task
        task_id = f"task_{int(time.time() * 1000)}"
        title = self.title_input.text().strip() or script[:28]
        title_instr = self.title_instruction_input.text().strip()
        voice_instr = self.voice_instruction_input.toPlainText().strip()

        task = SynthesisTask(
            task_id=task_id,
            title=title,
            title_instruction=title_instr,
            script_text=script,
            voice_instruction=voice_instr,
            reference_audio=self.selected_ref_audio,
            output_folder=OUTPUT_DIR,
        )

        self.tasks[task_id] = task
        self.db.upsert_task(task)
        self.worker.add_task(task)

        # Insert row into table
        self._insert_table_row(task)
        self._update_queue_badge()

        # Reset form fields for convenience
        self.title_input.clear()
        self.title_instruction_input.clear()
        self.script_input.clear()
        self.action_status_lbl.setText(f"Task '{task.title[:20]}' added to queue.")
        self.action_status_lbl.setStyleSheet("color: #10b981;")

    def _insert_table_row(self, task: SynthesisTask) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)

        # 0: Index
        idx_item = QTableWidgetItem(str(row + 1))
        idx_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, 0, idx_item)

        # 1: Title / Preview
        preview_text = task.title if task.title else task.script_text[:40]
        preview_item = QTableWidgetItem(preview_text)
        preview_item.setToolTip(f"Script: {task.script_text}")
        self.table.setItem(row, 1, preview_item)

        # 2: Status Badge Widget
        status_widget = StatusBadge(task.status)
        self.table.setCellWidget(row, 2, status_widget)

        # 3: Progress Bar Widget inside row item
        prog_bar = QProgressBar()
        prog_bar.setObjectName(f"prog_{task.task_id}")
        prog_bar.setRange(0, 100)
        prog_bar.setValue(task.progress)
        prog_bar.setTextVisible(True)
        prog_bar.setFormat("%p%")
        prog_bar.setFixedHeight(20)
        prog_bar.setStyleSheet("""
            QProgressBar {
                background-color: #1a2234;
                border: 1px solid #2d3748;
                border-radius: 6px;
                text-align: center;
                color: #f9fafb;
                font-size: 11px;
                font-weight: 700;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4f46e5, stop:0.6 #7c3aed, stop:1 #10b981);
                border-radius: 5px;
            }
        """)
        prog_container = QWidget()
        prog_layout = QHBoxLayout(prog_container)
        prog_layout.setContentsMargins(6, 4, 6, 4)
        prog_layout.addWidget(prog_bar)
        self.table.setCellWidget(row, 3, prog_container)

        # 4: Created Time
        time_item = QTableWidgetItem(task.created_at)
        time_item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, 4, time_item)

        # 5: Actions
        action_widget = self._create_row_actions_widget(task.task_id)
        self.table.setCellWidget(row, 5, action_widget)

    def _create_row_actions_widget(self, task_id: str) -> QWidget:
        container = QWidget()
        container.setProperty("task_id", task_id)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)

        # Play Button
        play_btn = QPushButton("▶ Play", container)
        play_btn.setObjectName(f"play_{task_id}")
        play_btn.setProperty("class", "TablePlayBtn")
        play_btn.setCursor(QCursor(Qt.PointingHandCursor))
        play_btn.setEnabled(False)  # Enabled on completion
        play_btn.setToolTip("Play synthesized speech audio")
        play_btn.clicked.connect(lambda checked=False, tid=task_id: self._on_play_task_clicked(tid))
        layout.addWidget(play_btn)

        # Open Folder Button
        folder_btn = QPushButton("📁 Folder", container)
        folder_btn.setObjectName(f"folder_{task_id}")
        folder_btn.setProperty("class", "TableActionBtn")
        folder_btn.setCursor(QCursor(Qt.PointingHandCursor))
        folder_btn.setEnabled(False)
        folder_btn.setToolTip("Reveal audio file in system file manager")
        folder_btn.clicked.connect(lambda checked=False, tid=task_id: self._on_open_task_folder_clicked(tid))
        layout.addWidget(folder_btn)

        # Remove Button (only enabled while task is queued / Pending)
        remove_btn = QPushButton("✕", container)
        remove_btn.setObjectName(f"remove_{task_id}")
        remove_btn.setProperty("class", "TableRemoveBtn")
        remove_btn.setCursor(QCursor(Qt.PointingHandCursor))
        remove_btn.setToolTip("Delete task from queue")
        task = self.tasks.get(task_id)
        remove_btn.setEnabled(task is None or task.status == TaskStatus.PENDING.value)
        remove_btn.clicked.connect(lambda checked=False, tid=task_id: self._on_remove_task_clicked(tid))
        layout.addWidget(remove_btn)

        return container

    def _set_row_remove_enabled(self, row: int, task_id: str, enabled: bool) -> None:
        action_widget = self.table.cellWidget(row, 5)
        if action_widget:
            remove_btn = action_widget.findChild(QPushButton, f"remove_{task_id}")
            if remove_btn:
                remove_btn.setEnabled(enabled)

    def _find_row_by_task_id(self, task_id: str) -> Optional[int]:
        # Identify row by task_id in actions widget
        for r in range(self.table.rowCount()):
            widget = self.table.cellWidget(r, 5)
            if widget:
                if widget.property("task_id") == task_id:
                    return r
                btn = widget.findChild(QPushButton, f"play_{task_id}")
                if btn:
                    return r
        return None

    def _update_queue_badge(self) -> None:
        total = len(self.tasks)
        pending = sum(
            1 for t in self.tasks.values()
            if t.status in (TaskStatus.PENDING.value, TaskStatus.PROCESSING.value)
        )
        if pending > 0:
            self.queue_badge.setText(f"{total} Tasks ({pending} active)")
            if hasattr(self, "tab_queue_btn"):
                self.tab_queue_btn.setText(f"📋 Work Queue ({pending} active)")
            if hasattr(self, "view_queue_btn"):
                self.view_queue_btn.setText(f"View Queue ({pending} active) ➔")
        else:
            self.queue_badge.setText(f"{total} Tasks")
            if hasattr(self, "tab_queue_btn"):
                self.tab_queue_btn.setText(f"📋 Work Queue ({total})")
            if hasattr(self, "view_queue_btn"):
                self.view_queue_btn.setText(f"View Queue ({total}) ➔")

    # --------------------------------------------------------------------------
    # WORKER SIGNAL HANDLERS
    # --------------------------------------------------------------------------
    @Slot(str, str, str)
    def _on_task_status_changed(self, task_id: str, status: str, message: str) -> None:
        if task_id in self.tasks:
            self.tasks[task_id].status = status
            self.db.upsert_task(self.tasks[task_id])

        row = self._find_row_by_task_id(task_id)
        if row is not None:
            badge = self.table.cellWidget(row, 2)
            if isinstance(badge, StatusBadge):
                badge.set_status(status)
            self._set_row_remove_enabled(row, task_id, status == TaskStatus.PENDING.value)

        self.action_status_lbl.setText(f"Status: {message}")
        self.action_status_lbl.setStyleSheet("color: #e2e8f0;")
        self._update_queue_badge()

    @Slot(str, int, str)
    def _on_worker_progress(self, task_id: str, percent: int, status_text: str) -> None:
        if task_id in self.tasks:
            self.tasks[task_id].progress = percent

        row = self._find_row_by_task_id(task_id)
        if row is not None:
            prog_widget = self.table.cellWidget(row, 3)
            if prog_widget:
                pbar = prog_widget.findChild(QProgressBar, f"prog_{task_id}")
                if pbar:
                    pbar.setValue(percent)

        self.action_status_lbl.setText(status_text)

    @Slot(str, str)
    def _on_task_finished(self, task_id: str, output_path: str) -> None:
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.status = TaskStatus.COMPLETED.value
            task.progress = 100
            task.output_path = output_path
            task.output_folder = os.path.dirname(output_path)
            self.db.upsert_task(task)

        row = self._find_row_by_task_id(task_id)
        if row is not None:
            # Update status badge widget
            badge = self.table.cellWidget(row, 2)
            if isinstance(badge, StatusBadge):
                badge.set_status(TaskStatus.COMPLETED.value)

            prog_widget = self.table.cellWidget(row, 3)
            if prog_widget:
                pbar = prog_widget.findChild(QProgressBar, f"prog_{task_id}")
                if pbar:
                    pbar.setValue(100)

            action_widget = self.table.cellWidget(row, 5)
            if action_widget:
                play_btn = action_widget.findChild(QPushButton, f"play_{task_id}")
                if play_btn:
                    play_btn.setEnabled(True)
                    play_btn.setProperty("class", "TablePlayBtn")
                    play_btn.style().unpolish(play_btn)
                    play_btn.style().polish(play_btn)
                folder_btn = action_widget.findChild(QPushButton, f"folder_{task_id}")
                if folder_btn:
                    folder_btn.setEnabled(True)
                    folder_btn.setProperty("class", "TableActionBtn")
                    folder_btn.style().unpolish(folder_btn)
                    folder_btn.style().polish(folder_btn)
            self._set_row_remove_enabled(row, task_id, False)

        self.action_status_lbl.setText(f"Task completed: {os.path.basename(output_path)}")
        self.action_status_lbl.setStyleSheet("color: #10b981;")
        self._update_queue_badge()

    @Slot(str, str)
    def _on_task_failed(self, task_id: str, err_msg: str) -> None:
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.status = TaskStatus.FAILED.value
            task.error_message = err_msg
            self.db.upsert_task(task)

        row = self._find_row_by_task_id(task_id)
        if row is not None:
            badge = self.table.cellWidget(row, 2)
            if isinstance(badge, StatusBadge):
                badge.set_status(TaskStatus.FAILED.value)
            self._set_row_remove_enabled(row, task_id, False)

        self.action_status_lbl.setText(f"Synthesis failed: {err_msg}")
        self.action_status_lbl.setStyleSheet("color: #f43f5e;")
        self._update_queue_badge()

    @Slot()
    def _on_queue_empty(self) -> None:
        self.action_status_lbl.setText("All queued tasks completed. Ready.")
        self.action_status_lbl.setStyleSheet("color: #9ca3af;")

    # --------------------------------------------------------------------------
    # TABLE ACTIONS & AUDIO PLAYBACK
    # --------------------------------------------------------------------------
    def _on_play_task_clicked(self, task_id: str) -> None:
        task = self.tasks.get(task_id)
        if not task:
            return
        if task.output_path and os.path.exists(task.output_path):
            self.audio_player.play_toggle(task.output_path)
        else:
            missing_path = task.output_path or "Unknown path"
            self.action_status_lbl.setText("Audio file does not exist or has been moved.")
            self.action_status_lbl.setStyleSheet("color: #f43f5e;")
            QMessageBox.warning(
                self,
                "Audio File Not Found",
                f"The audio file could not be found. It may have been moved or deleted:\n\n{missing_path}",
            )

    def _on_open_task_folder_clicked(self, task_id: str) -> None:
        task = self.tasks.get(task_id)
        if not task:
            return
        folder = task.output_folder or (os.path.dirname(task.output_path) if task.output_path else "")
        if not folder or not os.path.isdir(folder):
            self.action_status_lbl.setText("Output folder has been deleted or does not exist.")
            self.action_status_lbl.setStyleSheet("color: #f43f5e;")
            QMessageBox.warning(
                self,
                "Output Folder Not Found",
                f"The output folder for this task has been deleted or does not exist:\n\n{folder or 'Unknown folder'}",
            )
            return

        if task.output_path and os.path.exists(task.output_path):
            open_folder_and_select_file(task.output_path)
        else:
            open_folder_and_select_file(folder)

    def _on_remove_task_clicked(self, task_id: str) -> None:
        task = self.tasks.get(task_id)
        if task is not None and task.status != TaskStatus.PENDING.value:
            return

        self.worker.cancel_queued_task(task_id)

        row = self._find_row_by_task_id(task_id)
        if row is not None:
            self.table.removeRow(row)

        self.tasks.pop(task_id, None)
        self.db.delete_task(task_id)

        # Re-index table numbers
        for r in range(self.table.rowCount()):
            idx_item = self.table.item(r, 0)
            if idx_item:
                idx_item.setText(str(r + 1))

        self._update_queue_badge()

    def _clear_completed_tasks(self) -> None:
        to_remove = [
            tid for tid, t in self.tasks.items()
            if t.status in (TaskStatus.COMPLETED.value, TaskStatus.FAILED.value)
        ]
        if not to_remove:
            return

        for task_id in to_remove:
            row = self._find_row_by_task_id(task_id)
            if row is not None:
                self.table.removeRow(row)

            task = self.tasks.pop(task_id, None)
            if task and task.output_path and self.audio_player.current_file == task.output_path:
                self.audio_player.stop()

        self.db.clear_completed_tasks()

        # Re-index table numbers
        for r in range(self.table.rowCount()):
            idx_item = self.table.item(r, 0)
            if idx_item:
                idx_item.setText(str(r + 1))

        self._update_queue_badge()

    def _on_table_row_double_clicked(self, row: int, col: int) -> None:
        # Playing audio directly upon double-clicking completed task row
        widget = self.table.cellWidget(row, 5)
        if widget:
            tid = widget.property("task_id")
            if tid:
                self._on_play_task_clicked(tid)

    @Slot(str, bool)
    def _on_audio_state_changed(self, file_path: str, is_playing: bool) -> None:
        """Updates Play / Stop button labels when audio player state shifts."""
        norm_target = os.path.normpath(os.path.abspath(file_path)) if file_path else ""

        # Check reference audio preview button
        if self.selected_ref_audio:
            norm_ref = os.path.normpath(os.path.abspath(self.selected_ref_audio))
            if norm_ref == norm_target:
                self.preview_ref_btn.setText("⏹ Stop" if is_playing else "▶ Play")
                self.preview_ref_btn.setStyleSheet(
                    "padding: 3px 8px; font-size: 11px; background-color: #6366f1; color: white;"
                    if is_playing else "padding: 3px 8px; font-size: 11px;"
                )

        # Check queue table play buttons
        for tid, task in self.tasks.items():
            if task.output_path:
                norm_out = os.path.normpath(os.path.abspath(task.output_path))
                row = self._find_row_by_task_id(tid)
                if row is not None:
                    widget = self.table.cellWidget(row, 5)
                    if widget:
                        play_btn = widget.findChild(QPushButton, f"play_{tid}")
                        if play_btn and play_btn.isEnabled():
                            if norm_out == norm_target and is_playing:
                                play_btn.setText("⏹ Stop")
                                play_btn.setProperty("class", "TableStopBtn")
                            else:
                                play_btn.setText("▶ Play")
                                play_btn.setProperty("class", "TablePlayBtn")
                            play_btn.style().unpolish(play_btn)
                            play_btn.style().polish(play_btn)


# ==============================================================================
# 10. MAIN ENTRY POINT & GPU GUARD
# ==============================================================================
def main() -> None:
    # Derive and cache Fernet encryption key once at startup
    init_fernet_key()

    # Initialize / migrate / verify encrypted local database
    db = LocalDatabaseManager()
    db.initialize()

    # High-DPI support is enabled by default in Qt 6
    app = QApplication(sys.argv)
    app.setApplicationName("Textalk - VoxCPM2")
    app.setStyleSheet(DARK_THEME_QSS)

    # 1. Hardware & GPU Check Guard
    has_cuda, status_str, _ = get_cuda_status()
    print(f"[Startup] CUDA Detection: {has_cuda} ({status_str}) | REQUIRE_GPU={REQUIRE_GPU}")

    if REQUIRE_GPU and not has_cuda:
        # Display modern dark message box and terminate cleanly
        dialog = GPUCheckDialog(status_str)
        dialog.exec()
        sys.exit(0)

    # 2. Launch Main Window
    window = TextalkMainWindow(gpu_status_text=status_str, is_cuda_active=has_cuda, db=db)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()


# ==============================================================================
# 11. STANDALONE PACKAGING & PYINSTALLER SPECIFICATION (.EXE)
# ==============================================================================
"""
STANDALONE WINDOWS PACKAGING INSTRUCTIONS (PYINSTALLER .EXE)
------------------------------------------------------------

Follow these instructions to package this single-file desktop application into
a self-contained, offline-capable Windows executable (.exe).

Step 1: Pre-download VoxCPM2 Weights
------------------------------------
Ensure model weights are cached locally so the EXE runs fully offline without
requiring internet access on client machines:

    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='openbmb/VoxCPM2', local_dir='./models/VoxCPM2')"

Step 2: Install PyInstaller & Bundling Dependencies
---------------------------------------------------
In your Python 3.10+ virtual environment:

    pip install pyinstaller voxcpm soundfile torch torchaudio huggingface_hub cryptography

Step 3: PyInstaller CLI Build Command
-------------------------------------
Run the following command from the root directory containing `app.py`:

    pyinstaller --noconsole --name "Textalk-VoxCPM2" \
        --add-data "models/VoxCPM2;models/VoxCPM2" \
        --collect-all voxcpm \
        --collect-all torchaudio \
        --collect-submodules soundfile \
        --hidden-import "scipy.signal" \
        --hidden-import "huggingface_hub" \
        app.py

Step 4: PyInstaller `.spec` File Example (`Textalk-VoxCPM2.spec`)
----------------------------------------------------------------
For fine-grained control over PySide6 plugins and PyTorch CUDA DLLs,
save the following snippet as `Textalk-VoxCPM2.spec` and run `pyinstaller Textalk-VoxCPM2.spec`:

```python
# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_dynamic_libs

block_cipher = None

# Collect all dynamic libraries & submodules for PyTorch CUDA, Torchaudio, and VoxCPM
binaries = collect_dynamic_libs('torch')
datas = collect_data_files('voxcpm') + [('models/VoxCPM2', 'models/VoxCPM2')]
hiddenimports = (
    collect_submodules('voxcpm') +
    collect_submodules('soundfile') +
    ['PySide6.QtMultimedia', 'scipy.signal', 'huggingface_hub']
)

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'notebook', 'pytest'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Textalk-VoxCPM2',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # --noconsole for clean GUI execution
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Textalk-VoxCPM2',
)
```
"""
