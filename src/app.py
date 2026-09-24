"""
DermaScan - Skin Lesion Classifier
Standalone desktop application (Tkinter + TensorFlow/Keras)
"""

import sys
import os
import json
import threading
import tkinter as tk
from tkinter import filedialog, ttk
from pathlib import Path
import numpy as np

# ── Resource path helper (handles PyInstaller _MEIPASS) ───────────────────────
def resource_path(relative):
    base = getattr(sys, "_MEIPASS", Path(__file__).parent.parent)
    return os.path.join(base, relative)

MODEL_PATH = resource_path("model_weights.keras")
META_PATH  = resource_path("model_meta.json")

# Load metadata
try:
    with open(META_PATH) as f:
        META = json.load(f)
    ACCURACY = META.get("accuracy", "?")
    IMG_SIZE = META.get("img_size", 224)
except Exception:
    ACCURACY = "?"
    IMG_SIZE = 224

# ── Lazy-load heavy imports (faster startup) ───────────────────────────────────
_model = None

def load_model():
    global _model
    if _model is None:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow as tf
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def predict_image(image_path):
    """Returns (label, confidence_pct) or raises."""
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    arr = preprocess_input(np.array(img, dtype=np.float32))
    arr = np.expand_dims(arr, 0)
    model = load_model()
    prob = float(model.predict(arr, verbose=0)[0][0])
    if prob >= 0.5:
        return "MALIGNANT", round(prob * 100, 1)
    else:
        return "BENIGN", round((1 - prob) * 100, 1)


# ── Colours ────────────────────────────────────────────────────────────────────
BG        = "#0f0f0f"
SURFACE   = "#1a1a1a"
SURFACE2  = "#242424"
ACCENT    = "#00c896"
RED       = "#ff4d6d"
TEXT      = "#f0f0f0"
SUBTEXT   = "#888888"
BORDER    = "#333333"

FONT_TITLE  = ("Segoe UI", 22, "bold")
FONT_BODY   = ("Segoe UI", 11)
FONT_SMALL  = ("Segoe UI", 9)
FONT_RESULT = ("Segoe UI", 28, "bold")
FONT_CONF   = ("Segoe UI", 14)
FONT_MONO   = ("Consolas", 10)


class DermaApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DermaScan — Skin Lesion Classifier")
        self.geometry("720x620")
        self.minsize(640, 560)
        self.configure(bg=BG)
        self.resizable(True, True)

        # Try to set window icon
        try:
            ico = resource_path(os.path.join("assets", "icon.ico"))
            self.iconbitmap(ico)
        except Exception:
            pass

        self._image_path = None
        self._build_ui()
        self._warm_up()

    # ── Warm-up model in background ────────────────────────────────────────────
    def _warm_up(self):
        def _load():
            try:
                load_model()
                self.after(0, lambda: self._set_status("Model ready. Drop an image or click Browse."))
            except Exception as e:
                self.after(0, lambda: self._set_status(f"⚠ Model load error: {e}", error=True))
        threading.Thread(target=_load, daemon=True).start()

    # ── UI build ───────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=24, pady=(20, 0))

        tk.Label(hdr, text="DermaScan", font=FONT_TITLE, bg=BG, fg=ACCENT).pack(side="left")
        tk.Label(
            hdr,
            text=f"Skin Lesion Classifier  •  Val accuracy {ACCURACY}%",
            font=FONT_SMALL, bg=BG, fg=SUBTEXT,
        ).pack(side="left", padx=(12, 0), pady=(8, 0))

        sep = tk.Frame(self, height=1, bg=BORDER)
        sep.pack(fill="x", padx=24, pady=(12, 0))

        # Drop zone
        self._drop_frame = tk.Frame(
            self, bg=SURFACE, bd=0, highlightthickness=2,
            highlightbackground=BORDER, highlightcolor=ACCENT,
        )
        self._drop_frame.pack(fill="both", expand=True, padx=24, pady=16)

        self._canvas = tk.Canvas(self._drop_frame, bg=SURFACE, highlightthickness=0)
        self._canvas.pack(fill="both", expand=True)

        self._canvas.bind("<Button-1>", lambda e: self._browse())
        self._drop_frame.bind("<Enter>", self._on_hover)
        self._drop_frame.bind("<Leave>", self._on_leave)

        self._draw_placeholder()

        # Drag-and-drop via tkinterdnd2 (optional, graceful fallback)
        try:
            self.drop_target_register = getattr(self, "drop_target_register", None)
            self._drop_frame.drop_target_register("DND_Files")  # type: ignore
            self._drop_frame.dnd_bind("<<Drop>>", self._on_drop)  # type: ignore
        except Exception:
            pass

        # Buttons row
        btn_row = tk.Frame(self, bg=BG)
        btn_row.pack(fill="x", padx=24, pady=(0, 8))

        self._browse_btn = self._make_btn(btn_row, "Browse Image", self._browse, ACCENT, BG)
        self._browse_btn.pack(side="left")

        self._analyze_btn = self._make_btn(btn_row, "Analyze", self._analyze, SURFACE2, ACCENT)
        self._analyze_btn.pack(side="left", padx=(12, 0))
        self._analyze_btn.config(state="disabled")

        self._clear_btn = self._make_btn(btn_row, "Clear", self._clear, SURFACE2, SUBTEXT)
        self._clear_btn.pack(side="left", padx=(12, 0))

        # Result panel
        self._result_frame = tk.Frame(self, bg=SURFACE, bd=0, highlightthickness=1,
                                      highlightbackground=BORDER)
        self._result_frame.pack(fill="x", padx=24, pady=(0, 8))

        self._result_label = tk.Label(
            self._result_frame, text="—", font=FONT_RESULT,
            bg=SURFACE, fg=SUBTEXT, pady=10,
        )
        self._result_label.pack()

        self._conf_label = tk.Label(
            self._result_frame, text="", font=FONT_CONF,
            bg=SURFACE, fg=SUBTEXT,
        )
        self._conf_label.pack(pady=(0, 4))

        self._bar_frame = tk.Frame(self._result_frame, bg=SURFACE)
        self._bar_frame.pack(fill="x", padx=20, pady=(0, 10))

        self._bar_bg = tk.Frame(self._bar_frame, bg=SURFACE2, height=6)
        self._bar_bg.pack(fill="x")
        self._bar_fill = tk.Frame(self._bar_bg, bg=SUBTEXT, height=6, width=0)
        self._bar_fill.place(x=0, y=0, relheight=1)

        # Progress & status
        self._progress = ttk.Progressbar(self, mode="indeterminate")
        self._progress.pack(fill="x", padx=24, pady=(0, 4))

        self._status = tk.Label(self, text="Loading model…", font=FONT_SMALL,
                                bg=BG, fg=SUBTEXT)
        self._status.pack(pady=(0, 12))

        # Style progress bar
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TProgressbar", troughcolor=SURFACE2, background=ACCENT,
                        thickness=4, borderwidth=0)

    def _make_btn(self, parent, text, cmd, bg, fg):
        b = tk.Button(
            parent, text=text, command=cmd,
            bg=bg, fg=fg, activebackground=ACCENT, activeforeground=BG,
            font=FONT_BODY, relief="flat", padx=18, pady=8, cursor="hand2",
        )
        b.bind("<Enter>", lambda e: b.config(bg=ACCENT, fg=BG))
        b.bind("<Leave>", lambda e: b.config(bg=bg, fg=fg))
        return b

    def _draw_placeholder(self, text="Click or drag & drop a skin lesion image"):
        self._canvas.delete("all")
        self._canvas.update_idletasks()
        w, h = self._canvas.winfo_width() or 640, self._canvas.winfo_height() or 260
        cx, cy = w // 2, h // 2
        # Dashed rect
        self._canvas.create_rectangle(20, 20, w - 20, h - 20,
                                      outline=BORDER, dash=(6, 4), width=2, tags="ph")
        # Icon
        self._canvas.create_text(cx, cy - 26, text="⊕", font=("Segoe UI", 36),
                                 fill=BORDER, tags="ph")
        self._canvas.create_text(cx, cy + 18, text=text, font=FONT_BODY,
                                 fill=SUBTEXT, tags="ph")

    def _on_hover(self, e):
        self._drop_frame.config(highlightbackground=ACCENT)

    def _on_leave(self, e):
        self._drop_frame.config(highlightbackground=BORDER)

    # ── File handling ──────────────────────────────────────────────────────────
    def _on_drop(self, e):
        path = e.data.strip().strip("{}")  # Windows path may have braces
        self._load_image(path)

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select skin lesion image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        if path:
            self._load_image(path)

    def _load_image(self, path):
        from PIL import Image, ImageTk
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            self._set_status(f"Cannot open image: {e}", error=True)
            return

        self._image_path = path
        self._canvas.delete("all")
        self._canvas.update_idletasks()
        w = self._canvas.winfo_width() or 640
        h = self._canvas.winfo_height() or 280

        img_copy = img.copy()
        img_copy.thumbnail((w - 40, h - 40), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(img_copy)
        cx, cy = w // 2, h // 2
        self._canvas.create_image(cx, cy, image=self._tk_img, anchor="center")
        self._canvas.create_text(cx, h - 14, text=Path(path).name,
                                 font=FONT_SMALL, fill=SUBTEXT)

        self._analyze_btn.config(state="normal")
        self._set_status("Image loaded. Click Analyze.")
        self._reset_result()

    def _clear(self):
        self._image_path = None
        self._canvas.delete("all")
        self._draw_placeholder()
        self._analyze_btn.config(state="disabled")
        self._reset_result()
        self._set_status("Ready.")

    # ── Analysis ───────────────────────────────────────────────────────────────
    def _analyze(self):
        if not self._image_path:
            return
        self._analyze_btn.config(state="disabled")
        self._browse_btn.config(state="disabled")
        self._set_status("Analyzing…")
        self._progress.start(12)
        self._reset_result()

        def _run():
            try:
                label, conf = predict_image(self._image_path)
                self.after(0, lambda: self._show_result(label, conf))
            except Exception as ex:
                self.after(0, lambda: self._set_status(f"Error: {ex}", error=True))
            finally:
                self.after(0, self._stop_progress)

        threading.Thread(target=_run, daemon=True).start()

    def _stop_progress(self):
        self._progress.stop()
        self._analyze_btn.config(state="normal")
        self._browse_btn.config(state="normal")

    def _show_result(self, label, conf):
        color = RED if label == "MALIGNANT" else ACCENT
        self._result_label.config(text=label, fg=color)
        self._conf_label.config(text=f"Confidence: {conf}%", fg=TEXT)

        # Animate bar
        self._bar_fill.config(bg=color)
        self._animate_bar(conf)

        notice = (
            "⚠ High confidence for malignancy — consult a dermatologist."
            if label == "MALIGNANT" and conf > 70
            else "This tool is for educational purposes only. Not a medical diagnosis."
        )
        self._set_status(notice, error=(label == "MALIGNANT"))

    def _animate_bar(self, target_pct, step=0):
        max_w = self._bar_bg.winfo_width()
        steps = 20
        if step <= steps:
            frac = (target_pct / 100) * (step / steps)
            self._bar_fill.place(x=0, y=0, relheight=1, width=int(max_w * frac))
            self.after(20, lambda: self._animate_bar(target_pct, step + 1))

    def _reset_result(self):
        self._result_label.config(text="—", fg=SUBTEXT)
        self._conf_label.config(text="", fg=SUBTEXT)
        self._bar_fill.place(x=0, y=0, relheight=1, width=0)

    def _set_status(self, msg, error=False):
        self._status.config(text=msg, fg=RED if error else SUBTEXT)


def main():
    # On Windows, suppress console if launched from EXE
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleTitleW("DermaScan")
        except Exception:
            pass

    app = DermaApp()
    app.mainloop()


if __name__ == "__main__":
    main()
