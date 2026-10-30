"""SignSpeak — train your own signs on webcam and translate them live."""

from __future__ import annotations

import threading
import time
from collections import deque

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image

from camera import HandCamera
from trainer import SignTrainer

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BG = "#0b1020"
CARD = "#141a2e"
ACCENT = "#6ee7d0"
ACCENT_DIM = "#2dd4bf"
TEXT = "#e8eefc"
MUTED = "#8b97b8"
DANGER = "#fb7185"
VIDEO_W, VIDEO_H = 720, 540


class SignSpeakApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("SignSpeak")
        self.geometry("1280x760")
        self.minsize(1100, 680)
        self.configure(fg_color=BG)

        self.trainer = SignTrainer()
        self.latest_features: np.ndarray | None = None
        self.capturing = False
        self.capture_buffer: list[np.ndarray] = []
        self.photo = None
        self.last_committed: str | None = None
        self.last_commit_at: float = 0.0
        self.hold_started: float | None = None
        self.sentence_parts: list[str] = []
        self.pred_smooth: deque[tuple[str, float]] = deque(maxlen=6)
        self._tts_lock = threading.Lock()

        self._build()
        self.camera = HandCamera(self._on_frame)
        self.camera.start()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.after(80, self._tick)

    def _build(self) -> None:
        root = ctk.CTkFrame(self, fg_color=BG)
        root.pack(fill="both", expand=True, padx=18, pady=16)
        root.grid_columnconfigure(0, weight=3)
        root.grid_columnconfigure(1, weight=2)
        root.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(root, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 14))
        ctk.CTkLabel(
            header,
            text="SignSpeak",
            font=ctk.CTkFont(family="Segoe UI", size=28, weight="bold"),
            text_color=ACCENT,
        ).pack(side="left")
        ctk.CTkLabel(
            header,
            text="  train your signs  ·  translate them live",
            font=ctk.CTkFont(family="Segoe UI", size=14),
            text_color=MUTED,
        ).pack(side="left", pady=(8, 0))

        video_card = ctk.CTkFrame(root, fg_color=CARD, corner_radius=18)
        video_card.grid(row=1, column=0, sticky="nsew", padx=(0, 14))
        self.video = ctk.CTkLabel(video_card, text="Starting camera…", text_color=MUTED)
        self.video.pack(fill="both", expand=True, padx=12, pady=12)

        overlay = ctk.CTkFrame(video_card, fg_color="#0b1020", corner_radius=12)
        overlay.pack(fill="x", padx=16, pady=(0, 16))
        self.live_sign = ctk.CTkLabel(
            overlay,
            text="show a trained sign",
            font=ctk.CTkFont(family="Segoe UI", size=26, weight="bold"),
            text_color=TEXT,
        )
        self.live_sign.pack(anchor="w", padx=16, pady=(12, 0))
        self.conf_bar = ctk.CTkProgressBar(overlay, height=8, progress_color=ACCENT_DIM)
        self.conf_bar.pack(fill="x", padx=16, pady=(10, 6))
        self.conf_bar.set(0)
        self.conf_label = ctk.CTkLabel(overlay, text="confidence  0%", text_color=MUTED)
        self.conf_label.pack(anchor="w", padx=16, pady=(0, 12))

        side = ctk.CTkFrame(root, fg_color="transparent")
        side.grid(row=1, column=1, sticky="nsew")

        sentence_card = ctk.CTkFrame(side, fg_color=CARD, corner_radius=18)
        sentence_card.pack(fill="x")
        ctk.CTkLabel(sentence_card, text="TRANSLATION", text_color=MUTED, font=ctk.CTkFont(size=11, weight="bold")).pack(
            anchor="w", padx=16, pady=(12, 0)
        )
        self.sentence = ctk.CTkLabel(
            sentence_card,
            text="Your sentence will appear here.",
            wraplength=420,
            justify="left",
            font=ctk.CTkFont(family="Segoe UI", size=18),
            text_color=TEXT,
        )
        self.sentence.pack(anchor="w", padx=16, pady=10)

        actions = ctk.CTkFrame(sentence_card, fg_color="transparent")
        actions.pack(fill="x", padx=12, pady=(0, 12))
        ctk.CTkButton(actions, text="Space", width=80, command=self._space, fg_color="#243049", hover_color="#31405d").pack(
            side="left", padx=4
        )
        ctk.CTkButton(actions, text="Delete", width=80, command=self._backspace, fg_color="#243049", hover_color="#31405d").pack(
            side="left", padx=4
        )
        ctk.CTkButton(actions, text="Clear", width=80, command=self._clear, fg_color="#243049", hover_color="#31405d").pack(
            side="left", padx=4
        )
        ctk.CTkButton(actions, text="Speak", width=80, command=self._speak, fg_color=ACCENT_DIM, text_color="#042f2e").pack(
            side="left", padx=4
        )

        self.tabs = ctk.CTkTabview(side, fg_color=CARD, corner_radius=18, segmented_button_selected_color=ACCENT_DIM)
        self.tabs.pack(fill="both", expand=True, pady=(14, 0))
        self.tabs.add("Train")
        self.tabs.add("Library")
        self.tabs.add("How to")

        train = self.tabs.tab("Train")
        ctk.CTkLabel(
            train,
            text="Hold a sign in frame, name it, then capture samples. Train once you have two or more signs.",
            wraplength=400,
            justify="left",
            text_color=MUTED,
        ).pack(anchor="w", padx=8, pady=(8, 10))

        self.sign_name = ctk.CTkEntry(train, placeholder_text="sign meaning  e.g. hello", height=40)
        self.sign_name.pack(fill="x", padx=8)

        self.capture_btn = ctk.CTkButton(
            train,
            text="Hold to capture  (2s)",
            height=44,
            fg_color=ACCENT_DIM,
            hover_color=ACCENT,
            text_color="#042f2e",
            font=ctk.CTkFont(weight="bold"),
        )
        self.capture_btn.pack(fill="x", padx=8, pady=(12, 6))
        self.capture_btn.bind("<ButtonPress-1>", self._start_capture)
        self.capture_btn.bind("<ButtonRelease-1>", self._stop_capture)

        self.capture_status = ctk.CTkLabel(train, text="No samples this session.", text_color=MUTED)
        self.capture_status.pack(anchor="w", padx=8)

        self.train_btn = ctk.CTkButton(train, text="Train model", height=42, command=self._train_model)
        self.train_btn.pack(fill="x", padx=8, pady=(16, 6))
        self.train_status = ctk.CTkLabel(train, text=self._model_blurb(), text_color=MUTED, wraplength=400, justify="left")
        self.train_status.pack(anchor="w", padx=8)

        library = self.tabs.tab("Library")
        self.lib_box = ctk.CTkScrollableFrame(library, fg_color="transparent")
        self.lib_box.pack(fill="both", expand=True, padx=4, pady=8)
        self._refresh_library()

        howto = self.tabs.tab("How to")
        tips = (
            "1. Open Train and type what the sign means.\n"
            "2. Hold the pose clearly in the camera and press Hold to capture.\n"
            "3. Repeat for more signs — letters, words, or gestures you invent.\n"
            "4. Hit Train model, then sign live. Hold a pose ~0.7s to add it.\n"
            "5. Use Space / Delete if you want to build a sentence.\n\n"
            "Tips: sit in good light, keep one pose per sign, and recapture a sign if it keeps getting confused."
        )
        ctk.CTkLabel(howto, text=tips, justify="left", wraplength=400, text_color=TEXT).pack(anchor="w", padx=8, pady=12)

    def _model_blurb(self) -> str:
        n = len(self.trainer.labels())
        samples = len(self.trainer.y)
        ready = "Model ready." if self.trainer.model is not None else "No model yet."
        return f"{ready}  {n} signs, {samples} samples."

    def _on_frame(self, frame_bgr: np.ndarray, features: np.ndarray | None) -> None:
        self.latest_features = features
        if self.capturing and features is not None:
            self.capture_buffer.append(features.copy())

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb).resize((VIDEO_W, VIDEO_H), Image.Resampling.BILINEAR)
        self.photo = ctk.CTkImage(light_image=image, dark_image=image, size=(VIDEO_W, VIDEO_H))
        self.after(0, lambda img=self.photo: self.video.configure(image=img, text=""))

    def _tick(self) -> None:
        if not self.capturing:
            label, conf = self.trainer.predict(self.latest_features)
            if label:
                self.pred_smooth.append((label, conf))
            else:
                self.pred_smooth.clear()
                self.hold_started = None
                self.live_sign.configure(text="show a trained sign")
                self.conf_bar.set(0)
                self.conf_label.configure(text="confidence  0%")

            if self.pred_smooth:
                best = max(self.pred_smooth, key=lambda item: item[1])
                name, conf = best
                self.live_sign.configure(text=name)
                self.conf_bar.set(conf)
                self.conf_label.configure(text=f"confidence  {conf * 100:.0f}%")
                if conf >= 0.62:
                    if self.hold_started is None:
                        self.hold_started = time.time()
                    elif time.time() - self.hold_started >= 0.75:
                        self._commit(name)
                        self.hold_started = None
                else:
                    self.hold_started = None

        if self.capturing:
            n = len(self.capture_buffer)
            self.capture_status.configure(text=f"Capturing… {n} frames  (keep the pose still)")

        self.after(80, self._tick)

    def _commit(self, name: str) -> None:
        now = time.time()
        if name == self.last_committed and now - self.last_commit_at < 1.4:
            return
        self.last_committed = name
        self.last_commit_at = now
        self.sentence_parts.append(name)
        self._render_sentence()
        self.after(1400, self._unlock_commit)

    def _unlock_commit(self) -> None:
        self.last_committed = None

    def _render_sentence(self) -> None:
        text = " ".join(self.sentence_parts).strip() or "Your sentence will appear here."
        self.sentence.configure(text=text)

    def _space(self) -> None:
        if self.sentence_parts and self.sentence_parts[-1] != "":
            self.sentence_parts.append("")
        self._render_sentence()

    def _backspace(self) -> None:
        if self.sentence_parts:
            self.sentence_parts.pop()
        self._render_sentence()

    def _clear(self) -> None:
        self.sentence_parts.clear()
        self._render_sentence()

    def _speak(self) -> None:
        text = " ".join(part for part in self.sentence_parts if part).strip()
        if not text:
            return

        def run() -> None:
            with self._tts_lock:
                try:
                    import pyttsx3

                    engine = pyttsx3.init()
                    engine.say(text)
                    engine.runAndWait()
                except Exception:
                    pass

        threading.Thread(target=run, daemon=True).start()

    def _start_capture(self, _event=None) -> None:
        name = self.sign_name.get().strip()
        if not name:
            self.capture_status.configure(text="Type a sign meaning first.")
            return
        self.capturing = True
        self.capture_buffer = []
        self.capture_status.configure(text="Capturing… hold the pose.")

    def _stop_capture(self, _event=None) -> None:
        if not self.capturing:
            return
        self.capturing = False
        name = self.sign_name.get().strip()
        added = self.trainer.add_samples(name, self.capture_buffer)
        self.capture_buffer = []
        if added == 0:
            self.capture_status.configure(text="No hand detected. Try again in better light.")
        else:
            self.capture_status.configure(text=f"Saved {added} samples for “{name.lower()}”.")
        self.train_status.configure(text=self._model_blurb())
        self._refresh_library()

    def _train_model(self) -> None:
        try:
            stats = self.trainer.train()
        except ValueError as exc:
            self.train_status.configure(text=str(exc))
            return
        self.train_status.configure(
            text=f"Trained on {stats['signs']} signs / {stats['samples']} samples  ·  fit {stats['accuracy'] * 100:.0f}%"
        )

    def _refresh_library(self) -> None:
        for child in self.lib_box.winfo_children():
            child.destroy()
        counts = self.trainer.counts()
        if not counts:
            ctk.CTkLabel(self.lib_box, text="No signs yet. Capture a few in Train.", text_color=MUTED).pack(anchor="w")
            return
        for label, count in counts.items():
            row = ctk.CTkFrame(self.lib_box, fg_color="#1b2338", corner_radius=10)
            row.pack(fill="x", pady=4)
            ctk.CTkLabel(row, text=f"{label}", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=12, pady=8)
            ctk.CTkLabel(row, text=f"{count} samples", text_color=MUTED).pack(side="left")
            ctk.CTkButton(
                row,
                text="Remove",
                width=70,
                fg_color="#3b2130",
                hover_color=DANGER,
                command=lambda name=label: self._delete(name),
            ).pack(side="right", padx=8, pady=6)

    def _delete(self, name: str) -> None:
        self.trainer.delete_label(name)
        self.train_status.configure(text=self._model_blurb())
        self._refresh_library()

    def _close(self) -> None:
        try:
            self.camera.stop()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    SignSpeakApp().mainloop()
