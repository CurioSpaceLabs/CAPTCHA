"""Retake guidance window shown after failed face recognition."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class RetakePrompt:
    """Display guidance and wait for the user to request another photo."""

    def __init__(self) -> None:
        self._should_retake = False

    def show(self, confidence_text: str, best_match: str | None) -> bool:
        """Return True when the user clicks Retake Photo."""
        root = tk.Tk()
        root.title("CAPTCHA verification needed")
        root.resizable(False, False)
        root.protocol("WM_DELETE_WINDOW", root.destroy)

        self._center_window(root, width=460, height=330)
        self._build_content(root, confidence_text, best_match)

        root.mainloop()
        return self._should_retake

    def _build_content(
        self,
        root: tk.Tk,
        confidence_text: str,
        best_match: str | None,
    ) -> None:
        """Create the denial guidance UI."""
        frame = ttk.Frame(root, padding=24)
        frame.grid(row=0, column=0, sticky="nsew")

        title = ttk.Label(
            frame,
            text="Access denied",
            font=("Segoe UI", 18, "bold"),
        )
        title.grid(row=0, column=0, sticky="w")

        message = ttk.Label(
            frame,
            text=(
                "We could not confidently verify your face.\n\n"
                "Before retaking the photo:\n"
                "- Remove sunglasses, masks, hats, or anything covering your face.\n"
                "- Sit upright and face the camera directly.\n"
                "- Move into better lighting.\n"
                "- Keep your face centered and still."
            ),
            font=("Segoe UI", 10),
            justify="left",
            wraplength=400,
        )
        message.grid(row=1, column=0, sticky="w", pady=(16, 12))

        details = ttk.Label(
            frame,
            text=f"Confidence: {confidence_text}\nBest match: {best_match or 'None'}",
            font=("Segoe UI", 10),
            justify="left",
        )
        details.grid(row=2, column=0, sticky="w", pady=(0, 18))

        retake_button = ttk.Button(
            frame,
            text="Retake Photo",
            command=lambda: self._request_retake(root),
        )
        retake_button.grid(row=3, column=0, sticky="e")
        retake_button.focus_set()

        root.bind("<Return>", lambda _event: self._request_retake(root))
        root.bind("<Escape>", lambda _event: root.destroy())

    def _request_retake(self, root: tk.Tk) -> None:
        """Close the window and signal that another capture should run."""
        self._should_retake = True
        root.destroy()

    @staticmethod
    def _center_window(root: tk.Tk, width: int, height: int) -> None:
        """Center the window on the current screen."""
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        root.geometry(f"{width}x{height}+{x}+{y}")


def ask_for_retake(confidence_text: str, best_match: str | None) -> bool:
    """Show the failed-verification prompt and return whether to retake."""
    return RetakePrompt().show(confidence_text, best_match)
