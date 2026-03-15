"""
SRT-to-Podcast — Professional GUI (Chatterbox Multilingual)
============================================================
Trilingual interface with voice cloning, GPU status, emotion control.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

try:
    import customtkinter as ctk
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "customtkinter"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    import customtkinter as ctk

from engine import (
    Language, PodcastConfig, TextSegment, VoiceConfig,
    check_ffmpeg, extract_segments, generate_podcast, get_engine, parse_srt,
)
from i18n import t

# ─────────────────────────────────────────────
# Colors
# ─────────────────────────────────────────────

C = {
    "bg": "#0F1419", "card": "#1A1F2E", "input": "#232A3B",
    "hover": "#2A3347", "accent": "#3B82F6", "accent_h": "#2563EB",
    "success": "#10B981", "error": "#EF4444",
    "t1": "#F1F5F9", "t2": "#94A3B8", "t3": "#64748B",
    "border": "#2A3347", "prog_bg": "#1E293B",
    "ar": "#8B5CF6", "ru": "#3B82F6", "en": "#10B981",
}


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.lang = "ar"
        self.is_generating = False
        self.title("SRT → Podcast (Chatterbox GPU)")
        self.geometry("920x780")
        self.minsize(820, 700)
        ctk.set_appearance_mode("dark")
        self.configure(fg_color=C["bg"])
        self._build()
        self._apply_lang()

    # ── Build ──

    def _build(self):
        # Top bar
        top = ctk.CTkFrame(self, fg_color=C["card"], height=50, corner_radius=0)
        top.pack(fill="x"); top.pack_propagate(False)

        self.lbl_title = ctk.CTkLabel(top, text="", font=ctk.CTkFont(size=18, weight="bold"), text_color=C["t1"])
        self.lbl_title.pack(side="left", padx=20)

        # GPU status badge
        self.lbl_gpu = ctk.CTkLabel(top, text="", font=ctk.CTkFont(size=11), text_color=C["t3"])
        self.lbl_gpu.pack(side="left", padx=10)
        self._update_gpu_badge()

        lang_fr = ctk.CTkFrame(top, fg_color="transparent")
        lang_fr.pack(side="right", padx=15)
        self.lang_btns = {}
        for code, label in [("ar", "عربي"), ("en", "EN"), ("ru", "RU")]:
            btn = ctk.CTkButton(lang_fr, text=label, width=55, height=30,
                font=ctk.CTkFont(size=13, weight="bold"), corner_radius=15,
                fg_color=C["input"], hover_color=C["accent"],
                command=lambda c=code: self._set_lang(c))
            btn.pack(side="left", padx=3)
            self.lang_btns[code] = btn

        self.lbl_sub = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=12), text_color=C["t3"])
        self.lbl_sub.pack(pady=(8, 2))

        # Scrollable body
        body = ctk.CTkScrollableFrame(self, fg_color=C["bg"])
        body.pack(fill="both", expand=True, padx=20, pady=(5, 10))

        # ── File Card ──
        fc = self._card(body)
        self.lbl_file = ctk.CTkLabel(fc, text="", font=ctk.CTkFont(size=14, weight="bold"), text_color=C["t1"])
        self.lbl_file.pack(anchor="w", padx=15, pady=(12, 5))

        fr = ctk.CTkFrame(fc, fg_color="transparent")
        fr.pack(fill="x", padx=15, pady=(0, 5))
        self.var_file = tk.StringVar()
        self.ent_file = ctk.CTkEntry(fr, textvariable=self.var_file, font=ctk.CTkFont(size=13), height=38,
            fg_color=C["input"], border_color=C["border"], text_color=C["t1"])
        self.ent_file.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.btn_browse = ctk.CTkButton(fr, text="", width=100, height=38,
            fg_color=C["accent"], hover_color=C["accent_h"], command=self._browse_srt)
        self.btn_browse.pack(side="right")

        self.frm_info = ctk.CTkFrame(fc, fg_color=C["input"], corner_radius=8)
        self.lbl_info = ctk.CTkLabel(self.frm_info, text="", font=ctk.CTkFont(size=12),
            text_color=C["t2"], justify="left")
        self.lbl_info.pack(padx=12, pady=8, anchor="w")

        # ── Output Card ──
        oc = self._card(body)
        self.lbl_out = ctk.CTkLabel(oc, text="", font=ctk.CTkFont(size=14, weight="bold"), text_color=C["t1"])
        self.lbl_out.pack(anchor="w", padx=15, pady=(12, 5))
        or_ = ctk.CTkFrame(oc, fg_color="transparent")
        or_.pack(fill="x", padx=15, pady=(0, 12))
        self.var_out = tk.StringVar(value=str(Path.home() / "podcast.mp3"))
        ctk.CTkEntry(or_, textvariable=self.var_out, font=ctk.CTkFont(size=13), height=38,
            fg_color=C["input"], border_color=C["border"], text_color=C["t1"]).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ctk.CTkButton(or_, text="💾", width=45, height=38, fg_color=C["input"],
            hover_color=C["hover"], border_color=C["border"], border_width=1,
            command=self._browse_out).pack(side="right")

        # ── Voice Card ──
        vc = self._card(body)
        self.lbl_voice = ctk.CTkLabel(vc, text="", font=ctk.CTkFont(size=14, weight="bold"), text_color=C["t1"])
        self.lbl_voice.pack(anchor="w", padx=15, pady=(12, 5))

        # Voice file selectors
        vg = ctk.CTkFrame(vc, fg_color="transparent")
        vg.pack(fill="x", padx=15, pady=(0, 8))
        vg.columnconfigure((0, 1), weight=1)

        self.var_voice = tk.StringVar()
        self.var_voice_ar = tk.StringVar()
        self.var_voice_ru = tk.StringVar()
        self.var_voice_en = tk.StringVar()

        self.lbl_vg = self._voice_row(vg, "", self.var_voice, 0, 0, span=2)
        self.lbl_var = self._voice_row(vg, "", self.var_voice_ar, 1, 0, color=C["ar"])
        self.lbl_vru = self._voice_row(vg, "", self.var_voice_ru, 1, 1, color=C["ru"])
        self.lbl_ven = self._voice_row(vg, "", self.var_voice_en, 2, 0, color=C["en"])

        # Sliders
        sl_fr = ctk.CTkFrame(vg, fg_color="transparent")
        sl_fr.grid(row=2, column=1, sticky="ew", padx=4, pady=3)

        self.lbl_exag = ctk.CTkLabel(sl_fr, text="", font=ctk.CTkFont(size=11), text_color=C["t2"])
        self.lbl_exag.pack(anchor="w")
        self.var_exag = tk.DoubleVar(value=0.5)
        ctk.CTkSlider(sl_fr, from_=0, to=1, variable=self.var_exag, height=16,
            fg_color=C["prog_bg"], progress_color=C["accent"], button_color=C["t1"]).pack(fill="x")
        self.lbl_exag_v = ctk.CTkLabel(sl_fr, text="0.5", font=ctk.CTkFont(size=10), text_color=C["t3"])
        self.lbl_exag_v.pack(anchor="e")
        self.var_exag.trace_add("write", lambda *_: self.lbl_exag_v.configure(text=f"{self.var_exag.get():.2f}"))

        # cfg_weight
        cfg_fr = ctk.CTkFrame(vc, fg_color="transparent")
        cfg_fr.pack(fill="x", padx=15, pady=(0, 12))
        self.lbl_cfg = ctk.CTkLabel(cfg_fr, text="", font=ctk.CTkFont(size=11), text_color=C["t2"])
        self.lbl_cfg.pack(anchor="w")
        self.var_cfg = tk.DoubleVar(value=0.5)
        ctk.CTkSlider(cfg_fr, from_=0, to=1, variable=self.var_cfg, height=16,
            fg_color=C["prog_bg"], progress_color=C["ar"], button_color=C["t1"]).pack(fill="x")

        # ── Timing Card ──
        tc = self._card(body)
        self.lbl_timing = ctk.CTkLabel(tc, text="", font=ctk.CTkFont(size=14, weight="bold"), text_color=C["t1"])
        self.lbl_timing.pack(anchor="w", padx=15, pady=(12, 8))
        tg = ctk.CTkFrame(tc, fg_color="transparent")
        tg.pack(fill="x", padx=15, pady=(0, 12))
        tg.columnconfigure((0, 1, 2), weight=1)
        self.var_pl = tk.IntVar(value=600)
        self.var_pb = tk.IntVar(value=1200)
        self.var_pr = tk.IntVar(value=400)
        self.lbl_pl = self._spin(tg, "", self.var_pl, 0, 0)
        self.lbl_pb = self._spin(tg, "", self.var_pb, 0, 1)
        self.lbl_pr = self._spin(tg, "", self.var_pr, 0, 2)

        # ── Generate + Progress ──
        self.btn_gen = ctk.CTkButton(body, text="", height=48,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color=C["accent"], hover_color=C["accent_h"],
            corner_radius=12, command=self._start)
        self.btn_gen.pack(fill="x", pady=(10, 5))

        pf = ctk.CTkFrame(body, fg_color=C["card"], corner_radius=10)
        pf.pack(fill="x", pady=(5, 5))
        self.prog = ctk.CTkProgressBar(pf, height=6, fg_color=C["prog_bg"],
            progress_color=C["accent"], corner_radius=3)
        self.prog.pack(fill="x", padx=15, pady=(12, 4)); self.prog.set(0)
        self.lbl_status = ctk.CTkLabel(pf, text="", font=ctk.CTkFont(size=12), text_color=C["t2"])
        self.lbl_status.pack(padx=15, pady=(0, 4), anchor="w")
        self.lbl_detail = ctk.CTkLabel(pf, text="", font=ctk.CTkFont(size=11), text_color=C["t3"])
        self.lbl_detail.pack(padx=15, pady=(0, 10), anchor="w")

        self.frm_result = ctk.CTkFrame(body, fg_color="transparent")
        self.btn_open = ctk.CTkButton(self.frm_result, text="", height=36, fg_color=C["success"],
            hover_color="#059669", corner_radius=8, command=self._open_file)
        self.btn_open.pack(side="left", expand=True, fill="x", padx=(0, 5))
        self.btn_folder = ctk.CTkButton(self.frm_result, text="", height=36, fg_color=C["input"],
            hover_color=C["hover"], border_color=C["border"], border_width=1,
            corner_radius=8, command=self._open_folder)
        self.btn_folder.pack(side="left", expand=True, fill="x", padx=(5, 0))

    # ── Helpers ──

    def _card(self, parent) -> ctk.CTkFrame:
        c = ctk.CTkFrame(parent, fg_color=C["card"], corner_radius=12,
            border_width=1, border_color=C["border"])
        c.pack(fill="x", pady=(0, 8)); return c

    def _spin(self, parent, label, var, row, col):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.grid(row=row, column=col, sticky="ew", padx=4)
        lbl = ctk.CTkLabel(f, text=label, font=ctk.CTkFont(size=11), text_color=C["t2"])
        lbl.pack(anchor="w")
        ctk.CTkEntry(f, textvariable=var, font=ctk.CTkFont(size=12), height=32,
            fg_color=C["input"], border_color=C["border"], text_color=C["t1"],
            justify="center").pack(fill="x")
        return lbl

    def _voice_row(self, parent, label, var, row, col, span=1, color=None):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.grid(row=row, column=col, columnspan=span, sticky="ew", padx=4, pady=3)
        lbl = ctk.CTkLabel(f, text=label, font=ctk.CTkFont(size=11),
            text_color=color or C["t2"])
        lbl.pack(anchor="w")
        row_f = ctk.CTkFrame(f, fg_color="transparent")
        row_f.pack(fill="x")
        ctk.CTkEntry(row_f, textvariable=var, font=ctk.CTkFont(size=11), height=30,
            fg_color=C["input"], border_color=C["border"], text_color=C["t1"],
            placeholder_text=".wav").pack(side="left", fill="x", expand=True, padx=(0, 4))
        ctk.CTkButton(row_f, text="📂", width=35, height=30, fg_color=C["input"],
            hover_color=C["hover"], command=lambda: self._pick_wav(var)).pack(side="right")
        return lbl

    def _pick_wav(self, var):
        p = filedialog.askopenfilename(filetypes=[("WAV", "*.wav"), ("All", "*.*")])
        if p: var.set(p)

    def _update_gpu_badge(self):
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
                self.lbl_gpu.configure(text=f"🟢 {name} ({mem:.0f}GB)", text_color=C["success"])
            else:
                self.lbl_gpu.configure(text="🟡 CPU mode", text_color=C["t3"])
        except Exception:
            self.lbl_gpu.configure(text="⚪ torch not loaded", text_color=C["t3"])

    # ── Language ──

    def _set_lang(self, lang):
        self.lang = lang; self._apply_lang()

    def _apply_lang(self):
        T = lambda k: t(k, self.lang)
        for c, b in self.lang_btns.items():
            b.configure(fg_color=C["accent"] if c == self.lang else C["input"])
        self.lbl_title.configure(text=T("app_title"))
        self.lbl_sub.configure(text=T("app_subtitle"))
        self.lbl_file.configure(text=T("select_file"))
        self.btn_browse.configure(text=T("browse"))
        self.lbl_out.configure(text=T("output_path"))
        self.lbl_voice.configure(text=T("voice_section"))
        self.lbl_vg.configure(text=T("voice_global"))
        self.lbl_var.configure(text=T("voice_ar"))
        self.lbl_vru.configure(text=T("voice_ru"))
        self.lbl_ven.configure(text=T("voice_en"))
        self.lbl_exag.configure(text=T("exaggeration"))
        self.lbl_cfg.configure(text=T("cfg_weight"))
        self.lbl_timing.configure(text=T("timing_section"))
        self.lbl_pl.configure(text=T("pause_lines"))
        self.lbl_pb.configure(text=T("pause_blocks"))
        self.lbl_pr.configure(text=T("pause_reps"))
        self.btn_open.configure(text=T("open_file"))
        self.btn_folder.configure(text=T("open_folder"))
        if not self.is_generating:
            self.btn_gen.configure(text=T("generate"))
            self.lbl_status.configure(text=T("ready"), text_color=C["t2"])

    # ── File ──

    def _browse_srt(self):
        p = filedialog.askopenfilename(filetypes=[("SRT", "*.srt"), ("All", "*.*")])
        if p:
            self.var_file.set(p)
            self.var_out.set(str(Path(p).with_suffix(".mp3")))
            self._preview(p)

    def _preview(self, path):
        try:
            entries = parse_srt(Path(path).read_text(encoding="utf-8"))
            segs = extract_segments(entries)
            T = lambda k: t(k, self.lang)
            ar = sum(1 for s in segs if s.language == Language.ARABIC)
            ru = sum(1 for s in segs if s.language == Language.RUSSIAN)
            en = sum(1 for s in segs if s.language == Language.ENGLISH)
            self.lbl_info.configure(
                text=f"{T('total_segments')}: {len(segs)}  |  🟣 {T('ar_count')}: {ar}  🔵 {T('ru_count')}: {ru}  🟢 {T('en_count')}: {en}")
            self.frm_info.pack(fill="x", padx=15, pady=(0, 12))
        except Exception:
            self.frm_info.pack_forget()

    def _browse_out(self):
        p = filedialog.asksaveasfilename(defaultextension=".mp3",
            filetypes=[("MP3", "*.mp3"), ("WAV", "*.wav"), ("OGG", "*.ogg"), ("FLAC", "*.flac")])
        if p: self.var_out.set(p)

    # ── Generate ──

    def _start(self):
        if self.is_generating: return
        T = lambda k: t(k, self.lang)
        srt = self.var_file.get().strip()
        if not srt or not Path(srt).exists():
            messagebox.showerror(T("error"), T("no_file")); return
        if not check_ffmpeg():
            messagebox.showerror(T("error"), T("ffmpeg_missing")); return

        self.is_generating = True
        self.btn_gen.configure(text=T("generating"), state="disabled")
        self.prog.set(0); self.frm_result.pack_forget()

        vc = VoiceConfig(
            voice_prompt_path=self.var_voice.get() or None,
            voice_prompt_ar=self.var_voice_ar.get() or None,
            voice_prompt_ru=self.var_voice_ru.get() or None,
            voice_prompt_en=self.var_voice_en.get() or None,
            exaggeration=self.var_exag.get(),
            cfg_weight=self.var_cfg.get(),
        )
        pc = PodcastConfig(
            pause_between_lines_ms=self.var_pl.get(),
            pause_between_blocks_ms=self.var_pb.get(),
            pause_between_repetitions_ms=self.var_pr.get(),
        )
        threading.Thread(target=self._run, args=(srt, self.var_out.get(), vc, pc), daemon=True).start()

    def _run(self, srt, out, vc, pc):
        T = lambda k: t(k, self.lang)
        def on_prog(done, total, seg):
            e = {"ar": "🟣", "ru": "🔵", "en": "🟢"}.get(seg.language.value, "⚪")
            self.after(0, lambda: self.prog.set(done / total))
            self.after(0, lambda: self.lbl_detail.configure(text=f"[{done}/{total}] {e} {seg.text[:50]}"))
        def on_st(msg):
            self.after(0, lambda: self.lbl_status.configure(text=msg, text_color=C["t2"]))
        try:
            result = generate_podcast(srt, out, vc, pc, on_progress=on_prog, on_status=on_st)
            self.after(0, lambda: self._done(result))
        except Exception as e:
            self.after(0, lambda: self._err(str(e)))

    def _done(self, result):
        T = lambda k: t(k, self.lang)
        self.is_generating = False
        self.btn_gen.configure(text=T("generate"), state="normal")
        self.prog.set(1.0)
        self.lbl_status.configure(
            text=f"{T('success')}  —  {result.duration_seconds:.1f} {T('seconds')}",
            text_color=C["success"])
        self.lbl_detail.configure(text=f"{result.total_segments} segments")
        self.frm_result.pack(fill="x", pady=(5, 10))
        self._update_gpu_badge()

    def _err(self, msg):
        T = lambda k: t(k, self.lang)
        self.is_generating = False
        self.btn_gen.configure(text=T("generate"), state="normal")
        self.lbl_status.configure(text=f"{T('error')}: {msg[:80]}", text_color=C["error"])

    def _open_file(self):
        p = self.var_out.get()
        if Path(p).exists():
            if platform.system() == "Windows": os.startfile(p)
            elif platform.system() == "Darwin": subprocess.run(["open", p])
            else: subprocess.run(["xdg-open", p])

    def _open_folder(self):
        p = Path(self.var_out.get()).parent
        if p.exists():
            if platform.system() == "Windows": subprocess.run(["explorer", str(p)])
            elif platform.system() == "Darwin": subprocess.run(["open", str(p)])
            else: subprocess.run(["xdg-open", str(p)])


if __name__ == "__main__":
    App().mainloop()
