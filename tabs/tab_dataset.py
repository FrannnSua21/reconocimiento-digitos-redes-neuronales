"""Pestaña 4 · Dataset: muestras, distribución de clases y dígito promedio."""
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.figure import Figure

from config import ACENTO, TXT, VERDE
from ui_utils import aviso_ax, combo, estilo_ax, etiqueta, incrustar, leyenda


class TabDataset(ttk.Frame):
    VISTAS = ["Muestras aleatorias", "Distribución de clases", "Dígito promedio"]

    def __init__(self, padre, app):
        super().__init__(padre)
        self.app = app
        self.vista = tk.StringVar(value=self.VISTAS[0])
        barra = ttk.Frame(self, style="Panel.TFrame", padding=10)
        barra.pack(fill="x", padx=10, pady=(10, 6))
        etiqueta(barra, "Vista").pack(side="left")
        cb = combo(barra, self.vista, self.VISTAS, 24, state="readonly")
        cb.pack(side="left", padx=8)
        cb.bind("<<ComboboxSelected>>", lambda e: self.dibujar())
        ttk.Button(barra, text="Otras muestras", command=self.dibujar).pack(side="left")
        self.lbl = etiqueta(barra, "Cargando MNIST…", "Muted.TLabel")
        self.lbl.pack(side="left", padx=14)
        marco = ttk.Frame(self, style="Panel.TFrame", padding=6)
        marco.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.fig = Figure(figsize=(9, 6), dpi=100, constrained_layout=True)
        self.lienzo = incrustar(self.fig, marco)
        aviso_ax(self.fig.add_subplot(111), "Cargando MNIST…")

    def dibujar(self):
        d = self.app.datos
        if d is None:
            return
        (x_tr, y_tr), (x_te, y_te) = d
        self.lbl.config(text=f"{len(x_tr):,} imágenes de entrenamiento · {len(x_te):,} de "
                             f"prueba · 28×28 · píxel medio {x_tr.mean():.3f} "
                             f"(desv. {x_tr.std():.3f})")
        self.fig.clear()
        vista = self.vista.get()
        if vista == "Muestras aleatorias":
            idx = np.random.choice(len(x_tr), 40, replace=False)
            for ax, i in zip(self.fig.subplots(4, 10).flat, idx):
                ax.imshow(x_tr[i, ..., 0], cmap="gray")
                ax.set_title(str(y_tr[i]), color=TXT, fontsize=9)
                ax.axis("off")
        elif vista == "Distribución de clases":
            ax = self.fig.add_subplot(111)
            estilo_ax(ax, "Imágenes por dígito")
            n_tr, n_te = np.bincount(y_tr, minlength=10), np.bincount(y_te, minlength=10)
            ax.bar(np.arange(10) - 0.2, n_tr, 0.4, color=ACENTO, label="Entrenamiento")
            ax.bar(np.arange(10) + 0.2, n_te * 6, 0.4, color=VERDE,
                   label="Prueba (×6 para comparar)")
            ax.set_xticks(range(10))
            leyenda(ax)
        else:
            axs = self.fig.subplots(2, 5)
            for k, ax in enumerate(axs.flat):
                ax.imshow(x_tr[y_tr == k].mean(axis=0)[..., 0], cmap="magma")
                ax.set_title(f"Promedio del {k}", color=TXT, fontsize=10)
                ax.axis("off")
        self.lienzo.draw_idle()
