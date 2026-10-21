"""Pestaña 3 · Evaluar: métricas por clase, matriz de confusión y errores."""
from tkinter import ttk

import numpy as np
from matplotlib.figure import Figure

from config import MUTED, TXT
from nucleo import matriz_confusion, metricas_por_clase
from ui_utils import aviso_ax, etiqueta, incrustar


class TabEvaluacion(ttk.Frame):
    def __init__(self, padre, app):
        super().__init__(padre)
        self.app = app
        self.construir()

    def construir(self):
        barra = ttk.Frame(self, style="Panel.TFrame", padding=10)
        barra.pack(fill="x", padx=10, pady=(10, 6))
        ttk.Button(barra, text="Evaluar modelo activo", style="Acento.TButton",
                   command=self.app.evaluar_modelo).pack(side="left")
        self.lbl = etiqueta(barra, "Evalúa el modelo activo con las 10 000 imágenes de prueba.",
                            "Muted.TLabel")
        self.lbl.pack(side="left", padx=14)

        cuerpo = ttk.Frame(self)
        cuerpo.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        cuerpo.columnconfigure(0, weight=1, uniform="c")
        cuerpo.columnconfigure(1, weight=1, uniform="c")
        cuerpo.rowconfigure(0, weight=1)

        izq = ttk.Frame(cuerpo)
        izq.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        izq.rowconfigure(1, weight=1)
        izq.columnconfigure(0, weight=1)
        cols = ("digito", "precision", "recall", "f1", "soporte")
        self.tabla = ttk.Treeview(izq, columns=cols, show="headings", height=12)
        for c, t in zip(cols, ["Dígito", "Precisión", "Recall", "F1", "Soporte"]):
            self.tabla.heading(c, text=t)
            self.tabla.column(c, width=90, anchor="center")
        self.tabla.grid(row=0, column=0, sticky="ew")
        marco = ttk.Frame(izq, style="Panel.TFrame", padding=6)
        marco.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        self.fig_err = Figure(figsize=(5, 3), dpi=100, constrained_layout=True)
        self.lienzo_err = incrustar(self.fig_err, marco)

        der = ttk.Frame(cuerpo, style="Panel.TFrame", padding=6)
        der.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self.fig_cm = Figure(figsize=(5, 5), dpi=100, constrained_layout=True)
        self.lienzo_cm = incrustar(self.fig_cm, der)
        aviso_ax(self.fig_cm.add_subplot(111), "Matriz de confusión")
        aviso_ax(self.fig_err.add_subplot(111), "Errores más seguros del modelo")

    def on_eval(self, d):
        probs, y = d["probs"], d["y"]
        x_te = self.app.datos[1][0]
        pred = probs.argmax(axis=1)
        cm = matriz_confusion(y, pred)
        prec, rec, f1, sop = metricas_por_clase(cm)
        exactitud = float((pred == y).mean())
        perdida = float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1)).mean())
        self.lbl.config(text=f"Exactitud {exactitud:.2%} · pérdida {perdida:.4f} · "
                             f"{int((pred != y).sum())} errores de {len(y)}")

        self.tabla.delete(*self.tabla.get_children())
        for i in range(10):
            self.tabla.insert("", "end", values=(i, f"{prec[i]:.3f}", f"{rec[i]:.3f}",
                                                 f"{f1[i]:.3f}", int(sop[i])))
        self.tabla.insert("", "end", values=("Macro", f"{prec.mean():.3f}", f"{rec.mean():.3f}",
                                             f"{f1.mean():.3f}", int(sop.sum())))

        self.fig_cm.clear()
        ax = self.fig_cm.add_subplot(111)
        ax.imshow(cm, cmap="magma")
        ax.set_title("Matriz de confusión", color=TXT, fontsize=10, loc="left")
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xlabel("Predicho", color=MUTED)
        ax.set_ylabel("Real", color=MUTED)
        ax.tick_params(colors=MUTED)
        for i in range(10):
            for j in range(10):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] < cm.max() / 2 else "black")
        self.lienzo_cm.draw_idle()

        self.fig_err.clear()
        idx = np.where(pred != y)[0]
        idx = idx[np.argsort(-probs[idx].max(axis=1))][:15]
        if len(idx) == 0:
            aviso_ax(self.fig_err.add_subplot(111), "¡Sin errores!")
        else:
            axs = self.fig_err.subplots(3, 5)
            for ax in axs.flat:
                ax.axis("off")
            for ax, i in zip(axs.flat, idx):
                ax.imshow(x_te[i, ..., 0], cmap="gray")
                ax.set_title(f"R:{y[i]} P:{pred[i]} ({probs[i].max():.0%})", color=TXT,
                             fontsize=8)
        self.lienzo_err.draw_idle()
