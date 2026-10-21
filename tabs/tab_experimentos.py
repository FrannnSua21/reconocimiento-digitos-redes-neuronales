"""Pestaña 5 · Experimentos: historial de entrenamientos y comparación de curvas."""
import csv
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import tensorflow as tf
from matplotlib.figure import Figure

from config import ACENTO, MUTED, VERDE
from ui_utils import aviso_ax, estilo_ax, etiqueta, incrustar, leyenda


class TabExperimentos(ttk.Frame):
    COLS = [("id", "#", 40), ("arq", "Arquitectura", 105), ("opt", "Optimizador", 115),
            ("lr", "LR", 70), ("sched", "Programación", 175), ("ep", "Épocas", 55),
            ("batch", "Batch", 55), ("drop", "Dropout", 60), ("aug", "Aug.", 45),
            ("val", "Val acc", 70), ("test", "Test acc", 75), ("params", "Parámetros", 90),
            ("t", "Tiempo", 60)]

    def __init__(self, padre, app):
        super().__init__(padre)
        self.app = app
        barra = ttk.Frame(self, style="Panel.TFrame", padding=10)
        barra.pack(fill="x", padx=10, pady=(10, 6))
        ttk.Button(barra, text="Usar modelo seleccionado", style="Acento.TButton",
                   command=self.usar).pack(side="left")
        ttk.Button(barra, text="Eliminar", command=self.eliminar).pack(side="left", padx=6)
        ttk.Button(barra, text="Exportar CSV", command=self.exportar).pack(side="left")
        self.lbl = etiqueta(barra, "", "Muted.TLabel")
        self.lbl.pack(side="left", padx=14)

        marco = ttk.Frame(self, style="Panel.TFrame", padding=6)
        marco.pack(fill="x", padx=10)
        self.tabla = ttk.Treeview(marco, columns=[c[0] for c in self.COLS], show="headings",
                                  height=7, selectmode="extended")
        for c, t, w in self.COLS:
            self.tabla.heading(c, text=t)
            self.tabla.column(c, width=w, anchor="center")
        sb = ttk.Scrollbar(marco, orient="vertical", command=self.tabla.yview)
        self.tabla.configure(yscrollcommand=sb.set)
        self.tabla.pack(side="left", fill="x", expand=True)
        sb.pack(side="left", fill="y")
        self.tabla.bind("<<TreeviewSelect>>", lambda e: self.dibujar())
        self.tabla.bind("<Double-1>", lambda e: self.usar())

        graf = ttk.Frame(self, style="Panel.TFrame", padding=6)
        graf.pack(fill="both", expand=True, padx=10, pady=10)
        self.fig = Figure(figsize=(9, 3.4), dpi=100, constrained_layout=True)
        self.axs = self.fig.subplots(1, 2)
        self.lienzo = incrustar(self.fig, graf)
        self.refrescar()

    def refrescar(self):
        self.tabla.delete(*self.tabla.get_children())
        for e in self.app.experimentos:
            self.tabla.insert("", "end", iid=str(e["id"]), values=(
                e["id"], e["arq"], e["opt"], f"{e['lr']:g}", e["sched"], e["epocas"],
                e["batch"], e["dropout"], "sí" if e["aug"] else "no", f"{e['val_acc']:.2%}",
                f"{e['test_acc']:.2%}", f"{e['params']:,}", f"{e['tiempo']:.0f}s"))
        if self.app.experimentos:
            mejor = max(self.app.experimentos, key=lambda e: e["test_acc"])
            self.lbl.config(text=f"{len(self.app.experimentos)} experimentos · mejor: "
                                 f"#{mejor['id']} {mejor['arq']} con {mejor['test_acc']:.2%}")
        else:
            self.lbl.config(text="Aún no hay experimentos: entrena un modelo.")
        self.dibujar()

    def seleccion(self):
        ids = {int(i) for i in self.tabla.selection()}
        return [e for e in self.app.experimentos if e["id"] in ids]

    def dibujar(self):
        todos = self.app.experimentos
        usar = self.seleccion() or todos
        a1, a2 = self.axs
        for ax, t in zip(self.axs, ["Precisión de validación por época",
                                    "Precisión en prueba por experimento"]):
            ax.clear()
            estilo_ax(ax, t)
        if not todos:
            for ax in self.axs:
                aviso_ax(ax, "Sin experimentos")
        else:
            for e in usar:
                h = e["hist"]["val_accuracy"]
                a1.plot(range(1, len(h) + 1), h, marker="o", ms=3,
                        label=f"#{e['id']} {e['arq']} lr={e['lr']:g}")
            a1.set_xlabel("Época", color=MUTED, fontsize=8)
            leyenda(a1, loc="lower right")
            accs = [e["test_acc"] * 100 for e in todos]
            colores = [VERDE if e in usar and self.tabla.selection() else ACENTO
                       for e in todos]
            a2.bar([f"#{e['id']}" for e in todos], accs, color=colores)
            a2.grid(axis="x", visible=False)
            a2.set_ylim(max(0, min(accs) - 2), 100)
            a2.set_ylabel("%", color=MUTED, fontsize=8)
        self.lienzo.draw_idle()

    def usar(self):
        sel = self.seleccion()
        if not sel:
            return
        ruta = Path(sel[0]["ruta"])
        if not ruta.exists():
            messagebox.showwarning("Modelo no encontrado", f"No existe el archivo:\n{ruta}")
            return
        self.app.activar(tf.keras.models.load_model(ruta), ruta.name, sel[0]["test_acc"])

    def eliminar(self):
        sel = self.seleccion()
        if sel and messagebox.askyesno("Eliminar", f"¿Eliminar {len(sel)} experimento(s) del "
                                                   "historial? (los modelos guardados se "
                                                   "conservan)"):
            ids = {e["id"] for e in sel}
            self.app.experimentos = [e for e in self.app.experimentos if e["id"] not in ids]
            self.app.guardar_experimentos()
            self.refrescar()

    def exportar(self):
        ruta = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV", "*.csv")])
        if not ruta:
            return
        campos = ["id", "fecha", "arq", "opt", "lr", "sched", "epocas", "batch", "dropout",
                  "l2", "aug", "subset", "val_acc", "test_acc", "test_loss", "params", "tiempo"]
        with open(ruta, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.app.experimentos)
