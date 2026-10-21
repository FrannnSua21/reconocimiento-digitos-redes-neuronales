"""Pestaña 1 · Entrenar: hiperparámetros, curvas en vivo y barridos.
"""
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import numpy as np
from matplotlib.figure import Figure

from config import (ACENTO, AMARILLO, ARQUITECTURAS, BATCHES, CIAN, DEFECTOS, MUTED,
                     OPTIMIZADORES, PANEL2, PROGRAMACIONES, ROSA, TXT, VARIABLES_BARRIDO, VERDE)
from nucleo import construir_modelo
from ui_utils import (aviso_ax, combo, deslizador, estilo_ax, etiqueta, fila, incrustar,
                      leyenda)


class TabEntrenamiento(ttk.Frame):
    def __init__(self, padre, app):
        super().__init__(padre)
        self.app = app
        self.hist = None
        self.total_epocas = 1
        self._entrenando = False
        d = DEFECTOS
        self.v = {
            "arq": tk.StringVar(value=d["arq"]), "opt": tk.StringVar(value=d["opt"]),
            "lr": tk.StringVar(value=d["lr"]), "sched": tk.StringVar(value=d["sched"]),
            "dropout": tk.DoubleVar(value=d["dropout"]), "l2": tk.StringVar(value=d["l2"]),
            "epocas": tk.IntVar(value=d["epocas"]), "batch": tk.StringVar(value=d["batch"]),
            "val": tk.DoubleVar(value=d["val"]), "subset": tk.DoubleVar(value=d["subset"]),
            "es": tk.BooleanVar(value=d["es"]), "paciencia": tk.IntVar(value=d["paciencia"]),
            "aug": tk.BooleanVar(value=d["aug"]), "semilla": tk.IntVar(value=d["semilla"]),
        }
        self.v_variar = tk.StringVar(value="Tasa de aprendizaje")
        self.v_valores = tk.StringVar(value=VARIABLES_BARRIDO["Tasa de aprendizaje"][2])
        self.construir()
        self.activo(False)
        self.dibujar()

    def construir(self):
        v = self.v

        # ----- banner guiado -----
        self.banner = ttk.Frame(self, style="Banner.TFrame", padding=(16, 10))
        self.banner.pack(fill="x", padx=10, pady=(10, 0))
        self.banner.columnconfigure(0, weight=1)
        textos = ttk.Frame(self.banner, style="Banner.TFrame")
        textos.grid(row=0, column=0, sticky="w")
        self.lbl_banner_t = ttk.Label(textos, style="BannerTitulo.TLabel")
        self.lbl_banner_t.pack(anchor="w")
        self.lbl_banner_d = ttk.Label(textos, style="Banner.TLabel", wraplength=880)
        self.lbl_banner_d.pack(anchor="w", pady=(2, 0))
        self.btn_ir = ttk.Button(self.banner, text="Ir a 2 · Reconocer  →",
                                 style="Acento.TButton", command=self.app.ir_a_reconocer)
        self.btn_ir.grid(row=0, column=1, sticky="e", padx=(12, 0))

        sup = ttk.Frame(self)
        sup.pack(fill="x", padx=10, pady=(6, 6))
        for c in range(3):
            sup.columnconfigure(c, weight=1, uniform="g")

        g1 = ttk.LabelFrame(sup, text=" Modelo y optimización ", padding=10)
        g1.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        g1.columnconfigure(1, weight=1)
        fila(g1, 0, "Arquitectura", combo(g1, v["arq"], ARQUITECTURAS, state="readonly"))
        fila(g1, 1, "Optimizador", combo(g1, v["opt"], OPTIMIZADORES, state="readonly"))
        fila(g1, 2, "Tasa de aprendizaje", combo(
            g1, v["lr"], ["0.1", "0.03", "0.01", "0.003", "0.001", "0.0003", "0.0001"]))
        fila(g1, 3, "Programación del LR", combo(g1, v["sched"], PROGRAMACIONES,
                                                  state="readonly"))
        deslizador(g1, 4, "Dropout", v["dropout"], 0, 0.7)
        fila(g1, 5, "Regularización L2", combo(g1, v["l2"], ["0", "0.0001", "0.001", "0.01"]))

        g2 = ttk.LabelFrame(sup, text=" Entrenamiento y datos ", padding=10)
        g2.grid(row=0, column=1, sticky="nsew", padx=6)
        g2.columnconfigure(1, weight=1)
        fila(g2, 0, "Épocas", ttk.Spinbox(g2, from_=1, to=200, textvariable=v["epocas"],
                                          width=8))
        fila(g2, 1, "Tamaño de batch", combo(g2, v["batch"], BATCHES, state="readonly"))
        deslizador(g2, 2, "Validación (%)", v["val"], 5, 30, "{:.0f}")
        deslizador(g2, 3, "Datos usados (%)", v["subset"], 5, 100, "{:.0f}")
        es = ttk.Frame(g2, style="Panel.TFrame")
        ttk.Checkbutton(es, text="Early stopping", variable=v["es"]).pack(side="left")
        etiqueta(es, "paciencia", "Muted.TLabel").pack(side="left", padx=(10, 4))
        ttk.Spinbox(es, from_=1, to=20, textvariable=v["paciencia"], width=4).pack(side="left")
        fila(g2, 4, "", es)
        fila(g2, 5, "", ttk.Checkbutton(g2, text="Aumento de datos",
                                        variable=v["aug"]))
        fila(g2, 6, "Semilla", ttk.Spinbox(g2, from_=0, to=99999, textvariable=v["semilla"],
                                           width=8))

        g3 = ttk.LabelFrame(sup, text=" Acciones ", padding=10)
        g3.grid(row=0, column=2, sticky="nsew", padx=(6, 0))
        g3.columnconfigure(0, weight=1)
        botones = ttk.Frame(g3, style="Panel.TFrame")
        botones.grid(row=0, column=0, sticky="ew")
        self.btn_entrenar = ttk.Button(botones, text="▶  Entrenar", style="Acento.TButton",
                                       command=self.entrenar)
        self.btn_entrenar.pack(side="left")
        self.btn_parar = ttk.Button(botones, text="■  Detener", state="disabled",
                                    command=self.app.detener)
        self.btn_parar.pack(side="left", padx=6)
        ttk.Button(botones, text="Ver capas", command=self.ver_arquitectura).pack(
            side="left")
        self.progreso = ttk.Progressbar(g3, mode="determinate", maximum=1000)
        self.progreso.grid(row=1, column=0, sticky="ew", pady=(10, 2))
        self.lbl_estado = etiqueta(g3, "Listo para entrenar.", "Muted.TLabel", wraplength=330)
        self.lbl_estado.grid(row=2, column=0, sticky="w")
        ttk.Separator(g3).grid(row=3, column=0, sticky="ew", pady=8)
        etiqueta(g3, "Barrido automático: entrena varias veces cambiando una variable",
                 "Muted.TLabel", wraplength=330).grid(row=4, column=0, sticky="w")
        fb = ttk.Frame(g3, style="Panel.TFrame")
        fb.grid(row=5, column=0, sticky="ew", pady=4)
        fb.columnconfigure(1, weight=1)
        cb = combo(fb, self.v_variar, list(VARIABLES_BARRIDO), 18, state="readonly")
        cb.grid(row=0, column=0, padx=(0, 6))
        cb.bind("<<ComboboxSelected>>", lambda e: self.v_valores.set(
            VARIABLES_BARRIDO[self.v_variar.get()][2]))
        ttk.Entry(fb, textvariable=self.v_valores).grid(row=0, column=1, sticky="ew")
        self.btn_barrido = ttk.Button(g3, text="Ejecutar barrido", command=self.barrido)
        self.btn_barrido.grid(row=6, column=0, sticky="ew", pady=(4, 0))

        inf = ttk.Frame(self)
        inf.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        inf.columnconfigure(0, weight=1)
        inf.rowconfigure(0, weight=1)
        izq = ttk.Frame(inf, style="Panel.TFrame", padding=6)
        izq.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.fig = Figure(figsize=(6.6, 3.4), dpi=100, constrained_layout=True)
        self.axs = self.fig.subplots(1, 3)
        self.lienzo = incrustar(self.fig, izq)
        der = ttk.Frame(inf, style="Panel.TFrame", padding=6)
        der.grid(row=0, column=1, sticky="nsew")
        self.txt = ScrolledText(der, width=46, bg=PANEL2, fg=TXT, bd=0, highlightthickness=0,
                                font=("Consolas", 9), insertbackground=TXT, wrap="word",
                                state="disabled")
        self.txt.pack(fill="both", expand=True)

    # ----- banner -----
    def actualizar_banner(self):
        app = self.app
        mostrar_ir = False
        if app.datos is None:
            titulo, color = "Preparando los datos…", MUTED
            detalle = ("Descargando MNIST (solo la primera vez, necesita internet). "
                       "El botón Entrenar se activará en cuanto termine.")
        elif self._entrenando:
            titulo, color = "Entrenando…", AMARILLO
            detalle = ("Mira las curvas y el registro de abajo. Puedes detener el "
                       "entrenamiento con ■ Detener.")
        elif app.modelo is None:
            titulo, color = "Paso 1 · Entrena un modelo", ACENTO
            detalle = ("Todavía no hay ningún modelo entrenado. Los valores ya vienen listos: "
                       "pulsa ▶ Entrenar (tarda unos segundos). Después podrás dibujar números "
                       "en la pestaña 2 · Reconocer.")
        else:
            titulo, color = f"✔ Ya hay un modelo entrenado · {app.nombre_modelo}", VERDE
            detalle = ("Ya puedes dibujar números en 2 · Reconocer. Si quieres, entrena otro "
                       "cambiando los parámetros y compáralos en 5 · Experimentos.")
            mostrar_ir = True
        self.lbl_banner_t.config(text=titulo, foreground=color)
        self.lbl_banner_d.config(text=detalle)
        if mostrar_ir:
            self.btn_ir.grid()
        else:
            self.btn_ir.grid_remove()

    # ----- configuración -----
    def leer_config(self, **cambios):
        v = self.v
        try:
            lr, l2 = float(v["lr"].get()), float(v["l2"].get())
            if lr <= 0 or l2 < 0:
                raise ValueError
            cfg = {"arq": v["arq"].get(), "opt": v["opt"].get(), "lr": lr,
                   "sched": v["sched"].get(), "dropout": round(v["dropout"].get(), 2),
                   "l2": l2, "epocas": int(v["epocas"].get()), "batch": int(v["batch"].get()),
                   "val": int(round(v["val"].get())), "subset": int(round(v["subset"].get())),
                   "es": v["es"].get(), "paciencia": int(v["paciencia"].get()),
                   "aug": v["aug"].get(), "semilla": int(v["semilla"].get())}
        except (ValueError, tk.TclError):
            raise ValueError("Revisa los parámetros: la tasa de aprendizaje debe ser un "
                             "número > 0, L2 un número ≥ 0 y los demás campos enteros.")
        cfg.update(cambios)
        return cfg

    def entrenar(self):
        try:
            self.app.iniciar([self.leer_config()])
        except ValueError as e:
            messagebox.showwarning("Parámetros inválidos", str(e))

    def barrido(self):
        clave, tipo, _ = VARIABLES_BARRIDO[self.v_variar.get()]
        try:
            crudos = [s.strip() for s in self.v_valores.get().split(",") if s.strip()]
            if len(crudos) < 2:
                raise ValueError("Escribe al menos dos valores separados por comas.")
            valores = [tipo(s) for s in crudos]
            validos = {"opt": OPTIMIZADORES, "arq": ARQUITECTURAS}.get(clave)
            if validos and any(x not in validos for x in valores):
                raise ValueError(f"Valores permitidos: {', '.join(validos)}")
            if clave == "lr" and any(x <= 0 for x in valores):
                raise ValueError("Las tasas de aprendizaje deben ser positivas.")
            configs = [self.leer_config(**{clave: x}) for x in valores]
        except ValueError as e:
            messagebox.showwarning("Barrido inválido", str(e))
            return
        self.app.iniciar(configs)

    def ver_arquitectura(self):
        try:
            cfg = self.leer_config()
        except ValueError as e:
            messagebox.showwarning("Parámetros inválidos", str(e))
            return
        modelo = construir_modelo(cfg["arq"], cfg["dropout"], cfg["l2"], cfg["aug"])
        lineas = []
        modelo.summary(print_fn=lambda s, **k: lineas.append(s))
        self.log("\n" + "\n".join(lineas))

    # ----- eventos del entrenamiento -----
    def activo(self, si):
        """Activa/desactiva botones según haya un entrenamiento en curso o datos listos."""
        self._entrenando = si
        puede = (not si) and self.app.datos is not None
        self.btn_entrenar.config(state="normal" if puede else "disabled")
        self.btn_barrido.config(state="normal" if puede else "disabled")
        self.btn_parar.config(state="normal" if si else "disabled")
        self.actualizar_banner()

    def log(self, texto):
        self.txt.config(state="normal")
        self.txt.insert("end", texto + "\n")
        self.txt.see("end")
        self.txt.config(state="disabled")

    def on_run_ini(self, i, total, cfg):
        self.hist = None
        self.dibujar()
        self.total_epocas = cfg["epocas"]
        self.progreso["value"] = 0
        self.lbl_estado.config(text=f"Entrenamiento {i + 1} de {total}: {cfg['arq']} · "
                                    f"{cfg['opt']} · lr={cfg['lr']:g}")
        self.log(f"\n━━ {i + 1}/{total} · {cfg['arq']} · {cfg['opt']} · lr={cfg['lr']:g} · "
                 f"{cfg['sched']} · batch {cfg['batch']} ━━")

    def on_batch(self, epoca, frac):
        self.progreso["value"] = 1000 * (epoca + frac) / self.total_epocas

    def on_epoca(self, d):
        self.hist, h, e = d["hist"], d["hist"], d["epoca"]
        self.log(f"Época {e:>3}/{d['total']}  loss {h['loss'][-1]:.4f}  acc {h['accuracy'][-1]:.4f}"
                 f"  │ val_loss {h['val_loss'][-1]:.4f}  val_acc {h['val_accuracy'][-1]:.4f}"
                 f"  │ lr {h['lr'][-1]:.2e}")
        self.progreso["value"] = 1000 * e / d["total"]
        self.dibujar()

    def on_fin(self, res):
        self.log(f"✔ Prueba (10 000 imágenes): precisión {res['test_acc']:.2%} · pérdida "
                 f"{res['test_loss']:.4f} · {res['params']:,} parámetros · {res['tiempo']:.0f} s")
        self.lbl_estado.config(text=f"Terminado: precisión en prueba {res['test_acc']:.2%}")

    # ----- gráficos -----
    def dibujar(self):
        h = self.hist
        for ax, t in zip(self.axs, ["Pérdida", "Precisión", "Tasa de aprendizaje"]):
            ax.clear()
            estilo_ax(ax, t)
        if not (h and h["loss"]):
            for ax in self.axs:
                aviso_ax(ax, "Sin datos aún")
        else:
            ep = range(1, len(h["loss"]) + 1)
            a0, a1, a2 = self.axs
            a0.plot(ep, h["loss"], color=CIAN, marker="o", ms=3, label="entrenamiento")
            a0.plot(ep, h["val_loss"], color=AMARILLO, marker="o", ms=3, label="validación")
            a1.plot(ep, h["accuracy"], color=VERDE, marker="o", ms=3, label="entrenamiento")
            a1.plot(ep, h["val_accuracy"], color=ROSA, marker="o", ms=3, label="validación")
            a2.plot(ep, h["lr"], color=ACENTO, marker="o", ms=3)
            if np.all(np.isfinite(h["lr"])) and min(h["lr"]) > 0:
                a2.set_yscale("log")
                a2.tick_params(which="both", colors=MUTED, labelsize=8)
            leyenda(a0)
            leyenda(a1, loc="lower right")
            for ax in self.axs:
                ax.set_xlabel("Época", color=MUTED, fontsize=8)
        self.lienzo.draw_idle()
