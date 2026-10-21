"""Pestaña 2 · Reconocer: lienzo de dibujo y predicción en tiempo real.
"""
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

import numpy as np
import tensorflow as tf
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageTk

from config import (ACENTO, AMARILLO, BORDE, FUENTE, MUTED, PANEL, PANEL2, TAM, TXT, TXT_ACENTO,
                     VERDE, VISTA)
from nucleo import preparar_imagen
from ui_utils import etiqueta, incrustar


class TabReconocer(ttk.Frame):
    def __init__(self, padre, app):
        super().__init__(padre)
        self.app = app
        self.imagen = Image.new("L", (TAM, TAM), 0)
        self.pincel = ImageDraw.Draw(self.imagen)
        self.trazos = []
        self.pendiente = False
        self.ultimo_x = None
        self.grosor = tk.IntVar(value=30)
        self.tiempo_real = tk.BooleanVar(value=True)
        self.construir()
        self.actualizar_estado()

    def construir(self):
        izq = ttk.Frame(self, style="Panel.TFrame", padding=16)
        izq.pack(side="left", fill="y", padx=(10, 6), pady=10)
        etiqueta(izq, "Dibuja un dígito del 0 al 9", "Titulo.TLabel").pack(anchor="w")
        self.lbl_sub = etiqueta(izq, "", "Muted.TLabel", wraplength=TAM)
        self.lbl_sub.pack(anchor="w", pady=(0, 10))
        self.lienzo = tk.Canvas(izq, width=TAM, height=TAM, bg="black", cursor="crosshair",
                                highlightthickness=2, highlightbackground=ACENTO)
        self.lienzo.pack()
        self.lienzo.bind("<ButtonPress-1>", self.presionar)
        self.lienzo.bind("<B1-Motion>", self.mover)
        self.lienzo.bind("<ButtonRelease-1>", lambda e: self.reconocer(registrar=True))

        # Aviso que cubre el lienzo mientras no haya modelo.
        self.aviso = tk.Frame(izq, bg=PANEL2, highlightthickness=2, highlightbackground=BORDE)
        centro = tk.Frame(self.aviso, bg=PANEL2)
        centro.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(centro, text="Aún no hay un modelo entrenado", bg=PANEL2, fg=TXT,
                 font=(FUENTE, 15, "bold"), wraplength=330).pack()
        tk.Label(centro, text="Para reconocer dígitos primero hay que entrenar la red. "
                              "Es rápido: con los valores por defecto tarda unos segundos.",
                 bg=PANEL2, fg=MUTED, font=(FUENTE, 10), wraplength=330,
                 justify="center").pack(pady=(8, 16))
        ttk.Button(centro, text="Ir a 1 · Entrenar  →", style="Acento.TButton",
                   command=self.app.ir_a_entrenar).pack(fill="x")
        ttk.Button(centro, text="Cargar un modelo existente…",
                   command=self.app.cargar_modelo).pack(fill="x", pady=(8, 0))

        barra = ttk.Frame(izq, style="Panel.TFrame")
        barra.pack(fill="x", pady=(12, 4))
        etiqueta(barra, "Grosor").pack(side="left")
        ttk.Scale(barra, from_=10, to=60, variable=self.grosor, length=150).pack(
            side="left", padx=8)
        ttk.Checkbutton(barra, text="Tiempo real", variable=self.tiempo_real).pack(
            side="right")
        botones = ttk.Frame(izq, style="Panel.TFrame")
        botones.pack(fill="x", pady=4)
        ttk.Button(botones, text="Limpiar", command=self.limpiar).pack(side="left")
        ttk.Button(botones, text="Deshacer", command=self.deshacer).pack(side="left", padx=6)
        ttk.Button(botones, text="Guardar", command=self.guardar).pack(side="left")
        ttk.Button(botones, text="Abrir imagen", command=self.abrir).pack(side="right")

        der = ttk.Frame(self, style="Panel.TFrame", padding=16)
        der.pack(side="left", fill="both", expand=True, padx=(6, 10), pady=10)
        etiqueta(der, "Predicción", "Titulo.TLabel").pack(anchor="w")
        cab = ttk.Frame(der, style="Panel.TFrame")
        cab.pack(fill="x")
        self.lbl_digito = tk.Label(cab, text="–", bg=PANEL, fg=VERDE, width=3,
                                   font=(FUENTE, 88, "bold"))
        self.lbl_digito.pack(side="left")
        col = ttk.Frame(cab, style="Panel.TFrame")
        col.pack(side="left", padx=18)
        self.lbl_conf = etiqueta(col, "Sin dibujo", "Panel.TLabel", font=(FUENTE, 13))
        self.lbl_conf.pack(anchor="w")
        self.lbl_info = etiqueta(col, "", "Muted.TLabel", wraplength=300)
        self.lbl_info.pack(anchor="w", pady=(6, 0))

        medio = ttk.Frame(der, style="Panel.TFrame")
        medio.pack(fill="x", pady=8)
        self.barras = tk.Canvas(medio, width=340, height=262, bg=PANEL, highlightthickness=0)
        self.barras.pack(side="left")
        vista = ttk.Frame(medio, style="Panel.TFrame")
        vista.pack(side="left", padx=20, anchor="n")
        etiqueta(vista, "Lo que ve la red (28×28)", "Muted.TLabel").pack()
        self.vista = tk.Label(vista, bg="black", width=VISTA, height=VISTA)
        self.vista.pack(pady=6)
        self.mostrar_vista(np.zeros((28, 28), dtype="float32"))
        ttk.Button(vista, text="Mapas de activación", command=self.mostrar_mapas).pack(
            fill="x", pady=(8, 0))

        etiqueta(der, "Historial de reconocimientos", "Muted.TLabel").pack(anchor="w")
        self.historial = tk.Listbox(der, height=5, bg=PANEL2, fg=TXT, bd=0,
                                    highlightthickness=0, font=("Consolas", 11),
                                    selectbackground=ACENTO, selectforeground=TXT_ACENTO)
        self.historial.pack(fill="x", pady=4)
        self.mostrar_probs(None)

    # ----- estado del modelo -----
    def actualizar_estado(self):
        """Muestra u oculta el aviso según haya o no un modelo activo."""
        if self.app.modelo is None:
            self.aviso.place(in_=self.lienzo, x=0, y=0, relwidth=1, relheight=1)
            self.aviso.lift()
            self.lbl_sub.config(text="Primero entrena un modelo en la pestaña 1 · Entrenar.",
                                foreground=AMARILLO)
        else:
            self.aviso.place_forget()
            self.lbl_sub.config(text="✔ Modelo listo: dibuja y la red lo reconoce al instante.",
                                foreground=VERDE)

    # ----- dibujo -----
    def segmento(self, x0, y0, x1, y1, g):
        h = g / 2
        self.lienzo.create_line(x0, y0, x1, y1, width=g, fill="white", capstyle=tk.ROUND)
        self.lienzo.create_oval(x1 - h, y1 - h, x1 + h, y1 + h, fill="white", outline="white")
        self.pincel.line([x0, y0, x1, y1], fill=255, width=g)
        self.pincel.ellipse([x1 - h, y1 - h, x1 + h, y1 + h], fill=255)

    def presionar(self, e):
        if self.app.modelo is None:
            return
        self.trazos.append({"pts": [(e.x, e.y)], "g": self.grosor.get()})
        self.segmento(e.x, e.y, e.x, e.y, self.trazos[-1]["g"])
        self.programar()

    def mover(self, e):
        if not self.trazos or self.app.modelo is None:
            return
        t = self.trazos[-1]
        x0, y0 = t["pts"][-1]
        t["pts"].append((e.x, e.y))
        self.segmento(x0, y0, e.x, e.y, t["g"])
        self.programar()

    def redibujar(self):
        self.lienzo.delete("all")
        self.pincel.rectangle([0, 0, TAM, TAM], fill=0)
        for t in self.trazos:
            p = t["pts"]
            self.segmento(*p[0], *p[0], t["g"])
            for a, b in zip(p, p[1:]):
                self.segmento(*a, *b, t["g"])

    def limpiar(self):
        self.trazos.clear()
        self.redibujar()
        self.reconocer()

    def deshacer(self):
        if self.trazos:
            self.trazos.pop()
            self.redibujar()
            self.reconocer()

    def guardar(self):
        ruta = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png")])
        if ruta:
            self.imagen.save(ruta)

    def abrir(self):
        if self.app.modelo is None:
            messagebox.showinfo("Sin modelo", "Primero entrena o carga un modelo "
                                              "(pestaña 1 · Entrenar).")
            return
        ruta = filedialog.askopenfilename(
            filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp")])
        if ruta:
            self.limpiar()
            self.analizar(Image.open(ruta), registrar=True)

    # ----- reconocimiento -----
    def programar(self):
        """Limita la predicción a ~25 por segundo mientras se dibuja."""
        if self.tiempo_real.get() and not self.pendiente:
            self.pendiente = True
            self.after(40, self.reconocer)

    def reconocer(self, registrar=False):
        self.pendiente = False
        self.analizar(self.imagen, registrar)

    def analizar(self, img, registrar=False):
        if self.app.modelo is None:
            self.mostrar(None)
            self.lbl_conf.config(text="Sin modelo: entrena o carga uno")
            return
        x = preparar_imagen(img)
        self.ultimo_x = x
        self.mostrar_vista(x)
        if x.max() == 0:
            self.mostrar(None)
            return
        probs = self.app.predecir(x)
        self.mostrar(probs)
        if registrar:
            d = int(probs.argmax())
            self.historial.insert(0, f"  {d}   {probs[d]:6.1%}   {datetime.now():%H:%M:%S}")
            if self.historial.size() > 30:
                self.historial.delete(30, "end")

    def actualizar_info(self, texto):
        self.lbl_info.config(text=texto)

    # ----- visualización -----
    def mostrar(self, probs):
        if probs is None:
            self.lbl_digito.config(text="–")
            self.lbl_conf.config(text="Sin dibujo")
        else:
            d = int(probs.argmax())
            self.lbl_digito.config(text=str(d))
            self.lbl_conf.config(text=f"Confianza: {probs[d]:.1%}")
        self.mostrar_probs(probs)

    def mostrar_probs(self, probs):
        c = self.barras
        c.delete("all")
        mejor = int(probs.argmax()) if probs is not None else -1
        for i in range(10):
            y = 4 + i * 26
            p = float(probs[i]) if probs is not None else 0.0
            c.create_text(8, y + 10, text=str(i), fill=TXT, font=("Consolas", 12, "bold"))
            c.create_rectangle(26, y, 26 + 230, y + 20, fill=PANEL2, outline="")
            c.create_rectangle(26, y, 26 + 230 * p, y + 20,
                               fill=VERDE if i == mejor else ACENTO, outline="")
            c.create_text(264, y + 10, text=f"{p:.1%}", fill=TXT, anchor="w",
                          font=("Consolas", 10))

    def mostrar_vista(self, x):
        im = Image.fromarray((x * 255).astype("uint8")).resize((VISTA, VISTA), Image.NEAREST)
        self.foto = ImageTk.PhotoImage(im)
        self.vista.config(image=self.foto, width=VISTA, height=VISTA)

    def mostrar_mapas(self):
        m = self.app.modelo
        if m is None or self.ultimo_x is None or self.ultimo_x.max() == 0:
            messagebox.showinfo("Mapas de activación", "Primero dibuja un número.")
            return
        h = tf.convert_to_tensor(self.ultimo_x[np.newaxis, ..., np.newaxis])
        mapas = []
        for capa in m.layers:
            if isinstance(capa, tf.keras.layers.InputLayer):
                continue
            h = capa(h, training=False)
            if isinstance(capa, tf.keras.layers.Conv2D):
                mapas.append((capa.name, h.numpy()[0]))
        if not mapas:
            messagebox.showinfo("Mapas de activación",
                                "El modelo activo no tiene capas convolucionales (¿es un MLP?).")
            return
        win = tk.Toplevel(self.app.raiz)
        win.title("Mapas de activación de las capas convolucionales")
        win.configure(bg=PANEL)
        win.geometry("920x580")
        mapas = mapas[:3]
        fig = Figure(figsize=(9, 5.6), dpi=100, constrained_layout=True)
        gs = fig.add_gridspec(len(mapas) + 1, 8)
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(self.ultimo_x, cmap="gray")
        ax.set_title("Entrada 28×28", color=TXT, fontsize=9)
        ax.axis("off")
        for r, (nombre, a) in enumerate(mapas, start=1):
            for c in range(min(8, a.shape[-1])):
                ax = fig.add_subplot(gs[r, c])
                ax.imshow(a[..., c], cmap="magma")
                ax.axis("off")
                if c == 0:
                    ax.set_title(f"{nombre} · {a.shape[-1]} filtros (se muestran 8)",
                                 color=TXT, fontsize=9, loc="left")
        incrustar(fig, win)
