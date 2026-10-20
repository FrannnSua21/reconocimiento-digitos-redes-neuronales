"""Reconocedor de dígitos en tiempo real (TensorFlow/Keras + Tkinter)."""
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageTk

RUTA_MODELO = Path(__file__).resolve().parent / "modelo_mnist.keras"
TAM = 360       # lado del lienzo de dibujo
VISTA = 112     # lado de la vista previa 28x28 ampliada
BG, PANEL, TXT, ACENTO = "#1e1e2e", "#2a2a3d", "#e6e6f0", "#4ade80"


# ----------------------------- Modelo -----------------------------
def construir_modelo():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])


def entrenar_modelo(epocas=5, avisar=None):
    (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.mnist.load_data()
    x_tr = (x_tr[..., np.newaxis] / 255.0).astype("float32")
    x_te = (x_te[..., np.newaxis] / 255.0).astype("float32")

    modelo = construir_modelo()
    modelo.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])

    class Progreso(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if avisar:
                avisar(f"Entrenando… época {epoch + 1}/{epocas} "
                       f"(val_acc {logs['val_accuracy']:.4f})")

    modelo.fit(x_tr, y_tr, epochs=epocas, batch_size=128,
               validation_split=0.1, callbacks=[Progreso()], verbose=0)
    _, precision = modelo.evaluate(x_te, y_te, verbose=0)
    modelo.save(RUTA_MODELO)
    return modelo, precision


# ------------------------- Preprocesamiento -------------------------
def preparar_imagen(img):
    """Imagen cualquiera -> array 28x28 estilo MNIST (blanco sobre negro, centrado)."""
    a = np.array(img.convert("L"), dtype="float32")
    if a.mean() > 127:                       # fondo claro -> invertir
        a = 255 - a
    a[a < 30] = 0
    ys, xs = np.where(a > 0)
    if xs.size == 0:
        return np.zeros((28, 28), dtype="float32")
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    alto, ancho = a.shape
    if max(alto, ancho) < 15:                # trazo demasiado pequeño
        return np.zeros((28, 28), dtype="float32")

    k = 20.0 / max(alto, ancho)              # el dígito ocupa 20x20 como en MNIST
    peq = Image.fromarray(a.astype("uint8")).resize(
        (max(1, round(ancho * k)), max(1, round(alto * k))), Image.LANCZOS)
    lienzo = Image.new("L", (28, 28), 0)
    lienzo.paste(peq, ((28 - peq.width) // 2, (28 - peq.height) // 2))

    m = np.array(lienzo, dtype="float32")    # centrar por centro de masa
    total = m.sum()
    if total > 0:
        cy = (m.sum(axis=1) * np.arange(28)).sum() / total
        cx = (m.sum(axis=0) * np.arange(28)).sum() / total
        m = np.roll(m, (round(14 - cy), round(14 - cx)), axis=(0, 1))
    return m / 255.0


# ----------------------------- Interfaz -----------------------------
class App:
    def __init__(self, raiz):
        self.raiz = raiz
        self.modelo = None
        self.pendiente = False
        self.ultimo = (0, 0)
        self.entrenando = False
        self.msg_hilo = ""
        self.resultado = None
        self.imagen = Image.new("L", (TAM, TAM), 0)
        self.pincel = ImageDraw.Draw(self.imagen)
        self.grosor = tk.IntVar(value=28)
        self.estado = tk.StringVar(value="Cargando modelo…")
        self.construir_interfaz()
        self.raiz.after(100, self.cargar_modelo)

    def construir_interfaz(self):
        r = self.raiz
        r.title("Reconocedor de dígitos en tiempo real")
        r.configure(bg=BG)
        r.resizable(False, False)

        izq = tk.Frame(r, bg=BG)
        izq.grid(row=0, column=0, padx=14, pady=14)
        tk.Label(izq, text="Dibuja un dígito (0-9)", bg=BG, fg=TXT,
                 font=("Segoe UI", 12, "bold")).pack(pady=(0, 6))
        self.lienzo = tk.Canvas(izq, width=TAM, height=TAM, bg="black",
                                cursor="crosshair", highlightthickness=2,
                                highlightbackground=ACENTO)
        self.lienzo.pack()
        self.lienzo.bind("<ButtonPress-1>", self.presionar)
        self.lienzo.bind("<B1-Motion>", self.mover)
        self.lienzo.bind("<ButtonRelease-1>", lambda e: self.reconocer())

        barra = tk.Frame(izq, bg=BG)
        barra.pack(pady=8, fill="x")
        tk.Label(barra, text="Grosor", bg=BG, fg=TXT).pack(side="left")
        tk.Scale(barra, from_=10, to=50, orient="horizontal", variable=self.grosor,
                 bg=BG, fg=TXT, highlightthickness=0, length=140).pack(side="left", padx=6)
        tk.Button(barra, text="Limpiar", command=self.limpiar, width=9).pack(side="right")

        der = tk.Frame(r, bg=PANEL)
        der.grid(row=0, column=1, padx=(0, 14), pady=14, sticky="ns")
        tk.Label(der, text="Predicción", bg=PANEL, fg=TXT,
                 font=("Segoe UI", 11)).pack(pady=(10, 0))
        self.lbl_digito = tk.Label(der, text="–", bg=PANEL, fg=ACENTO,
                                   font=("Segoe UI", 80, "bold"), width=3)
        self.lbl_digito.pack()
        self.lbl_conf = tk.Label(der, text="", bg=PANEL, fg=TXT, font=("Segoe UI", 11))
        self.lbl_conf.pack()

        self.barras = tk.Canvas(der, width=290, height=260, bg=PANEL, highlightthickness=0)
        self.barras.pack(padx=12, pady=8)

        tk.Label(der, text="Lo que ve la red (28x28)", bg=PANEL, fg=TXT).pack()
        self.vista = tk.Label(der, bg="black")
        self.vista.pack(pady=4)
        self.mostrar_vista(np.zeros((28, 28), dtype="float32"))

        botones = tk.Frame(der, bg=PANEL)
        botones.pack(pady=6)
        tk.Button(botones, text="Abrir imagen", command=self.abrir_imagen).pack(side="left", padx=4)
        self.btn_entrenar = tk.Button(botones, text="Reentrenar", command=self.entrenar)
        self.btn_entrenar.pack(side="left", padx=4)
        tk.Label(der, textvariable=self.estado, bg=PANEL, fg=TXT, wraplength=280,
                 font=("Segoe UI", 9)).pack(pady=(2, 10))
        self.mostrar_probs(None)

    # ----- dibujo -----
    def presionar(self, e):
        self.ultimo = (e.x, e.y)
        self.trazo(e.x, e.y)
        self.programar()

    def mover(self, e):
        self.trazo(e.x, e.y)
        self.programar()

    def trazo(self, x, y):
        g = self.grosor.get()
        h = g / 2
        x0, y0 = self.ultimo
        self.lienzo.create_line(x0, y0, x, y, width=g, fill="white",
                                capstyle=tk.ROUND)
        self.lienzo.create_oval(x - h, y - h, x + h, y + h, fill="white", outline="white")
        self.pincel.line([x0, y0, x, y], fill=255, width=g)
        self.pincel.ellipse([x - h, y - h, x + h, y + h], fill=255)
        self.ultimo = (x, y)

    def limpiar(self):
        self.lienzo.delete("all")
        self.pincel.rectangle([0, 0, TAM, TAM], fill=0)
        self.mostrar_resultado(None)

    # ----- reconocimiento en tiempo real -----
    def programar(self):
        """Limita la predicción a ~20 por segundo mientras se dibuja."""
        if not self.pendiente and self.modelo is not None:
            self.pendiente = True
            self.raiz.after(50, self.reconocer)

    def reconocer(self):
        self.pendiente = False
        if self.modelo is None:
            return
        self.predecir_imagen(self.imagen)

    def predecir_imagen(self, img):
        x = preparar_imagen(img)
        self.mostrar_vista(x)
        if x.max() == 0:
            self.mostrar_resultado(None)
            return
        probs = self.modelo(x[np.newaxis, ..., np.newaxis], training=False).numpy()[0]
        self.mostrar_resultado(probs)

    # ----- visualización -----
    def mostrar_resultado(self, probs):
        if probs is None:
            self.lbl_digito.config(text="–")
            self.lbl_conf.config(text="")
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
            y = 6 + i * 25
            p = float(probs[i]) if probs is not None else 0.0
            c.create_text(8, y + 9, text=str(i), fill=TXT, font=("Consolas", 11, "bold"))
            c.create_rectangle(24, y, 24 + 200, y + 18, fill="#3a3a52", outline="")
            c.create_rectangle(24, y, 24 + 200 * p, y + 18,
                               fill=ACENTO if i == mejor else "#6b7bd6", outline="")
            c.create_text(232, y + 9, text=f"{p:.1%}", fill=TXT, anchor="w",
                          font=("Consolas", 9))

    def mostrar_vista(self, x):
        im = Image.fromarray((x * 255).astype("uint8")).resize((VISTA, VISTA), Image.NEAREST)
        self.foto = ImageTk.PhotoImage(im)
        self.vista.config(image=self.foto)

    # ----- modelo -----
    def cargar_modelo(self):
        if RUTA_MODELO.exists():
            try:
                self.modelo = tf.keras.models.load_model(RUTA_MODELO)
                self.estado.set("Modelo cargado. ¡Dibuja un número!")
                return
            except Exception as ex:
                self.estado.set(f"No se pudo cargar el modelo: {ex}")
                return
        self.estado.set("No hay modelo guardado. Presiona «Reentrenar».")

    def entrenar(self):
        if self.entrenando:
            return
        self.entrenando = True
        self.resultado = None
        self.btn_entrenar.config(state="disabled")
        self.msg_hilo = "Entrenando… (puede tardar un par de minutos)"

        def tarea():
            try:
                m, p = entrenar_modelo(5, lambda t: setattr(self, "msg_hilo", t))
                self.resultado = (m, p, None)
            except Exception as ex:
                self.resultado = (None, None, ex)

        threading.Thread(target=tarea, daemon=True).start()
        self.vigilar()

    def vigilar(self):
        if self.resultado is None:
            self.estado.set(self.msg_hilo)
            self.raiz.after(300, self.vigilar)
            return
        modelo, precision, error = self.resultado
        self.entrenando = False
        self.btn_entrenar.config(state="normal")
        if error:
            self.estado.set(f"Error al entrenar: {error}")
        else:
            self.modelo = modelo
            self.estado.set(f"Modelo entrenado. Precisión en prueba: {precision:.2%}")

    def abrir_imagen(self):
        if self.modelo is None:
            messagebox.showinfo("Sin modelo", "Primero carga o entrena el modelo.")
            return
        ruta = filedialog.askopenfilename(
            filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp")])
        if ruta:
            self.predecir_imagen(Image.open(ruta))


if __name__ == "__main__":
    raiz = tk.Tk()
    App(raiz)
    raiz.mainloop()