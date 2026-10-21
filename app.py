"""Ventana principal: une las pestañas, gestiona datos, modelo activo y la cola de eventos."""
import json
import queue
import shutil
import threading
import tkinter as tk
import traceback
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import tensorflow as tf

from config import BG, RUTA_ACTIVO, RUTA_EXPERIMENTOS
from entrenador import Entrenador
from tabs.tab_dataset import TabDataset
from tabs.tab_entrenamiento import TabEntrenamiento
from tabs.tab_evaluacion import TabEvaluacion
from tabs.tab_experimentos import TabExperimentos
from tabs.tab_reconocer import TabReconocer
from ui_utils import aplicar_estilo, etiqueta


class App:
    def __init__(self, raiz):
        self.raiz = raiz
        self.cola = queue.Queue()
        self.parar = threading.Event()
        self.hilo = None
        self.datos = None
        self.modelo = None
        self.barrido = False
        self.mejor = None
        self.experimentos = self.cargar_experimentos()
        self.nombre_modelo = "ninguno"
        self.var_modelo = tk.StringVar(
            value="Modelo activo: ninguno · entrena uno en la pestaña 1 · Entrenar")
        self.var_datos = tk.StringVar(value="MNIST: cargando…")

        raiz.title("Reconocedor de dígitos · Redes Neuronales")
        raiz.geometry("1320x860")
        raiz.minsize(1180, 760)
        raiz.configure(bg=BG)
        aplicar_estilo(raiz)
        self.construir_menu()

        barra = ttk.Frame(raiz, style="Panel.TFrame", padding=(14, 6))
        barra.pack(fill="x", side="bottom")
        etiqueta(barra, textvariable=self.var_modelo).pack(side="left")
        etiqueta(barra, textvariable=self.var_datos, estilo="Muted.TLabel").pack(side="right")

        self.nb = ttk.Notebook(raiz)
        self.nb.pack(fill="both", expand=True)
        self.tab_entrena = TabEntrenamiento(self.nb, self)
        self.tab_dibujo = TabReconocer(self.nb, self)
        self.tab_eval = TabEvaluacion(self.nb, self)
        self.tab_datos = TabDataset(self.nb, self)
        self.tab_exp = TabExperimentos(self.nb, self)
        # Orden pensado como flujo de trabajo: primero entrenar, luego probar.
        for tab, nombre in [(self.tab_entrena, "1 · Entrenar"), (self.tab_dibujo, "2 · Reconocer"),
                            (self.tab_eval, "3 · Evaluar"), (self.tab_datos, "4 · Dataset"),
                            (self.tab_exp, "5 · Experimentos")]:
            self.nb.add(tab, text=nombre)

        raiz.bind("<Control-z>", lambda e: self.si_dibujo(self.tab_dibujo.deshacer))
        raiz.bind("<Control-l>", lambda e: self.si_dibujo(self.tab_dibujo.limpiar))
        raiz.protocol("WM_DELETE_WINDOW", self.cerrar)

        threading.Thread(target=self.cargar_datos, daemon=True).start()
        raiz.after(200, self.cargar_modelo_inicial)
        self.sondear()

    # ----- menú y atajos -----
    def construir_menu(self):
        menu = tk.Menu(self.raiz)
        archivo = tk.Menu(menu, tearoff=0)
        archivo.add_command(label="Cargar modelo…", command=self.cargar_modelo)
        archivo.add_command(label="Guardar modelo activo como…", command=self.guardar_modelo)
        archivo.add_separator()
        archivo.add_command(label="Salir", command=self.cerrar)
        ayuda = tk.Menu(menu, tearoff=0)
        ayuda.add_command(label="Guía rápida", command=lambda: messagebox.showinfo(
            "Guía rápida",
            "1 · Entrenar    Pulsa ▶ Entrenar (los valores ya vienen listos).\n"
            "2 · Reconocer   Dibuja un número y la red lo reconoce al instante.\n"
            "3 · Evaluar     Mide la precisión con las 10 000 imágenes de prueba.\n"
            "4 · Dataset     Explora las imágenes de MNIST.\n"
            "5 · Experimentos  Compara todos tus entrenamientos."))
        ayuda.add_command(label="Atajos", command=lambda: messagebox.showinfo(
            "Atajos", "Ctrl+Z  deshacer el último trazo\nCtrl+L  limpiar el lienzo"))
        ayuda.add_command(label="Acerca de", command=lambda: messagebox.showinfo(
            "Acerca de", "Reconocedor de dígitos MNIST\nProyecto final · Unidad 1 · "
                         "Redes Neuronales\nTensorFlow/Keras + Tkinter"))
        menu.add_cascade(label="Archivo", menu=archivo)
        menu.add_cascade(label="Ayuda", menu=ayuda)
        self.raiz.config(menu=menu)

    def si_dibujo(self, accion):
        if self.nb.select() == str(self.tab_dibujo):
            accion()

    # ----- navegación entre pestañas -----
    def ir_a_entrenar(self):
        self.nb.select(self.tab_entrena)

    def ir_a_reconocer(self):
        self.nb.select(self.tab_dibujo)

    # ----- datos y modelo -----
    def cargar_datos(self):
        try:
            (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.mnist.load_data()
            f = np.float32(255)
            self.cola.put(("datos", ((x_tr.astype("float32")[..., np.newaxis] / f, y_tr),
                                     (x_te.astype("float32")[..., np.newaxis] / f, y_te))))
        except Exception as ex:
            self.cola.put(("datos_error", str(ex)))

    def cargar_modelo_inicial(self):
        if RUTA_ACTIVO.exists():
            try:
                self.activar(tf.keras.models.load_model(RUTA_ACTIVO), RUTA_ACTIVO.name)
                self.ir_a_reconocer()      # ya hay modelo de una sesión anterior: a dibujar
            except Exception as ex:
                self.var_modelo.set(f"No se pudo cargar {RUTA_ACTIVO.name}: {ex}")
        else:
            self.tab_dibujo.actualizar_info("Entrena un modelo en la pestaña 1 · Entrenar.")

    def cargar_modelo(self):
        ruta = filedialog.askopenfilename(filetypes=[("Modelo Keras", "*.keras *.h5")])
        if ruta:
            try:
                self.activar(tf.keras.models.load_model(ruta), Path(ruta).name)
            except Exception as ex:
                messagebox.showerror("Error", f"No se pudo cargar el modelo:\n{ex}")

    def guardar_modelo(self):
        if self.modelo is None:
            return
        ruta = filedialog.asksaveasfilename(defaultextension=".keras",
                                            filetypes=[("Modelo Keras", "*.keras")])
        if ruta:
            self.modelo.save(ruta)

    def activar(self, modelo, etiqueta_modelo, test_acc=None):
        self.modelo = modelo
        self.nombre_modelo = etiqueta_modelo
        texto = f"Modelo activo: {etiqueta_modelo} · {modelo.count_params():,} parámetros"
        if test_acc:
            texto += f" · prueba {test_acc:.2%}"
        self.var_modelo.set(texto)
        self.tab_dibujo.actualizar_info(texto.replace("Modelo activo: ", ""))
        self.tab_dibujo.actualizar_estado()
        self.tab_entrena.actualizar_banner()
        self.tab_dibujo.reconocer()

    def predecir(self, x):
        return self.modelo(x[np.newaxis, ..., np.newaxis], training=False).numpy()[0]

    # ----- experimentos -----
    def cargar_experimentos(self):
        try:
            return json.loads(RUTA_EXPERIMENTOS.read_text(encoding="utf-8"))
        except Exception:
            return []

    def guardar_experimentos(self):
        try:
            RUTA_EXPERIMENTOS.write_text(json.dumps(self.experimentos, ensure_ascii=False),
                                         encoding="utf-8")
        except OSError as ex:
            print("No se pudo guardar experimentos:", ex)

    def registrar(self, res):
        cfg, h = res["cfg"], res["hist"]
        val = [x for x in h["val_accuracy"] if x == x]
        self.experimentos.append({
            "id": max([e["id"] for e in self.experimentos], default=0) + 1,
            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"), "arq": cfg["arq"],
            "opt": cfg["opt"], "lr": cfg["lr"], "sched": cfg["sched"], "epocas": len(h["loss"]),
            "batch": cfg["batch"], "dropout": cfg["dropout"], "l2": cfg["l2"],
            "aug": cfg["aug"], "subset": cfg["subset"], "val_acc": max(val) if val else 0.0,
            "test_acc": res["test_acc"], "test_loss": res["test_loss"],
            "params": res["params"], "tiempo": res["tiempo"], "ruta": str(res["ruta"]),
            "hist": h})
        self.guardar_experimentos()
        self.tab_exp.refrescar()

    # ----- entrenamiento y evaluación -----
    def iniciar(self, configs):
        if self.hilo is not None and self.hilo.is_alive():
            messagebox.showinfo("Entrenando", "Ya hay un entrenamiento en curso.")
            return
        if self.datos is None:
            messagebox.showinfo("MNIST", "Todavía se están cargando los datos de MNIST.")
            return
        self.barrido, self.mejor = len(configs) > 1, None
        self.parar.clear()
        self.tab_entrena.activo(True)
        self.hilo = Entrenador(configs, self.datos, self.cola, self.parar)
        self.hilo.start()

    def detener(self):
        self.parar.set()
        self.tab_entrena.lbl_estado.config(text="Deteniendo…")

    def evaluar_modelo(self):
        if self.modelo is None or self.datos is None:
            messagebox.showinfo("Evaluar", "Primero entrena o carga un modelo "
                                           "(pestaña 1 · Entrenar) y espera a que carguen los datos.")
            return
        modelo, (x_te, y_te) = self.modelo, self.datos[1]
        self.tab_eval.lbl.config(text="Evaluando…")

        def tarea():
            try:
                probs = modelo.predict(x_te, batch_size=512, verbose=0)
                self.cola.put(("eval", {"probs": probs, "y": y_te}))
            except Exception as ex:
                self.cola.put(("error", f"{type(ex).__name__}: {ex}"))

        threading.Thread(target=tarea, daemon=True).start()

    def activar_resultado(self, res):
        shutil.copy(res["ruta"], RUTA_ACTIVO)
        self.activar(res["modelo"], RUTA_ACTIVO.name, res["test_acc"])

    # ----- cola de eventos -----
    def sondear(self):
        try:
            while True:
                tipo, dato = self.cola.get_nowait()
                try:
                    self.despachar(tipo, dato)
                except Exception:
                    traceback.print_exc()
        except queue.Empty:
            pass
        self.raiz.after(100, self.sondear)

    def despachar(self, tipo, dato):
        t = self.tab_entrena
        if tipo == "datos":
            self.datos = dato
            self.var_datos.set(f"MNIST: {len(dato[0][0]):,} entrenamiento · "
                               f"{len(dato[1][0]):,} prueba")
            self.tab_datos.dibujar()
            if self.hilo is None or not self.hilo.is_alive():
                t.activo(False)        # habilita el botón Entrenar
        elif tipo == "datos_error":
            self.var_datos.set("MNIST: error al cargar")
            messagebox.showerror("MNIST", f"No se pudo cargar MNIST (¿hay internet?):\n{dato}")
        elif tipo == "run_ini":
            t.on_run_ini(*dato)
        elif tipo == "batch":
            t.on_batch(*dato)
        elif tipo == "epoca":
            t.on_epoca(dato)
        elif tipo == "fin":
            self.registrar(dato)
            t.on_fin(dato)
            if not self.barrido:
                self.activar_resultado(dato)
            elif self.mejor is None or dato["test_acc"] > self.mejor["test_acc"]:
                self.mejor = dato
        elif tipo == "hilo_fin":
            t.activo(False)
            if self.barrido and self.mejor is not None:
                self.activar_resultado(self.mejor)
                t.log(f"★ Mejor del barrido: {self.mejor['cfg']['arq']} · "
                      f"{self.mejor['cfg']['opt']} · lr={self.mejor['cfg']['lr']:g} → "
                      f"{self.mejor['test_acc']:.2%} (ahora es el modelo activo)")
            self.mejor = None
        elif tipo == "eval":
            self.tab_eval.on_eval(dato)
        elif tipo == "error":
            t.activo(False)
            self.tab_eval.lbl.config(text="Error al evaluar")
            messagebox.showerror("Error", dato)

    def cerrar(self):
        self.parar.set()
        self.raiz.destroy()
