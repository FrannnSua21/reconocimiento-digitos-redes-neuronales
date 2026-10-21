"""Estilo (tema negro elegante) y pequeños constructores de widgets reutilizables."""
import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from config import (ACENTO, ACENTO_HOVER, ACENTO_OFF, BG, BORDE, FUENTE, MUTED, PANEL, PANEL2,
                     TXT, TXT_ACENTO)

BOTON = "#24242b"
BOTON_HOVER = "#2f2f38"


def aplicar_estilo(raiz):
    s = ttk.Style(raiz)
    s.theme_use("clam")
    fuente, negrita = (FUENTE, 10), (FUENTE, 10, "bold")
    s.configure(".", background=BG, foreground=TXT, fieldbackground=PANEL2,
                bordercolor=BORDE, lightcolor=PANEL2, darkcolor=PANEL2,
                troughcolor=PANEL2, font=fuente, focuscolor=PANEL)
    s.configure("TFrame", background=BG)
    s.configure("Panel.TFrame", background=PANEL)
    s.configure("Banner.TFrame", background=PANEL2)
    s.configure("TLabel", background=BG, foreground=TXT)
    s.configure("Panel.TLabel", background=PANEL, foreground=TXT)
    s.configure("Muted.TLabel", background=PANEL, foreground=MUTED)
    s.configure("Titulo.TLabel", background=PANEL, foreground=TXT, font=(FUENTE, 12, "bold"))
    s.configure("Banner.TLabel", background=PANEL2, foreground=MUTED)
    s.configure("BannerTitulo.TLabel", background=PANEL2, foreground=ACENTO,
                font=(FUENTE, 12, "bold"))
    s.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(8, 8, 8, 0))
    s.configure("TNotebook.Tab", background=PANEL, foreground=MUTED,
                padding=(22, 10), font=negrita, borderwidth=0)
    s.map("TNotebook.Tab", background=[("selected", ACENTO), ("active", PANEL2)],
          foreground=[("selected", TXT_ACENTO), ("active", TXT)])
    s.configure("TButton", background=BOTON, foreground=TXT, padding=(12, 7),
                borderwidth=1, bordercolor=BORDE, font=fuente)
    s.map("TButton", background=[("active", BOTON_HOVER), ("disabled", PANEL)],
          foreground=[("disabled", MUTED)])
    s.configure("Acento.TButton", background=ACENTO, foreground=TXT_ACENTO, font=negrita,
                bordercolor=ACENTO)
    s.map("Acento.TButton", background=[("active", ACENTO_HOVER), ("disabled", ACENTO_OFF)],
          foreground=[("disabled", MUTED)])
    s.configure("TLabelframe", background=PANEL, bordercolor=BORDE, borderwidth=1)
    s.configure("TLabelframe.Label", background=PANEL, foreground=ACENTO, font=negrita)
    s.configure("TCheckbutton", background=PANEL, foreground=TXT)
    s.map("TCheckbutton", background=[("active", PANEL)],
          indicatorcolor=[("selected", ACENTO), ("!selected", PANEL2)])
    s.configure("TScale", background=PANEL, troughcolor=BORDE)
    s.configure("Horizontal.TProgressbar", troughcolor=PANEL2, background=ACENTO,
                bordercolor=PANEL, lightcolor=ACENTO, darkcolor=ACENTO, thickness=10)
    s.configure("Treeview", background=PANEL2, fieldbackground=PANEL2, foreground=TXT,
                rowheight=25, borderwidth=0)
    s.configure("Treeview.Heading", background=PANEL, foreground=ACENTO, font=negrita,
                relief="flat", padding=5)
    s.map("Treeview", background=[("selected", ACENTO)],
          foreground=[("selected", TXT_ACENTO)])
    s.map("Treeview.Heading", background=[("active", PANEL)])
    s.configure("TCombobox", arrowcolor=TXT, selectbackground=PANEL2, selectforeground=TXT,
                padding=4)
    s.map("TCombobox", fieldbackground=[("readonly", PANEL2)],
          foreground=[("readonly", TXT)])
    s.configure("TEntry", fieldbackground=PANEL2, foreground=TXT, insertcolor=TXT, padding=4)
    s.configure("TSpinbox", fieldbackground=PANEL2, foreground=TXT, arrowcolor=TXT,
                insertcolor=TXT, padding=3)
    s.configure("TSeparator", background=BORDE)
    s.configure("Vertical.TScrollbar", background=BORDE, troughcolor=PANEL,
                arrowcolor=TXT, bordercolor=PANEL, lightcolor=BORDE, darkcolor=BORDE)
    raiz.option_add("*TCombobox*Listbox.background", PANEL2)
    raiz.option_add("*TCombobox*Listbox.foreground", TXT)
    raiz.option_add("*TCombobox*Listbox.selectBackground", ACENTO)
    raiz.option_add("*TCombobox*Listbox.selectForeground", TXT_ACENTO)
    raiz.option_add("*TCombobox*Listbox.font", fuente)


def etiqueta(padre, texto="", estilo="Panel.TLabel", **kw):
    return ttk.Label(padre, text=texto, style=estilo, **kw)


def combo(padre, var, valores, ancho=20, **kw):
    return ttk.Combobox(padre, textvariable=var, values=valores, width=ancho, **kw)


def deslizador(padre, fila, texto, var, desde, hasta, fmt="{:.2f}"):
    etiqueta(padre, texto).grid(row=fila, column=0, sticky="w", pady=3, padx=(0, 8))
    cont = ttk.Frame(padre, style="Panel.TFrame")
    cont.grid(row=fila, column=1, sticky="ew", pady=3)
    valor = etiqueta(cont, fmt.format(var.get()), "Muted.TLabel", width=5)

    def cambiar(_=None):
        valor.config(text=fmt.format(var.get()))

    ttk.Scale(cont, from_=desde, to=hasta, variable=var, command=cambiar).pack(
        side="left", fill="x", expand=True)
    valor.pack(side="left", padx=(6, 0))


def fila(padre, r, texto, widget):
    etiqueta(padre, texto).grid(row=r, column=0, sticky="w", pady=3, padx=(0, 8))
    widget.grid(row=r, column=1, sticky="ew", pady=3)


def estilo_ax(ax, titulo=""):
    ax.set_facecolor(PANEL2)
    ax.set_title(titulo, color=TXT, fontsize=10, loc="left")
    ax.tick_params(colors=MUTED, labelsize=8)
    for borde in ax.spines.values():
        borde.set_color(BORDE)
    ax.grid(True, color=BORDE, alpha=0.6, linewidth=0.6)


def aviso_ax(ax, texto):
    ax.set_facecolor(PANEL2)
    ax.set_xticks([])
    ax.set_yticks([])
    for borde in ax.spines.values():
        borde.set_color(BORDE)
    ax.text(0.5, 0.5, texto, ha="center", va="center", color=MUTED, fontsize=11,
            transform=ax.transAxes)


def incrustar(fig, padre):
    """Inserta una figura de matplotlib que SIEMPRE se adapta al espacio disponible.

    Se usa place() dentro de un contenedor de tamaño solicitado pequeño: así el gráfico
    no desborda su marco aunque la pantalla tenga escala de Windows > 100 %.
    """
    fig.patch.set_facecolor(PANEL)
    cont = tk.Frame(padre, bg=PANEL, width=200, height=150)
    cont.pack(fill="both", expand=True)
    lienzo = FigureCanvasTkAgg(fig, master=cont)
    lienzo.get_tk_widget().configure(bg=PANEL, highlightthickness=0)
    lienzo.get_tk_widget().place(x=0, y=0, relwidth=1, relheight=1)
    return lienzo


def leyenda(ax, **kw):
    ax.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDE, labelcolor=TXT, **kw)
