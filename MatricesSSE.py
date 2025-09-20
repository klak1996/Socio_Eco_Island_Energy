#DIMENSIONES SOCIOECOLOGICAS 100% DIÉSEL
#%% Instalación de dependencias (solo si no las tienes)
!pip install pandas openpyxl networkx matplotlib seaborn

#%% Importaciones
import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Carpeta de trabajo
WORKDIR = r"C:\Python\MatrizSSE"
os.makedirs(WORKDIR, exist_ok=True)
os.chdir(WORKDIR)

#%% Dimensiones con acrónimos y nombres completos
dimensiones = ["D1","D2","D3","D4","D5","D6","D7"]
nombres = {
    "D1": "Gobernanza y marco institucional",
    "D2": "Económica-productiva",
    "D3": "Social y capital humano",
    "D4": "Ecosistémica-ambiental",
    "D5": "Conectividad y factores externos",
    "D6": "Riesgos y conflictos de uso",
    "D7": "Infraestructura y tecnología energética",
}

#%% Tipos de relación (5)
TIPOS = {
    "+":  {"nombre":"Habilitadora",               "color":"#2ca02c", "style":"solid"},
    "-":  {"nombre":"Restrictiva",                "color":"#d62728", "style":"solid"},
    "↔+": {"nombre":"Refuerzo mutuo",             "color":"#1f77b4", "style":"solid"},
    "↔±": {"nombre":"Compensación mutua",         "color":"#ff7f0e", "style":"solid"},
    "⟂":  {"nombre":"Moduladora (gating/umbral)", "color":"#7f7f7f", "style":"dashed"},
}

#%% Relaciones probables (escenario 100% diésel)
relaciones = [
    ("D1","D2","+"), ("D2","D1","↔+"),
    ("D1","D4","⟂"), ("D1","D7","+"), ("D1","D6","⟂"),
    ("D2","D3","+"), ("D3","D2","+"),
    ("D2","D7","+"), ("D2","D4","↔±"),
    ("D3","D7","+"), ("D3","D6","-"),
    ("D7","D4","-"), ("D7","D6","⟂"),
    ("D4","D5","⟂"), ("D4","D2","↔±"),
    ("D5","D2","-"), ("D5","D7","⟂"), ("D5","D1","+"), ("D5","D6","-"),
    ("D6","D2","-"), ("D6","D3","-"),
]

#%% Crear matriz
matriz = pd.DataFrame("SR", index=dimensiones, columns=dimensiones)
for o, d, t in relaciones:
    matriz.loc[o, d] = t
for d in dimensiones:
    matriz.loc[d, d] = "NO"

matriz.to_excel("matriz_dimensiones_100diesel_simplificada.xlsx")

#%% Crear grafo
G = nx.DiGraph()
G.add_nodes_from(dimensiones)
for o, d, t in relaciones:
    G.add_edge(o, d, tipo=t)

#%% Calcular nodos críticos (clasificación matemática)
betw = nx.betweenness_centrality(G, normalized=True)
out_deg_n = {n: G.out_degree(n)/max(1, max(dict(G.out_degree()).values())) for n in G.nodes()}
in_deg_n  = {n: G.in_degree(n)/max(1, max(dict(G.in_degree()).values())) for n in G.nodes()}
score = {n: (betw[n] + out_deg_n[n] + in_deg_n[n]) / 3 for n in G.nodes()}
nodos_criticos = set(sorted(score, key=score.get, reverse=True)[:max(3, int(0.4*len(G.nodes())))])

#%% Layout
pos = nx.spring_layout(G, seed=42, k=1.2)

#%% Dibujar grafo con flechas visibles, entrada al borde del nodo y líneas más gruesas
plt.figure(figsize=(12, 10))

# Nodos
for nodo in G.nodes():
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=[nodo],
        node_color="white",
        edgecolors="blue" if nodo in nodos_criticos else "black",
        linewidths=3 if nodo in nodos_criticos else 1.5,
        node_size=2300
    )

# Etiquetas: acrónimo (D1, D2...) en el nodo
nx.draw_networkx_labels(G, pos, labels={n: n for n in dimensiones},
                        font_size=10, font_weight="bold")

# Aristas con flechas visibles y grosor aumentado
for (u, v, data) in G.edges(data=True):
    tipo = data["tipo"]
    color = TIPOS[tipo]["color"]
    style = TIPOS[tipo]["style"]

    if tipo in ("↔+", "↔±"):
        # Doble flecha recta (bidireccional)
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], edge_color=color, style=style,
            arrows=True, arrowstyle='-|>', arrowsize=22, width=2.8,
            connectionstyle='arc3,rad=0.0',
            min_source_margin=25, min_target_margin=25
        )
        nx.draw_networkx_edges(
            G, pos, edgelist=[(v, u)], edge_color=color, style=style,
            arrows=True, arrowstyle='-|>', arrowsize=22, width=2.8,
            connectionstyle='arc3,rad=0.0',
            min_source_margin=25, min_target_margin=25
        )
    else:
        # Flecha unidireccional recta
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], edge_color=color, style=style,
            arrows=True, arrowstyle='-|>', arrowsize=22, width=2.8,
            connectionstyle='arc3,rad=0.0',
            min_source_margin=25, min_target_margin=25
        )

# Leyenda de tipos de relación con explicación breve
legend_tipos = [
    Line2D([0], [0], color=TIPOS["+"]["color"], lw=2,
           label=f"+  — {TIPOS['+']['nombre']}: Aumenta o facilita el cambio en la otra variable"),
    Line2D([0], [0], color=TIPOS["-"]["color"], lw=2,
           label=f"-  — {TIPOS['-']['nombre']}: Reduce o limita el cambio en la otra variable"),
    Line2D([0], [0], color=TIPOS["↔+"]["color"], lw=2,
           label=f"↔+ — {TIPOS['↔+']['nombre']}: Ambas variables se refuerzan mutuamente"),
    Line2D([0], [0], color=TIPOS["↔±"]["color"], lw=2,
           label=f"↔± — {TIPOS['↔±']['nombre']}: Relación de equilibrio o trade-off"),
    Line2D([0], [0], color=TIPOS["⟂"]["color"], lw=2, linestyle="--",
           label=f"⟂  — {TIPOS['⟂']['nombre']}: Efecto condicionado por umbral o contexto"),
    Line2D([0], [0], marker='o', color='w', label='Nodo crítico: Alta influencia o vulnerabilidad',
           markerfacecolor='white', markeredgecolor='blue', markersize=14, markeredgewidth=3)
]

# Leyenda de dimensiones (significado de D1...D7)
legend_dims = [
    Line2D([0], [0], color='w', marker='o', markerfacecolor='white',
           label=f"{k}: {v}", markersize=0)
    for k, v in nombres.items()
]

# Unir ambas leyendas y colocarlas abajo
plt.legend(handles=legend_tipos + legend_dims,
           loc='upper center', bbox_to_anchor=(0.5, -0.15),
           ncol=2, fontsize=9, frameon=True)

plt.title("Relaciones entre dimensiones socioecológicas — Escenario 100% diésel")
plt.axis('off')
plt.tight_layout()
plt.savefig("grafo_dimensiones_100diesel.png", dpi=300, bbox_inches="tight")
plt.show()

#%% Análisis socioecológico-matemático del grafo — Escenario 100% diésel

import numpy as np

# --- 1. Métricas de red ---
in_deg = dict(G.in_degree())
out_deg = dict(G.out_degree())
betw = nx.betweenness_centrality(G, normalized=True)
close = nx.closeness_centrality(G)
eig = nx.eigenvector_centrality_numpy(G)

# Normalizar grados para comparabilidad
max_in = max(in_deg.values()) if max(in_deg.values()) > 0 else 1
max_out = max(out_deg.values()) if max(out_deg.values()) > 0 else 1
in_deg_n = {n: in_deg[n] / max_in for n in G.nodes()}
out_deg_n = {n: out_deg[n] / max_out for n in G.nodes()}

# --- 2. Puntajes de resiliencia y vulnerabilidad (ejemplo inicial) ---
# Aquí usamos un esquema simple: resiliencia = (out_deg_n + (1 - in_deg_n)) / 2
# vulnerabilidad = (in_deg_n + (1 - out_deg_n)) / 2
# En la práctica, sustituirías por indicadores reales de cada dimensión.

resiliencia = {}
vulnerabilidad = {}
for n in G.nodes():
    resiliencia[n] = round((out_deg_n[n] + (1 - in_deg_n[n])) / 2 * 5, 2)  # escala 0-5
    vulnerabilidad[n] = round((in_deg_n[n] + (1 - out_deg_n[n])) / 2 * 5, 2)

# --- 3. Criticidad topológica ---
# Score compuesto = promedio de betweenness, in_deg_n y out_deg_n
score = {n: (betw[n] + in_deg_n[n] + out_deg_n[n]) / 3 for n in G.nodes()}
critico_umbral = np.percentile(list(score.values()), 60)  # top 40% como críticos
criticidad = {n: "Alta" if score[n] >= critico_umbral else "Media/Baja" for n in G.nodes()}

# --- 4. Semáforo de estado ---
def semaforo_resiliencia(val):
    if val <= 1.5: return "Rojo"
    elif val <= 3.4: return "Ámbar"
    else: return "Verde"

def semaforo_vulnerabilidad(val):
    if val <= 1.5: return "Verde"
    elif val <= 3.4: return "Ámbar"
    else: return "Rojo"

# --- 5. Construir tabla final ---
tabla_final = pd.DataFrame({
    "Dimensión": [f"{n} - {nombres[n]}" for n in G.nodes()],
    "Grado entrada": [in_deg[n] for n in G.nodes()],
    "Grado salida": [out_deg[n] for n in G.nodes()],
    "Betweenness": [round(betw[n], 3) for n in G.nodes()],
    "Closeness": [round(close[n], 3) for n in G.nodes()],
    "Eigenvector": [round(eig[n], 3) for n in G.nodes()],
    "Resiliencia (0-5)": [resiliencia[n] for n in G.nodes()],
    "Semáforo resiliencia": [semaforo_resiliencia(resiliencia[n]) for n in G.nodes()],
    "Vulnerabilidad (0-5)": [vulnerabilidad[n] for n in G.nodes()],
    "Semáforo vulnerabilidad": [semaforo_vulnerabilidad(vulnerabilidad[n]) for n in G.nodes()],
    "Criticidad topológica": [criticidad[n] for n in G.nodes()]
})

# Ordenar por criticidad y vulnerabilidad descendente
tabla_final = tabla_final.sort_values(by=["Criticidad topológica", "Vulnerabilidad (0-5)"], ascending=[False, False])

# Mostrar tabla
display(tabla_final)

# --- 6. Exportar a Excel ---
with pd.ExcelWriter("analisis_dimensiones_100diesel.xlsx", engine="openpyxl") as writer:
    tabla_final.to_excel(writer, sheet_name="Analisis", index=False)


# VARIABLES SOCIOECOLOGICAS 100% DIÉSEL
#%% Instalación de dependencias (solo si no las tienes)
!pip install pandas openpyxl networkx matplotlib seaborn

#%% Importaciones
import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Carpeta de trabajo
WORKDIR = r"C:\Python\MatrizSSE"
os.makedirs(WORKDIR, exist_ok=True)
os.chdir(WORKDIR)

#%% Definición de variables por dimensión (IDs y etiquetas)
variables = [
    # D1 Gobernanza
    "D1.V1","D1.V2","D1.V3","D1.V4",
    # D2 Económica
    "D2.V1","D2.V2","D2.V3","D2.V4","D2.V5","D2.V6",
    # D3 Social
    "D3.V1","D3.V2","D3.V3","D3.V4","D3.V5",
    # D4 Ecosistémica
    "D4.V1","D4.V2","D4.V3","D4.V4","D4.V5",
    # D5 Conectividad
    "D5.V1","D5.V2","D5.V3","D5.V4",
    # D6 Riesgos/Conflictos
    "D6.V1","D6.V2","D6.V3","D6.V4",
    # D7 Infraestructura energética
    "D7.V1","D7.V2","D7.V3","D7.V4","D7.V5","D7.V6","D7.V7","D7.V8"
]

etiquetas = {
    "D1.V1":"Política energética local y nacional",
    "D1.V2":"Gestión municipal e interinstitucional",
    "D1.V3":"Regulación sectorial (pesca/acuícola/energía)",
    "D1.V4":"Normativa ambiental y sanitaria",
    "D2.V1":"Pesca artesanal y comercial",
    "D2.V2":"Mariscos y recolección",
    "D2.V3":"Acuicultura",
    "D2.V4":"Agricultura familiar y de subsistencia",
    "D2.V5":"Costos y logística energética (producción)",
    "D2.V6":"Turismo local y de naturaleza",
    "D3.V1":"Población y demografía",
    "D3.V2":"Organización comunitaria y cohesión social",
    "D3.V3":"Acceso a servicios esenciales",
    "D3.V4":"Cultura e identidad local",
    "D3.V5":"Capacidades locales generales",
    "D4.V1":"Biodiversidad marina",
    "D4.V2":"Biodiversidad terrestre",
    "D4.V3":"Potencial energético local (REN)",
    "D4.V4":"Calidad ambiental (aire/agua/suelo)",
    "D4.V5":"Amenazas y forzantes climáticos locales",
    "D5.V1":"Mercados y comercialización",
    "D5.V2":"Conectividad física y logística",
    "D5.V3":"Dependencia de insumos externos",
    "D5.V4":"Factores climáticos regionales",
    "D6.V1":"Espacio marino y uso del borde",
    "D6.V2":"Equidad energética",
    "D6.V3":"Operación y continuidad del servicio",
    "D6.V4":"Impactos ambientales de actividades",
    "D7.V1":"Generación diésel",
    "D7.V2":"Generación híbrida (referencia)",
    "D7.V3":"Almacenamiento energético",
    "D7.V4":"Red de distribución eléctrica",
    "D7.V5":"Gestión de cargas y demanda",
    "D7.V6":"Transporte y logística de combustible",
    "D7.V7":"Monitoreo y control del sistema",
    "D7.V8":"Capacidades técnicas locales O&M"
}

dim_de = {vid: vid.split(".")[0] for vid in variables}

nombres_dim = {
    "D1": "Gobernanza y marco institucional",
    "D2": "Económica-productiva",
    "D3": "Social y capital humano",
    "D4": "Ecosistémica-ambiental",
    "D5": "Conectividad y factores externos",
    "D6": "Riesgos y conflictos de uso",
    "D7": "Infraestructura y tecnología energética",
}

PALETA_DIM = {
    "D1": "#8dd3c7", "D2": "#ffffb3", "D3": "#bebada",
    "D4": "#fb8072", "D5": "#80b1d3", "D6": "#fdb462", "D7": "#b3de69",
}

#%% Tipos de relación (5)
TIPOS = {
    "+":  {"nombre":"Habilitadora",               "color":"#2ca02c", "style":"solid"},
    "-":  {"nombre":"Restrictiva",                "color":"#d62728", "style":"solid"},
    "↔+": {"nombre":"Refuerzo mutuo",             "color":"#1f77b4", "style":"solid"},
    "↔±": {"nombre":"Compensación mutua",         "color":"#ff7f0e", "style":"solid"},
    "⟂":  {"nombre":"Moduladora (gating/umbral)", "color":"#7f7f7f", "style":"dashed"},
}

#%% Relaciones probables (escenario 100% diésel)
relaciones = [
    ("D1.V1","D1.V2","+"), ("D1.V1","D2.V5","+"), ("D1.V2","D3.V2","↔+"),
    ("D1.V3","D4.V4","⟂"), ("D1.V4","D6.V4","+"), ("D1.V2","D7.V8","+"),
    ("D1.V1","D7.V4","+"), ("D2.V1","D4.V1","↔±"), ("D2.V3","D4.V1","↔±"),
    ("D2.V2","D4.V1","↔±"), ("D2.V4","D3.V3","+"), ("D2.V5","D7.V6","+"),
    ("D2.V6","D5.V1","↔+"), ("D2.V1","D7.V5","+"), ("D2.V3","D7.V1","+"),
    ("D3.V1","D3.V3","+"), ("D3.V2","D5.V2","↔+"), ("D3.V3","D7.V5","+"),
    ("D3.V5","D7.V8","+"), ("D3.V2","D6.V2","+"), ("D4.V1","D2.V1","⟂"),
    ("D4.V3","D7.V2","+"), ("D4.V4","D3.V3","+"), ("D4.V5","D5.V4","+"),
    ("D4.V5","D5.V2","⟂"), ("D5.V1","D2.V6","+"), ("D5.V2","D7.V6","+"),
    ("D5.V2","D2.V5","+"), ("D5.V3","D7.V1","-"), ("D5.V4","D6.V3","⟂"),
    ("D5.V2","D7.V4","+"), ("D6.V1","D2.V1","↔±"), ("D6.V2","D1.V4","+"),
    ("D6.V3","D2.V5","-"), ("D6.V3","D3.V3","-"), ("D6.V4","D4.V4","-"),
    ("D6.V3","D7.V1","-"), ("D7.V1","D4.V4","-"), ("D7.V3","D6.V3","+"),
    ("D7.V4","D3.V3","+"), ("D7.V5","D2.V5","+"), ("D7.V6","D7.V1","+"),
    ("D7.V7","D6.V3","+"), ("D7.V8","D7.V1","+"), ("D7.V3","D7.V4","+"),
    ("D7.V5","D7.V4","+"), ("D7.V7","D7.V4","+"), ("D2.V1","D2.V5","+"),
    ("D2.V5","D2.V1","-"), ("D7.V4","D7.V7","+"), ("D7.V4","D7.V5","+"),
    ("D3.V2","D3.V5","+"), ("D1.V2","D1.V4","+"), ("D1.V3","D1.V2","+"), 
    # Relaciones añadidas para cubrir variables aisladas
    ("D4.V3","D7.V2","+"),
    ("D7.V2","D7.V1","+"),
    ("D4.V2","D2.V4","↔±"),
    ("D3.V4","D3.V2","+"),
    ("D3.V2","D3.V4","+"),
    ("D5.V3","D2.V5","-"),
    ("D6.V1","D2.V1","↔±")
]


#%% Crear matriz
matriz = pd.DataFrame("SR", index=variables, columns=variables)
for o, d, t in relaciones:
    matriz.loc[o, d] = t
for v in variables:
    matriz.loc[v, v] = "NO"

# Exportar matriz y diccionario
diccionario = pd.DataFrame({
    "ID": variables,
    "Etiqueta": [etiquetas[v] for v in variables],
    "Dimension": [dim_de[v] for v in variables]
})
with pd.ExcelWriter("matriz_variables_100diesel_5tipos.xlsx", engine="openpyxl") as writer:
    matriz.to_excel(writer, sheet_name="Matriz_Variables")
    diccionario.to_excel(writer, sheet_name="Diccionario", index=False)

#%% Crear grafo
G = nx.DiGraph()
for v in variables:
    G.add_node(v, etiqueta=etiquetas[v], dimension=dim_de[v])
for (o, d, t) in relaciones:
    if t in ("↔+", "↔±"):
        G.add_edge(o, d, tipo=t)
        G.add_edge(d, o, tipo=t)
    else:
        G.add_edge(o, d, tipo=t)

#%% Calcular nodos críticos
betw = nx.betweenness_centrality(G, normalized=True)
in_deg = dict(G.in_degree())
out_deg = dict(G.out_degree())
max_in = max(in_deg.values()) if max(in_deg.values()) > 0 else 1
max_out = max(out_deg.values()) if max(out_deg.values()) > 0 else 1
in_deg_n = {n: in_deg[n] / max_in for n in G.nodes()}
out_deg_n = {n: out_deg[n] / max_out for n in G.nodes()}
score = {n: (betw[n] + in_deg_n[n] + out_deg_n[n]) / 3 for n in G.nodes()}
kcrit = max(7, int(0.35 * len(G.nodes())))
nodos_criticos = set(sorted(score, key=score.get, reverse=True)[:kcrit])

#%% Dibujar grafo con flechas visibles (sin clusters forzados) y leyenda de variables a la derecha
plt.figure(figsize=(20, 14))
pos = nx.spring_layout(G, seed=42, k=1.2)

# Nodos
for n in G.nodes():
    dim = dim_de[n]
    nx.draw_networkx_nodes(
        G, pos, nodelist=[n],
        node_color=PALETA_DIM[dim],
        edgecolors="blue" if n in nodos_criticos else "black",
        linewidths=2.6 if n in nodos_criticos else 1.5,
        node_size=1100
    )

# Etiquetas: solo nomenclatura
nx.draw_networkx_labels(G, pos, labels={n: n for n in G.nodes()},
                        font_size=8, font_weight="bold")

# Aristas con flechas
for (u, v, data) in G.edges(data=True):
    t = data["tipo"]
    color = TIPOS[t]["color"]
    style = TIPOS[t]["style"]
    nx.draw_networkx_edges(
        G, pos, edgelist=[(u, v)],
        edge_color=color, style=style,
        arrows=True, arrowstyle='-|>', arrowsize=22, width=2.8,
        connectionstyle='arc3,rad=0.0',
        min_source_margin=18, min_target_margin=18
    )

# Leyendas de tipos y dimensiones (abajo)
legend_tipos = [
    Line2D([0],[0], color=TIPOS["+"]["color"], lw=3, label=f"+ — {TIPOS['+']['nombre']}"),
    Line2D([0],[0], color=TIPOS["-"]["color"], lw=3, label=f"- — {TIPOS['-']['nombre']}"),
    Line2D([0],[0], color=TIPOS["↔+"]["color"], lw=3, label=f"↔+ — {TIPOS['↔+']['nombre']}"),
    Line2D([0],[0], color=TIPOS["↔±"]["color"], lw=3, label=f"↔± — {TIPOS['↔±']['nombre']}"),
    Line2D([0],[0], color=TIPOS["⟂"]["color"], lw=3, linestyle="--", label=f"⟂ — {TIPOS['⟂']['nombre']}"),
    Line2D([0],[0], marker='o', color='w', label='Nodo crítico (borde azul)',
           markerfacecolor='white', markeredgecolor='blue', markersize=10, markeredgewidth=3),
]
legend_dims = [
    Line2D([0],[0], marker='o', color='w', label=f"{d}: {nombres_dim[d]}",
           markerfacecolor=PALETA_DIM[d], markeredgecolor='black', markersize=10)
    for d in nombres_dim
]

plt.legend(handles=legend_tipos + legend_dims,
           loc='upper center', bbox_to_anchor=(0.5, -0.12),
           ncol=2, fontsize=9, frameon=True)

# Leyenda de variables (a la derecha)
leyenda_vars = "\n".join([f"{vid}: {etiquetas[vid]}" for vid in sorted(G.nodes())])
plt.gcf().text(0.82, 0.5, leyenda_vars, fontsize=7, va='center', ha='left')

plt.title("Relaciones entre variables socioecológicas — Escenario 100% diésel")
plt.axis('off')
plt.tight_layout(rect=[0, 0, 0.8, 1])  # deja espacio a la derecha para la leyenda
plt.savefig("grafo_variables_100diesel.png", dpi=300, bbox_inches="tight")
plt.show()

#%% Análisis socioecológico-matemático
close = nx.closeness_centrality(G)
eig = nx.eigenvector_centrality(G, max_iter=1000)
resil = {n: round((out_deg_n[n] + (1 - in_deg_n[n])) / 2 * 5, 2) for n in G.nodes()}
vuln = {n: round((in_deg_n[n] + (1 - out_deg_n[n])) / 2 * 5, 2) for n in G.nodes()}

def semaforo_res(v): return "Rojo" if v <= 1.5 else ("Ámbar" if v <= 3.4 else "Verde")
def semaforo_vul(v): return "Verde" if v <= 1.5 else ("Ámbar" if v <= 3.4 else "Rojo")

tabla_vars = pd.DataFrame({
    "ID": list(G.nodes()),
    "Dimensión": [dim_de[n] for n in G.nodes()],
    "Nombre": [etiquetas[n] for n in G.nodes()],
    "Grado entrada": [in_deg[n] for n in G.nodes()],
    "Grado salida": [out_deg[n] for n in G.nodes()],
    "Betweenness": [round(betw[n], 4) for n in G.nodes()],
    "Closeness": [round(close[n], 4) for n in G.nodes()],
    "Eigenvector": [round(eig[n], 4) for n in G.nodes()],
    "Resiliencia (0-5)": [resil[n] for n in G.nodes()],
    "Semáforo resiliencia": [semaforo_res(resil[n]) for n in G.nodes()],
    "Vulnerabilidad (0-5)": [vuln[n] for n in G.nodes()],
    "Semáforo vulnerabilidad": [semaforo_vul(vuln[n]) for n in G.nodes()],
    "Crítico": [n in nodos_criticos for n in G.nodes()],
    "Score topológico": [round(score[n], 4) for n in G.nodes()],
}).sort_values(["Crítico","Vulnerabilidad (0-5)","Score topológico"], ascending=[False, False, False])

# Resumen por dimensión
resumen_dim = tabla_vars.groupby("Dimensión").agg({
    "Grado entrada":"mean",
    "Grado salida":"mean",
    "Betweenness":"mean",
    "Closeness":"mean",
    "Eigenvector":"mean",
    "Resiliencia (0-5)":"mean",
    "Vulnerabilidad (0-5)":"mean",
    "Score topológico":"mean"
}).reset_index()
resumen_dim["Nombre dimensión"] = resumen_dim["Dimensión"].map(nombres_dim)
resumen_dim = resumen_dim[["Dimensión","Nombre dimensión"] + [c for c in resumen_dim.columns if c not in ("Dimensión","Nombre dimensión")]]

#%% Exportar análisis
with pd.ExcelWriter("analisis_variables_100diesel.xlsx", engine="openpyxl") as writer:
    matriz.to_excel(writer, sheet_name="Matriz_5Tipos")
    diccionario.to_excel(writer, sheet_name="Diccionario", index=False)
    tabla_vars.to_excel(writer, sheet_name="Analisis_Variables", index=False)
    resumen_dim.to_excel(writer, sheet_name="Resumen_Dimension", index=False)

print("Listo. Archivos generados en:", WORKDIR)

#DIMENSIONES SOCIOECOLOGICAS 80% SOLAR + 20% DIÉSEL
#%% Instalación de dependencias (solo si no las tienes)
!pip install pandas openpyxl networkx matplotlib seaborn

#%% Importaciones
import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Carpeta de trabajo
WORKDIR = r"C:\Python\MatrizSSE"
os.makedirs(WORKDIR, exist_ok=True)
os.chdir(WORKDIR)

#%% Dimensiones con acrónimos y nombres completos
dimensiones = ["D1","D2","D3","D4","D5","D6","D7"]
nombres = {
    "D1": "Gobernanza y marco institucional",
    "D2": "Económica-productiva",
    "D3": "Social y capital humano",
    "D4": "Ecosistémica-ambiental",
    "D5": "Conectividad y factores externos",
    "D6": "Riesgos y conflictos de uso",
    "D7": "Infraestructura y tecnología energética",
}

#%% Tipos de relación (5)
TIPOS = {
    "+":  {"nombre":"Habilitadora",               "color":"#2ca02c", "style":"solid"},
    "-":  {"nombre":"Restrictiva",                "color":"#d62728", "style":"solid"},
    "↔+": {"nombre":"Refuerzo mutuo",             "color":"#1f77b4", "style":"solid"},
    "↔±": {"nombre":"Compensación mutua",         "color":"#ff7f0e", "style":"solid"},
    "⟂":  {"nombre":"Moduladora (gating/umbral)", "color":"#7f7f7f", "style":"dashed"},
}

#%% Relaciones probables (escenario 80% solar + 20% diésel) — ¡MODIFICADO!
relaciones = [
    ("D1","D2","+"), ("D2","D1","↔+"),
    ("D1","D4","+"), ("D1","D7","+"), ("D1","D6","⟂"),
    ("D2","D3","+"), ("D3","D2","+"),
    ("D2","D7","+"), ("D2","D4","↔+"),
    ("D3","D7","+"), ("D3","D6","-"),
    ("D7","D4","+"), ("D7","D6","-"),
    ("D4","D5","⟂"), ("D4","D2","↔+"),
    ("D5","D2","-"), ("D5","D7","-"), ("D5","D1","+"), ("D5","D6","-"),
    ("D6","D2","-"), ("D6","D3","-"),
]

#%% Crear matriz
matriz = pd.DataFrame("SR", index=dimensiones, columns=dimensiones)
for o, d, t in relaciones:
    matriz.loc[o, d] = t
for d in dimensiones:
    matriz.loc[d, d] = "NO"

# Exportar matriz — ¡NOMBRE DE ARCHIVO MODIFICADO!
matriz.to_excel("matriz_dimensiones_80solar20diesel_simplificada.xlsx")

#%% Crear grafo
G = nx.DiGraph()
G.add_nodes_from(dimensiones)
for o, d, t in relaciones:
    G.add_edge(o, d, tipo=t)

#%% Calcular nodos críticos (clasificación matemática)
betw = nx.betweenness_centrality(G, normalized=True)
out_deg_n = {n: G.out_degree(n)/max(1, max(dict(G.out_degree()).values())) for n in G.nodes()}
in_deg_n  = {n: G.in_degree(n)/max(1, max(dict(G.in_degree()).values())) for n in G.nodes()}
score = {n: (betw[n] + out_deg_n[n] + in_deg_n[n]) / 3 for n in G.nodes()}
nodos_criticos = set(sorted(score, key=score.get, reverse=True)[:max(3, int(0.4*len(G.nodes())))])

#%% Layout
pos = nx.spring_layout(G, seed=42, k=1.2)

#%% Dibujar grafo con flechas visibles, entrada al borde del nodo y líneas más gruesas
plt.figure(figsize=(12, 10))

# Nodos
for nodo in G.nodes():
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=[nodo],
        node_color="white",
        edgecolors="blue" if nodo in nodos_criticos else "black",
        linewidths=3 if nodo in nodos_criticos else 1.5,
        node_size=2300
    )

# Etiquetas: acrónimo (D1, D2...) en el nodo
nx.draw_networkx_labels(G, pos, labels={n: n for n in dimensiones},
                        font_size=10, font_weight="bold")

# Aristas con flechas visibles y grosor aumentado
for (u, v, data) in G.edges(data=True):
    tipo = data["tipo"]
    color = TIPOS[tipo]["color"]
    style = TIPOS[tipo]["style"]

    if tipo in ("↔+", "↔±"):
        # Doble flecha recta (bidireccional)
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], edge_color=color, style=style,
            arrows=True, arrowstyle='-|>', arrowsize=22, width=2.8,
            connectionstyle='arc3,rad=0.0',
            min_source_margin=25, min_target_margin=25
        )
        nx.draw_networkx_edges(
            G, pos, edgelist=[(v, u)], edge_color=color, style=style,
            arrows=True, arrowstyle='-|>', arrowsize=22, width=2.8,
            connectionstyle='arc3,rad=0.0',
            min_source_margin=25, min_target_margin=25
        )
    else:
        # Flecha unidireccional recta
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], edge_color=color, style=style,
            arrows=True, arrowstyle='-|>', arrowsize=22, width=2.8,
            connectionstyle='arc3,rad=0.0',
            min_source_margin=25, min_target_margin=25
        )

# Leyenda de tipos de relación con explicación breve
legend_tipos = [
    Line2D([0], [0], color=TIPOS["+"]["color"], lw=2,
           label=f"+  — {TIPOS['+']['nombre']}: Aumenta o facilita el cambio en la otra variable"),
    Line2D([0], [0], color=TIPOS["-"]["color"], lw=2,
           label=f"-  — {TIPOS['-']['nombre']}: Reduce o limita el cambio en la otra variable"),
    Line2D([0], [0], color=TIPOS["↔+"]["color"], lw=2,
           label=f"↔+ — {TIPOS['↔+']['nombre']}: Ambas variables se refuerzan mutuamente"),
    Line2D([0], [0], color=TIPOS["↔±"]["color"], lw=2,
           label=f"↔± — {TIPOS['↔±']['nombre']}: Relación de equilibrio o trade-off"),
    Line2D([0], [0], color=TIPOS["⟂"]["color"], lw=2, linestyle="--",
           label=f"⟂  — {TIPOS['⟂']['nombre']}: Efecto condicionado por umbral o contexto"),
    Line2D([0], [0], marker='o', color='w', label='Nodo crítico: Alta influencia o vulnerabilidad',
           markerfacecolor='white', markeredgecolor='blue', markersize=14, markeredgewidth=3)
]

# Leyenda de dimensiones (significado de D1...D7)
legend_dims = [
    Line2D([0], [0], color='w', marker='o', markerfacecolor='white',
           label=f"{k}: {v}", markersize=0)
    for k, v in nombres.items()
]

# Unir ambas leyendas y colocarlas abajo
plt.legend(handles=legend_tipos + legend_dims,
           loc='upper center', bbox_to_anchor=(0.5, -0.15),
           ncol=2, fontsize=9, frameon=True)

plt.title("Relaciones entre dimensiones socioecológicas — Escenario 80% solar + 20% diésel")
plt.axis('off')
plt.tight_layout()
# Exportar grafo — ¡NOMBRE DE ARCHIVO MODIFICADO!
plt.savefig("grafo_dimensiones_80solar20diesel.png", dpi=300, bbox_inches="tight")
plt.show()

#%% Análisis socioecológico-matemático del grafo — Escenario 80% solar + 20% diésel

import numpy as np

# --- 1. Métricas de red ---
in_deg = dict(G.in_degree())
out_deg = dict(G.out_degree())
betw = nx.betweenness_centrality(G, normalized=True)
close = nx.closeness_centrality(G)
eig = nx.eigenvector_centrality_numpy(G)

# Normalizar grados para comparabilidad
max_in = max(in_deg.values()) if max(in_deg.values()) > 0 else 1
max_out = max(out_deg.values()) if max(out_deg.values()) > 0 else 1
in_deg_n = {n: in_deg[n] / max_in for n in G.nodes()}
out_deg_n = {n: out_deg[n] / max_out for n in G.nodes()}

# --- 2. Puntajes de resiliencia y vulnerabilidad (ejemplo inicial) ---
resiliencia = {}
vulnerabilidad = {}
for n in G.nodes():
    resiliencia[n] = round((out_deg_n[n] + (1 - in_deg_n[n])) / 2 * 5, 2)  # escala 0-5
    vulnerabilidad[n] = round((in_deg_n[n] + (1 - out_deg_n[n])) / 2 * 5, 2)

# --- 3. Criticidad topológica ---
score = {n: (betw[n] + in_deg_n[n] + out_deg_n[n]) / 3 for n in G.nodes()}
critico_umbral = np.percentile(list(score.values()), 60)  # top 40% como críticos
criticidad = {n: "Alta" if score[n] >= critico_umbral else "Media/Baja" for n in G.nodes()}

# --- 4. Semáforo de estado ---
def semaforo_resiliencia(val):
    if val <= 1.5: return "Rojo"
    elif val <= 3.4: return "Ámbar"
    else: return "Verde"

def semaforo_vulnerabilidad(val):
    if val <= 1.5: return "Verde"
    elif val <= 3.4: return "Ámbar"
    else: return "Rojo"

# --- 5. Construir tabla final ---
tabla_final = pd.DataFrame({
    "Dimensión": [f"{n} - {nombres[n]}" for n in G.nodes()],
    "Grado entrada": [in_deg[n] for n in G.nodes()],
    "Grado salida": [out_deg[n] for n in G.nodes()],
    "Betweenness": [round(betw[n], 3) for n in G.nodes()],
    "Closeness": [round(close[n], 3) for n in G.nodes()],
    "Eigenvector": [round(eig[n], 3) for n in G.nodes()],
    "Resiliencia (0-5)": [resiliencia[n] for n in G.nodes()],
    "Semáforo resiliencia": [semaforo_resiliencia(resiliencia[n]) for n in G.nodes()],
    "Vulnerabilidad (0-5)": [vulnerabilidad[n] for n in G.nodes()],
    "Semáforo vulnerabilidad": [semaforo_vulnerabilidad(vulnerabilidad[n]) for n in G.nodes()],
    "Criticidad topológica": [criticidad[n] for n in G.nodes()]
})

# Ordenar por criticidad y vulnerabilidad descendente
tabla_final = tabla_final.sort_values(by=["Criticidad topológica", "Vulnerabilidad (0-5)"], ascending=[False, False])

# Mostrar tabla
display(tabla_final)

# --- 6. Exportar a Excel — ¡NOMBRE DE ARCHIVO MODIFICADO!
with pd.ExcelWriter("analisis_dimensiones_80solar20diesel.xlsx", engine="openpyxl") as writer:
    tabla_final.to_excel(writer, sheet_name="Analisis", index=False)


# VARIABLES SOCIOECOLOGICAS 80% SOLAR + 20% DIÉSEL
#%% Instalación de dependencias (solo si no las tienes)
!pip install pandas openpyxl networkx matplotlib seaborn

#%% Importaciones
import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Carpeta de trabajo
WORKDIR = r"C:\Python\MatrizSSE"
os.makedirs(WORKDIR, exist_ok=True)
os.chdir(WORKDIR)

#%% Definición de variables por dimensión (IDs y etiquetas)
variables = [
    # D1 Gobernanza
    "D1.V1","D1.V2","D1.V3","D1.V4",
    # D2 Económica
    "D2.V1","D2.V2","D2.V3","D2.V4","D2.V5","D2.V6",
    # D3 Social
    "D3.V1","D3.V2","D3.V3","D3.V4","D3.V5",
    # D4 Ecosistémica
    "D4.V1","D4.V2","D4.V3","D4.V4","D4.V5",
    # D5 Conectividad
    "D5.V1","D5.V2","D5.V3","D5.V4",
    # D6 Riesgos/Conflictos
    "D6.V1","D6.V2","D6.V3","D6.V4",
    # D7 Infraestructura energética
    "D7.V1","D7.V2","D7.V3","D7.V4","D7.V5","D7.V6","D7.V7","D7.V8"
]

etiquetas = {
    "D1.V1":"Política energética local y nacional",
    "D1.V2":"Gestión municipal e interinstitucional",
    "D1.V3":"Regulación sectorial (pesca/acuícola/energía)",
    "D1.V4":"Normativa ambiental y sanitaria",
    "D2.V1":"Pesca artesanal y comercial",
    "D2.V2":"Mariscos y recolección",
    "D2.V3":"Acuicultura",
    "D2.V4":"Agricultura familiar y de subsistencia",
    "D2.V5":"Costos y logística energética (producción)",
    "D2.V6":"Turismo local y de naturaleza",
    "D3.V1":"Población y demografía",
    "D3.V2":"Organización comunitaria y cohesión social",
    "D3.V3":"Acceso a servicios esenciales",
    "D3.V4":"Cultura e identidad local",
    "D3.V5":"Capacidades locales generales",
    "D4.V1":"Biodiversidad marina",
    "D4.V2":"Biodiversidad terrestre",
    "D4.V3":"Potencial energético local (REN)",
    "D4.V4":"Calidad ambiental (aire/agua/suelo)",
    "D4.V5":"Amenazas y forzantes climáticos locales",
    "D5.V1":"Mercados y comercialización",
    "D5.V2":"Conectividad física y logística",
    "D5.V3":"Dependencia de insumos externos",
    "D5.V4":"Factores climáticos regionales",
    "D6.V1":"Espacio marino y uso del borde",
    "D6.V2":"Equidad energética",
    "D6.V3":"Operación y continuidad del servicio",
    "D6.V4":"Impactos ambientales de actividades",
    "D7.V1":"Generación diésel",
    "D7.V2":"Generación híbrida (referencia)",
    "D7.V3":"Almacenamiento energético",
    "D7.V4":"Red de distribución eléctrica",
    "D7.V5":"Gestión de cargas y demanda",
    "D7.V6":"Transporte y logística de combustible",
    "D7.V7":"Monitoreo y control del sistema",
    "D7.V8":"Capacidades técnicas locales O&M"
}

dim_de = {vid: vid.split(".")[0] for vid in variables}

nombres_dim = {
    "D1": "Gobernanza y marco institucional",
    "D2": "Económica-productiva",
    "D3": "Social y capital humano",
    "D4": "Ecosistémica-ambiental",
    "D5": "Conectividad y factores externos",
    "D6": "Riesgos y conflictos de uso",
    "D7": "Infraestructura y tecnología energética",
}

PALETA_DIM = {
    "D1": "#8dd3c7", "D2": "#ffffb3", "D3": "#bebada",
    "D4": "#fb8072", "D5": "#80b1d3", "D6": "#fdb462", "D7": "#b3de69",
}

#%% Tipos de relación (5)
TIPOS = {
    "+":  {"nombre":"Habilitadora",               "color":"#2ca02c", "style":"solid"},
    "-":  {"nombre":"Restrictiva",                "color":"#d62728", "style":"solid"},
    "↔+": {"nombre":"Refuerzo mutuo",             "color":"#1f77b4", "style":"solid"},
    "↔±": {"nombre":"Compensación mutua",         "color":"#ff7f0e", "style":"solid"},
    "⟂":  {"nombre":"Moduladora (gating/umbral)", "color":"#7f7f7f", "style":"dashed"},
}

#%% Relaciones probables (escenario 80% solar + 20% diésel) — ¡MODIFICADO!
relaciones = [
    ("D1.V1","D1.V2","+"), ("D1.V1","D2.V5","+"), ("D1.V2","D3.V2","↔+"),
    ("D1.V3","D4.V4","+"), ("D1.V4","D6.V4","+"), ("D1.V2","D7.V8","+"),
    ("D1.V1","D7.V4","+"), ("D2.V1","D4.V1","↔+"), ("D2.V3","D4.V1","↔+"),
    ("D2.V2","D4.V1","↔+"), ("D2.V4","D3.V3","+"), ("D2.V5","D7.V6","+"),
    ("D2.V6","D5.V1","↔+"), ("D2.V1","D7.V5","+"), ("D2.V3","D7.V1","+"),
    ("D3.V1","D3.V3","+"), ("D3.V2","D5.V2","↔+"), ("D3.V3","D7.V5","+"),
    ("D3.V5","D7.V8","+"), ("D3.V2","D6.V2","+"), ("D4.V1","D2.V1","⟂"),
    ("D4.V3","D7.V2","+"), ("D4.V4","D3.V3","+"), ("D4.V5","D5.V4","+"),
    ("D4.V5","D5.V2","⟂"), ("D5.V1","D2.V6","+"), ("D5.V2","D7.V6","+"),
    ("D5.V2","D2.V5","+"), ("D5.V3","D7.V1","-"), ("D5.V4","D6.V3","⟂"),
    ("D5.V2","D7.V4","+"), ("D6.V1","D2.V1","↔±"), ("D6.V2","D1.V4","+"),
    ("D6.V3","D2.V5","-"), ("D6.V3","D3.V3","-"), ("D6.V4","D4.V4","-"),
    ("D6.V3","D7.V1","-"), ("D7.V1","D4.V4","+"), ("D7.V3","D6.V3","+"),
    ("D7.V4","D3.V3","+"), ("D7.V5","D2.V5","+"), ("D7.V6","D7.V1","+"),
    ("D7.V7","D6.V3","+"), ("D7.V8","D7.V1","+"), ("D7.V3","D7.V4","+"),
    ("D7.V5","D7.V4","+"), ("D7.V7","D7.V4","+"), ("D2.V1","D2.V5","+"),
    ("D2.V5","D2.V1","-"), ("D7.V4","D7.V7","+"), ("D7.V4","D7.V5","+"),
    ("D3.V2","D3.V5","+"), ("D1.V2","D1.V4","+"), ("D1.V3","D1.V2","+"), 
    # Relaciones añadidas o modificadas
    ("D4.V3","D7.V2","+"),
    ("D7.V2","D7.V1","+"),
    ("D4.V2","D2.V4","↔+"),
    ("D3.V4","D3.V2","+"),
    ("D3.V2","D3.V4","+"),
    ("D5.V3","D2.V5","-"),
    ("D6.V1","D2.V1","↔±"),
    # NUEVAS RELACIONES CLAVE
    ("D7.V2","D4.V4","+"),
    ("D7.V2","D6.V3","-"),
    ("D7.V3","D4.V4","+"),
    ("D1.V1","D7.V2","+"),
    ("D3.V5","D7.V2","+"),
    ("D5.V3","D7.V2","-"),
    ("D7.V8","D6.V2","+"),
    ("D7.V2","D2.V5","-"),
    ("D7.V2","D7.V6","-"),
   
]

#%% Crear matriz
matriz = pd.DataFrame("SR", index=variables, columns=variables)
for o, d, t in relaciones:
    matriz.loc[o, d] = t
for v in variables:
    matriz.loc[v, v] = "NO"

# Exportar matriz y diccionario — ¡NOMBRE DE ARCHIVO MODIFICADO!
diccionario = pd.DataFrame({
    "ID": variables,
    "Etiqueta": [etiquetas[v] for v in variables],
    "Dimension": [dim_de[v] for v in variables]
})
with pd.ExcelWriter("matriz_variables_80solar20diesel_5tipos.xlsx", engine="openpyxl") as writer:
    matriz.to_excel(writer, sheet_name="Matriz_Variables")
    diccionario.to_excel(writer, sheet_name="Diccionario", index=False)

#%% Crear grafo
G = nx.DiGraph()
for v in variables:
    G.add_node(v, etiqueta=etiquetas[v], dimension=dim_de[v])
for (o, d, t) in relaciones:
    if t in ("↔+", "↔±"):
        G.add_edge(o, d, tipo=t)
        G.add_edge(d, o, tipo=t)
    else:
        G.add_edge(o, d, tipo=t)

#%% Calcular nodos críticos
betw = nx.betweenness_centrality(G, normalized=True)
in_deg = dict(G.in_degree())
out_deg = dict(G.out_degree())
max_in = max(in_deg.values()) if max(in_deg.values()) > 0 else 1
max_out = max(out_deg.values()) if max(out_deg.values()) > 0 else 1
in_deg_n = {n: in_deg[n] / max_in for n in G.nodes()}
out_deg_n = {n: out_deg[n] / max_out for n in G.nodes()}
score = {n: (betw[n] + in_deg_n[n] + out_deg_n[n]) / 3 for n in G.nodes()}
kcrit = max(7, int(0.35 * len(G.nodes())))
nodos_criticos = set(sorted(score, key=score.get, reverse=True)[:kcrit])

#%% Dibujar grafo con flechas visibles (sin clusters forzados) y leyenda de variables a la derecha
plt.figure(figsize=(20, 14))
pos = nx.spring_layout(G, seed=42, k=1.2)

# Nodos
for n in G.nodes():
    dim = dim_de[n]
    nx.draw_networkx_nodes(
        G, pos, nodelist=[n],
        node_color=PALETA_DIM[dim],
        edgecolors="blue" if n in nodos_criticos else "black",
        linewidths=2.6 if n in nodos_criticos else 1.5,
        node_size=1100
    )

# Etiquetas: solo nomenclatura
nx.draw_networkx_labels(G, pos, labels={n: n for n in G.nodes()},
                        font_size=8, font_weight="bold")

# Aristas con flechas
for (u, v, data) in G.edges(data=True):
    t = data["tipo"]
    color = TIPOS[t]["color"]
    style = TIPOS[t]["style"]
    nx.draw_networkx_edges(
        G, pos, edgelist=[(u, v)],
        edge_color=color, style=style,
        arrows=True, arrowstyle='-|>', arrowsize=22, width=2.8,
        connectionstyle='arc3,rad=0.0',
        min_source_margin=18, min_target_margin=18
    )

# Leyendas de tipos y dimensiones (abajo)
legend_tipos = [
    Line2D([0],[0], color=TIPOS["+"]["color"], lw=3, label=f"+ — {TIPOS['+']['nombre']}"),
    Line2D([0],[0], color=TIPOS["-"]["color"], lw=3, label=f"- — {TIPOS['-']['nombre']}"),
    Line2D([0],[0], color=TIPOS["↔+"]["color"], lw=3, label=f"↔+ — {TIPOS['↔+']['nombre']}"),
    Line2D([0],[0], color=TIPOS["↔±"]["color"], lw=3, label=f"↔± — {TIPOS['↔±']['nombre']}"),
    Line2D([0],[0], color=TIPOS["⟂"]["color"], lw=3, linestyle="--", label=f"⟂ — {TIPOS['⟂']['nombre']}"),
    Line2D([0],[0], marker='o', color='w', label='Nodo crítico (borde azul)',
           markerfacecolor='white', markeredgecolor='blue', markersize=10, markeredgewidth=3),
]
legend_dims = [
    Line2D([0],[0], marker='o', color='w', label=f"{d}: {nombres_dim[d]}",
           markerfacecolor=PALETA_DIM[d], markeredgecolor='black', markersize=10)
    for d in nombres_dim
]

plt.legend(handles=legend_tipos + legend_dims,
           loc='upper center', bbox_to_anchor=(0.5, -0.12),
           ncol=2, fontsize=9, frameon=True)

# Leyenda de variables (a la derecha)
leyenda_vars = "\n".join([f"{vid}: {etiquetas[vid]}" for vid in sorted(G.nodes())])
plt.gcf().text(0.82, 0.5, leyenda_vars, fontsize=7, va='center', ha='left')

plt.title("Relaciones entre variables socioecológicas — Escenario 80% solar + 20% diésel")
plt.axis('off')
plt.tight_layout(rect=[0, 0, 0.8, 1])  # deja espacio a la derecha para la leyenda
# Exportar grafo — ¡NOMBRE DE ARCHIVO MODIFICADO!
plt.savefig("grafo_variables_80solar20diesel.png", dpi=300, bbox_inches="tight")
plt.show()

#%% Análisis socioecológico-matemático
close = nx.closeness_centrality(G)
eig = nx.eigenvector_centrality(G, max_iter=1000)
resil = {n: round((out_deg_n[n] + (1 - in_deg_n[n])) / 2 * 5, 2) for n in G.nodes()}
vuln = {n: round((in_deg_n[n] + (1 - out_deg_n[n])) / 2 * 5, 2) for n in G.nodes()}

def semaforo_res(v): return "Rojo" if v <= 1.5 else ("Ámbar" if v <= 3.4 else "Verde")
def semaforo_vul(v): return "Verde" if v <= 1.5 else ("Ámbar" if v <= 3.4 else "Rojo")

tabla_vars = pd.DataFrame({
    "ID": list(G.nodes()),
    "Dimensión": [dim_de[n] for n in G.nodes()],
    "Nombre": [etiquetas[n] for n in G.nodes()],
    "Grado entrada": [in_deg[n] for n in G.nodes()],
    "Grado salida": [out_deg[n] for n in G.nodes()],
    "Betweenness": [round(betw[n], 4) for n in G.nodes()],
    "Closeness": [round(close[n], 4) for n in G.nodes()],
    "Eigenvector": [round(eig[n], 4) for n in G.nodes()],
    "Resiliencia (0-5)": [resil[n] for n in G.nodes()],
    "Semáforo resiliencia": [semaforo_res(resil[n]) for n in G.nodes()],
    "Vulnerabilidad (0-5)": [vuln[n] for n in G.nodes()],
    "Semáforo vulnerabilidad": [semaforo_vul(vuln[n]) for n in G.nodes()],
    "Crítico": [n in nodos_criticos for n in G.nodes()],
    "Score topológico": [round(score[n], 4) for n in G.nodes()],
}).sort_values(["Crítico","Vulnerabilidad (0-5)","Score topológico"], ascending=[False, False, False])

# Resumen por dimensión
resumen_dim = tabla_vars.groupby("Dimensión").agg({
    "Grado entrada":"mean",
    "Grado salida":"mean",
    "Betweenness":"mean",
    "Closeness":"mean",
    "Eigenvector":"mean",
    "Resiliencia (0-5)":"mean",
    "Vulnerabilidad (0-5)":"mean",
    "Score topológico":"mean"
}).reset_index()
resumen_dim["Nombre dimensión"] = resumen_dim["Dimensión"].map(nombres_dim)
resumen_dim = resumen_dim[["Dimensión","Nombre dimensión"] + [c for c in resumen_dim.columns if c not in ("Dimensión","Nombre dimensión")]]

#%% Exportar análisis — ¡NOMBRE DE ARCHIVO MODIFICADO!
with pd.ExcelWriter("analisis_variables_80solar20diesel.xlsx", engine="openpyxl") as writer:
    matriz.to_excel(writer, sheet_name="Matriz_5Tipos")
    diccionario.to_excel(writer, sheet_name="Diccionario", index=False)
    tabla_vars.to_excel(writer, sheet_name="Analisis_Variables", index=False)
    resumen_dim.to_excel(writer, sheet_name="Resumen_Dimension", index=False)

print("Listo. Archivos generados en:", WORKDIR)

#%% Comparativa de escenarios socioecológicos
import pandas as pd

# Archivos de entrada
file_100 = "analisis_variables_100diesel.xlsx"
file_8020 = "analisis_variables_80solar20diesel.xlsx"

# Leer hojas de métricas por variable
vars_100 = pd.read_excel(file_100, sheet_name="Analisis_Variables")
vars_8020 = pd.read_excel(file_8020, sheet_name="Analisis_Variables")

# Leer hojas de métricas por dimensión
dim_100 = pd.read_excel(file_100, sheet_name="Resumen_Dimension")
dim_8020 = pd.read_excel(file_8020, sheet_name="Resumen_Dimension")

# --- Comparativa por variable ---
comparativa_vars = vars_100.merge(vars_8020, on="ID", suffixes=("_100diesel", "_80solar20diesel"))

# Calcular diferencias y cambios porcentuales en métricas clave
for metrica in ["Resiliencia (0-5)", "Vulnerabilidad (0-5)", "Betweenness", "Score topológico"]:
    comparativa_vars[f"Δ_{metrica}"] = comparativa_vars[f"{metrica}_80solar20diesel"] - comparativa_vars[f"{metrica}_100diesel"]
    comparativa_vars[f"%Δ_{metrica}"] = 100 * comparativa_vars[f"Δ_{metrica}"] / comparativa_vars[f"{metrica}_100diesel"].replace(0, pd.NA)

# Cambios de semáforo y criticidad
comparativa_vars["Cambio_Semáforo_Res"] = comparativa_vars["Semáforo resiliencia_100diesel"] + " → " + comparativa_vars["Semáforo resiliencia_80solar20diesel"]
comparativa_vars["Cambio_Semáforo_Vul"] = comparativa_vars["Semáforo vulnerabilidad_100diesel"] + " → " + comparativa_vars["Semáforo vulnerabilidad_80solar20diesel"]
comparativa_vars["Cambio_Criticidad"] = comparativa_vars["Crítico_100diesel"] != comparativa_vars["Crítico_80solar20diesel"]

# --- Comparativa por dimensión ---
comparativa_dim = dim_100.merge(dim_8020, on="Dimensión", suffixes=("_100diesel", "_80solar20diesel"))

for metrica in ["Grado entrada", "Grado salida", "Betweenness", "Closeness", "Eigenvector", "Resiliencia (0-5)", "Vulnerabilidad (0-5)", "Score topológico"]:
    comparativa_dim[f"Δ_{metrica}"] = comparativa_dim[f"{metrica}_80solar20diesel"] - comparativa_dim[f"{metrica}_100diesel"]
    comparativa_dim[f"%Δ_{metrica}"] = 100 * comparativa_dim[f"Δ_{metrica}"] / comparativa_dim[f"{metrica}_100diesel"].replace(0, pd.NA)

# --- Exportar a Excel ---
with pd.ExcelWriter("comparativa_escenarios.xlsx", engine="openpyxl") as writer:
    comparativa_vars.to_excel(writer, sheet_name="Comparativa_Variables", index=False)
    comparativa_dim.to_excel(writer, sheet_name="Comparativa_Dimensiones", index=False)

# --- Resumen ejecutivo ---
print("\nTop 5 variables con mayor aumento de resiliencia:")
print(comparativa_vars.sort_values("Δ_Resiliencia (0-5)", ascending=False)[["ID","Nombre_100diesel","Δ_Resiliencia (0-5)"]].head(5))

print("\nTop 5 variables con mayor disminución de vulnerabilidad:")
print(comparativa_vars.sort_values("Δ_Vulnerabilidad (0-5)", ascending=True)[["ID","Nombre_100diesel","Δ_Vulnerabilidad (0-5)"]].head(5))

print("\nDimensiones con mayor mejora en score topológico:")
print(comparativa_dim.sort_values("Δ_Score topológico", ascending=False)[["Dimensión","Nombre dimensión_100diesel","Δ_Score topológico"]].head(5))

#%% Mapas comparativos: Centralidad y Resiliencia
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import networkx as nx

# Cargar comparativa
comparativa_vars = pd.read_excel("comparativa_escenarios.xlsx", sheet_name="Comparativa_Variables")

# Umbral de cambio significativo
umbral = 0.05  # 5% relativo

# Diccionarios de cambio
cambio_score = dict(zip(comparativa_vars["ID"], comparativa_vars["Δ_Score topológico"]))
cambio_resil = dict(zip(comparativa_vars["ID"], comparativa_vars["Δ_Resiliencia (0-5)"]))

# Función para asignar color
def color_por_cambio(delta):
    if delta > umbral:
        return "#2ca02c"  # verde
    elif delta < -umbral:
        return "#d62728"  # rojo
    else:
        return "#7f7f7f"  # gris

# Layout fijo para ambos mapas
pos = nx.spring_layout(G, seed=42, k=1.2)

# --- Mapa 1: Cambios en centralidad ---
node_colors_score = [color_por_cambio(cambio_score.get(n, 0)) for n in G.nodes()]
node_sizes_score = [800 + abs(cambio_score.get(n, 0)) * 3000 for n in G.nodes()]

plt.figure(figsize=(18, 14))
nx.draw_networkx_nodes(G, pos, node_color=node_colors_score, node_size=node_sizes_score, edgecolors="black")
nx.draw_networkx_labels(G, pos, labels={n: n for n in G.nodes()}, font_size=8, font_weight="bold")

for (u, v, data) in G.edges(data=True):
    t = data["tipo"]
    color = TIPOS[t]["color"]
    style = TIPOS[t]["style"]
    nx.draw_networkx_edges(
        G, pos, edgelist=[(u, v)],
        edge_color=color, style=style,
        arrows=True, arrowstyle='-|>', arrowsize=18, width=2.0,
        connectionstyle='arc3,rad=0.0',
        min_source_margin=15, min_target_margin=15
    )

legend_cambios = [
    Patch(facecolor="#2ca02c", edgecolor="black", label="Mejora"),
    Patch(facecolor="#d62728", edgecolor="black", label="Empeora"),
    Patch(facecolor="#7f7f7f", edgecolor="black", label="Cambio no significativo")
]
plt.legend(handles=legend_cambios, loc='upper right', fontsize=9, frameon=True)
plt.title("Mapa de cambios en centralidad (Score topológico) — 80% Solar vs 100% Diésel")
plt.axis('off')
plt.tight_layout()
plt.savefig("mapa_cambios_centralidad.png", dpi=300, bbox_inches="tight")
plt.show()

# --- Mapa 2: Cambios en resiliencia ---
node_colors_resil = [color_por_cambio(cambio_resil.get(n, 0)) for n in G.nodes()]
node_sizes_resil = [800 + abs(cambio_resil.get(n, 0)) * 3000 for n in G.nodes()]

plt.figure(figsize=(18, 14))
nx.draw_networkx_nodes(G, pos, node_color=node_colors_resil, node_size=node_sizes_resil, edgecolors="black")
nx.draw_networkx_labels(G, pos, labels={n: n for n in G.nodes()}, font_size=8, font_weight="bold")

for (u, v, data) in G.edges(data=True):
    t = data["tipo"]
    color = TIPOS[t]["color"]
    style = TIPOS[t]["style"]
    nx.draw_networkx_edges(
        G, pos, edgelist=[(u, v)],
        edge_color=color, style=style,
        arrows=True, arrowstyle='-|>', arrowsize=18, width=2.0,
        connectionstyle='arc3,rad=0.0',
        min_source_margin=15, min_target_margin=15
    )

plt.legend(handles=legend_cambios, loc='upper right', fontsize=9, frameon=True)
plt.title("Mapa de cambios en resiliencia (0–5) — 80% Solar vs 100% Diésel")
plt.axis('off')
plt.tight_layout()
plt.savefig("mapa_cambios_resiliencia.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
