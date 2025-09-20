# Socio_Eco_Island_Energy
Comparative framework for socio-ecological resilience &amp; vulnerability in island energy transitions. Models governance-economy-environment-technology networks. Evaluates scenarios (100% diesel vs hybrid) to guide sustainable policy. Modular, adaptable code under MIT license.
SocioEcoIslandEnergy is a flexible, open-source Python framework designed to model, visualize, and compare socio-ecological systems in island communities undergoing energy transitions — from fossil-fuel dependency (e.g., 100% diesel) to hybrid or renewable systems (e.g., 80% solar + 20% diesel).

It uses network analysis to map interactions between governance, economy, society, environment, infrastructure, and external factors. The framework outputs:

Interactive network graphs
Quantitative resilience/vulnerability scores
Comparative analysis between scenarios
Excel reports and visual change maps
This is not tied to Boutachauques Island — it’s a template you can adapt to your own context.

🧩 Core Components You Can Customize
Everything in this framework is modular and replaceable. Below are the key vectors and structures you should modify to adapt it to your case study.

1. ✏️ Dimensions (D1, D2, D3...)
Define your system’s main pillars.
dimensiones = ["D1", "D2", "D3", "D4", "D5", "D6", "D7"]

nombres = {
    "D1": "Governance and Institutional Framework",
    "D2": "Economic-Productive",
    "D3": "Social and Human Capital",
    "D4": "Ecosystem-Environmental",
    "D5": "Connectivity and External Factors",
    "D6": "Risks and Land-Use Conflicts",
    "D7": "Infrastructure and Energy Technology",
}
✅ Recommendation: Adjust dimension names and number to match your local context (e.g., add “Cultural Heritage” or “Water Security”). 
2. 🧬 Variables (Dx.Vy)
Break down each dimension into measurable or meaningful variables.
variables = [
    "D1.V1", "D1.V2", "D1.V3", "D1.V4",
    "D2.V1", "D2.V2", "D2.V3", "D2.V4", "D2.V5", "D2.V6",
    # ... etc.
]

etiquetas = {
    "D1.V1": "Local and national energy policy",
    "D1.V2": "Municipal and inter-institutional management",
    # ... etc.
}
✅ Recommendation: Replace with locally relevant indicators (e.g., “Tourism Pressure”, “Desalination Capacity”, “Indigenous Land Rights”). 
3. 🔄 Relationship Types
Define how variables/dimensions influence each other.
TIPOS = {
    "+":  {"nombre": "Enabling",        "color": "#2ca02c", "style": "solid"},
    "-":  {"nombre": "Restrictive",     "color": "#d62728", "style": "solid"},
    "↔+": {"nombre": "Mutual Reinforcement", "color": "#1f77b4", "style": "solid"},
    "↔±": {"nombre": "Mutual Compensation",  "color": "#ff7f0e", "style": "solid"},
    "⟂":  {"nombre": "Modulating (gating/threshold)", "color": "#7f7f7f", "style": "dashed"},
}
✅ Recommendation: Add new types if needed (e.g., “Delayed Effect”, “Nonlinear”, “Cyclical”). Just ensure you update the drawing logic.
4. ⚖️ Relationships Between Dimensions
Define the structural backbone of your system.
relaciones = [
    ("D1","D2","+"), 
    ("D2","D1","↔+"),
    ("D1","D4","+"), 
    ("D7","D4","+"),
    # ... etc.
]
✅ Recommendation: Use stakeholder workshops or literature reviews to define these. Test sensitivity by adding/removing links.
5. 🧭 Relationships Between Variables
Define granular interactions — the heart of the model.
relaciones = [
    ("D1.V1","D2.V5","+"),
    ("D7.V2","D4.V4","+"),
    ("D5.V3","D7.V2","-"),
    # ... etc.
]
✅ Recommendation: Start with 20–30 key relationships and expand iteratively. Use domain experts to validate.
6. 📊 Metrics & Scoring
The default uses degree centrality and a simple resiliency formula. You can redefine:
# Current resiliency formula (customizable)
resiliencia[n] = round((out_deg_n[n] + (1 - in_deg_n[n])) / 2 * 5, 2)

# Current vulnerability formula
vulnerabilidad[n] = round((in_deg_n[n] + (1 - out_deg_n[n])) / 2 * 5, 2)
✅ Recommendation: Replace with real data (e.g., survey scores, economic indicators, environmental indices). You can also integrate: 
+Eigenvector centrality for influence
+Betweenness for control points
+Closeness for accessibility
7. 🎨 Visualization & Layout
Colors, node sizes, arrow styles, and legends are fully customizable.
PALETA_DIM = {
    "D1": "#8dd3c7", "D2": "#ffffb3", "D3": "#bebada",
    "D4": "#fb8072", "D5": "#80b1d3", "D6": "#fdb462", "D7": "#b3de69",
}
✅ Recommendation: Use color-blind-friendly palettes. Adjust node sizes to reflect real-world weights (e.g., population, budget, emissions). 
8. 📈 Comparative Analysis Engine
The framework includes built-in comparison between two scenarios (e.g., baseline vs. future). You can:

Add a third scenario
Change comparison thresholds
Modify change categories (“Improved”, “Degraded”, “New”, “Lost”)
def clasificar_cambio(tipo_antiguo, tipo_nuevo):
    # Your custom logic here
    ...

    🚀 How to Get Started
Clone or download this repository.
Install dependencies
Open the main script (SocioEcoIslandEnergy.py or similar).
Replace dimension names, variables, and relationships with your own.
Run the script — outputs will be saved in C:\Python\MatrizSSE (or your chosen WORKDIR).
Review Excel reports and PNG graphs.
Iterate and refine with local knowledge.
📁 Output Files
After running, you’ll get:

grafo_dimensiones_*.png — Network graph of dimensions.
grafo_variables_*.png — Network graph of variables.
analisis_dimensiones_*.xlsx — Metrics per dimension.
analisis_variables_*.xlsx — Metrics per variable.
comparativa_escenarios.xlsx — Side-by-side comparison.
mapa_cambios_*.png — Visual change maps (resilience, centrality).

🌍 Why This Matters
Island communities face unique vulnerabilities: isolation, import dependency, climate exposure. Energy transitions are not just technical — they’re socio-ecological. This tool helps you:

Identify leverage points for intervention.
Anticipate unintended consequences.
Communicate complexity to policymakers and communities.
Monitor progress over time.
🤝 Contribution & Adaptation
This framework is designed for co-creation. You’re encouraged to:

Add new metrics (e.g., equity, gender, cultural resilience).
Integrate real-time data feeds.
Connect to agent-based or system dynamics models.
Translate into other languages.
Apply to non-island contexts (mountain communities, urban peripheries, etc.).
📜 License
MIT License — Use freely for academic, commercial, or community projects. Attribution appreciated.

📬 Contact & Support
For questions, adaptations, or collaboration:
→ [Carlos Lázaro Castillo García/Universidad de Concepción]
→ [carloscastillo2025@udec.cl]


