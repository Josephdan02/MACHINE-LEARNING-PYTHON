# ============================================================
# KNN PREDICT LAB - INTERFAZ WEB INTERACTIVA
# ============================================================
# Ejecutar con:
#   streamlit run app.py
#
# La logica de datos, entrenamiento y prediccion vive en
# knn_core.py. Los scripts originales no se modifican.
# ============================================================

from __future__ import annotations

import html

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

from knn_core import (
    K_VECINOS,
    ModelBundle,
    ModelSpec,
    entrenar_modelo,
    obtener_spec,
    predecir,
    tabla_dataset,
    tabla_reporte,
    tabla_vecinos,
)

# ------------------------------------------------------------
# CONFIGURACION GENERAL
# ------------------------------------------------------------

st.set_page_config(
    page_title="KNN Predict Lab",
    page_icon=":material/science:",
    layout="wide",
    initial_sidebar_state="auto",
)

AZUL_MARINO = "#172554"
VERDE_PETROLEO = "#0F766E"
TURQUESA = "#14B8A6"
GRIS_CLARO = "#F1F5F9"
BLANCO = "#FFFFFF"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": BLANCO,
    "axes.facecolor": BLANCO,
    "axes.edgecolor": "#CBD5E1",
    "axes.labelcolor": "#334155",
    "axes.titlecolor": AZUL_MARINO,
    "xtick.color": "#475569",
    "ytick.color": "#475569",
    "text.color": AZUL_MARINO,
    "grid.color": "#E2E8F0",
    "grid.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


@st.cache_resource(
    show_spinner="Entrenando el modelo y ajustando el escalador (solo la primera vez)..."
)
def cargar_modelo(model_key: str) -> ModelBundle:
    """Entrena cada modelo una sola vez y lo reutiliza en cada prediccion."""
    return entrenar_modelo(obtener_spec(model_key))


# ------------------------------------------------------------
# ESTILOS (paleta del laboratorio, sin gradientes)
# ------------------------------------------------------------

st.html("""
<style>
    :root {
        --navy: #172554;
        --petrol: #0F766E;
        --turquoise: #14B8A6;
        --light: #F1F5F9;
        --line: #E2E8F0;
    }

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 3.5rem;
        max-width: 1240px;
    }

    .lab-hero {
        background: var(--navy);
        color: #FFFFFF;
        border-radius: 18px;
        padding: 28px 32px;
        margin-bottom: 18px;
    }
    .lab-kicker {
        color: #5EEAD4;
        font-size: 0.72rem;
        letter-spacing: 0.14em;
        font-weight: 700;
    }
    .lab-hero h1 {
        color: #FFFFFF;
        font-size: 2.35rem;
        line-height: 1.1;
        margin: 4px 0 10px 0;
        letter-spacing: -0.02em;
    }
    .lab-hero p {
        color: #CBD5E1;
        font-size: 0.98rem;
        line-height: 1.55;
        margin: 0;
        max-width: 840px;
    }
    .lab-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 16px;
    }
    .lab-chips span {
        border: 1px solid rgba(255, 255, 255, 0.22);
        background: rgba(20, 184, 166, 0.12);
        color: #E2E8F0;
        border-radius: 999px;
        padding: 4px 11px;
        font-size: 0.75rem;
    }

    .model-intro {
        margin: 6px 0 14px 0;
    }
    .model-intro .eyebrow {
        color: var(--petrol);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
    }
    .model-intro h2 {
        color: var(--navy);
        margin: 2px 0 4px 0;
        font-size: 1.55rem;
    }
    .model-intro p {
        color: #475569;
        margin: 0;
        font-size: 0.92rem;
        line-height: 1.5;
    }

    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 13px 16px;
        box-shadow: 0 1px 2px rgba(23, 37, 84, 0.04);
    }
    [data-testid="stMetricLabel"] p {
        color: #475569;
        font-size: 0.78rem;
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        color: var(--petrol);
        font-size: 1.5rem;
    }

    .result-banner {
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 20px 18px 18px 18px;
        text-align: center;
    }
    .result-si {
        background: #ECFDF5;
        border-color: #99F6E4;
    }
    .result-no {
        background: var(--light);
        border-color: #CBD5E1;
    }
    .result-eyebrow {
        color: #475569;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.12em;
    }
    .result-value {
        font-size: 3.4rem;
        font-weight: 800;
        line-height: 1.05;
        letter-spacing: -0.03em;
        margin: 2px 0 4px 0;
    }
    .result-si .result-value { color: var(--petrol); }
    .result-no .result-value { color: var(--navy); }
    .result-desc {
        color: #334155;
        font-size: 0.88rem;
        line-height: 1.45;
    }

    .empty-result {
        border: 1px dashed #CBD5E1;
        background: #F8FAFC;
        border-radius: 16px;
        padding: 26px 22px;
        text-align: center;
        color: #475569;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    .empty-result strong { color: var(--navy); }
    .empty-result .empty-icon {
        font-size: 1.6rem;
        color: var(--petrol);
        display: block;
        margin-bottom: 6px;
    }

    .vote-head {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        color: #334155;
        font-size: 0.82rem;
        margin-bottom: 5px;
    }
    .vote-track {
        height: 8px;
        background: var(--line);
        border-radius: 999px;
        overflow: hidden;
        margin-bottom: 12px;
    }
    .vote-fill {
        height: 8px;
        border-radius: 999px;
    }

    .explain {
        background: #F8FAFC;
        border-left: 3px solid var(--turquoise);
        border-radius: 0 10px 10px 0;
        padding: 12px 14px;
        color: #334155;
        font-size: 0.86rem;
        line-height: 1.55;
    }

    .sidebar-brand {
        color: var(--navy);
        font-size: 1.15rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        margin-bottom: 2px;
    }
    .sidebar-sub {
        color: #64748B;
        font-size: 0.76rem;
        margin-bottom: 14px;
    }

    .small-note {
        color: #64748B;
        font-size: 0.78rem;
        line-height: 1.5;
    }

    @media (max-width: 760px) {
        .lab-hero { padding: 22px 20px; }
        .lab-hero h1 { font-size: 1.85rem; }
        .result-value { font-size: 2.8rem; }
    }
</style>
""")


# ------------------------------------------------------------
# GRAFICOS
# ------------------------------------------------------------

def figura_vecinos(bundle: ModelBundle, resultado: dict | None):
    """Dispersion 2D de los registros de entrenamiento y los 3 vecinos."""
    spec = bundle.spec
    y_train = bundle.y_train.to_numpy()
    X_train = bundle.X_train

    fig, ax = plt.subplots(figsize=(9.2, 4.9), dpi=120)

    for clase in sorted(bundle.modelo.classes_):
        mascara = y_train == clase
        ax.scatter(
            X_train[mascara, 0],
            X_train[mascara, 1],
            s=72,
            color=spec.colores_clase[int(clase)],
            alpha=0.85,
            edgecolors=BLANCO,
            linewidths=0.8,
            label=f"{spec.etiquetas[int(clase)]} ({clase})",
            zorder=2,
        )

    if resultado is not None:
        nuevo_x = resultado["punto_nuevo"]["x"]
        nuevo_y = resultado["punto_nuevo"]["y"]

        desplazamientos = {1: (8, 8), 2: (8, -16), 3: (-58, 8)}
        for vecino in resultado["vecinos"]:
            ax.plot(
                [nuevo_x, vecino["x"]],
                [nuevo_y, vecino["y"]],
                color=TURQUESA,
                linestyle=(0, (4, 3)),
                linewidth=1.5,
                zorder=3,
            )
            ax.scatter(
                vecino["x"],
                vecino["y"],
                s=280,
                facecolors="none",
                edgecolors=TURQUESA,
                linewidths=2.4,
                zorder=4,
            )
            ax.annotate(
                f"Vecino {vecino['orden']}",
                (vecino["x"], vecino["y"]),
                xytext=desplazamientos.get(vecino["orden"], (8, 8)),
                textcoords="offset points",
                fontsize=8.5,
                fontweight="bold",
                color=AZUL_MARINO,
                zorder=6,
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": BLANCO,
                    "edgecolor": "none",
                    "alpha": 0.8,
                },
            )

        ax.scatter(
            [nuevo_x],
            [nuevo_y],
            marker="*",
            s=440,
            color=AZUL_MARINO,
            edgecolors=BLANCO,
            linewidths=1.6,
            label=spec.nuevo_registro,
            zorder=7,
        )
        ax.set_title(
            "Los 3 vecinos más cercanos",
            loc="left",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
    else:
        ax.set_title(
            "Registros de entrenamiento por clase",
            loc="left",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )

    ax.set_xlabel(spec.x_label, fontsize=10)
    ax.set_ylabel(spec.y_label, fontsize=10)
    ax.margins(0.12)
    ax.grid(True, alpha=0.55)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, 0.09, 1, 1))

    # Leyenda fuera del area de datos para no tapar ningun registro.
    manejadores, etiquetas = ax.get_legend_handles_labels()
    fig.legend(
        manejadores,
        etiquetas,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        frameon=False,
        fontsize=8.5,
        handletextpad=0.6,
        columnspacing=1.2,
    )
    return fig


def figura_matriz(bundle: ModelBundle):
    """Mapa de calor de la matriz de confusion real."""
    spec = bundle.spec
    clases = [int(c) for c in bundle.modelo.classes_]
    matriz = bundle.matriz

    mapa = LinearSegmentedColormap.from_list(
        "knn_lab",
        ["#F8FAFC", "#CCFBF1", TURQUESA, VERDE_PETROLEO, AZUL_MARINO],
    )

    fig, ax = plt.subplots(figsize=(5.3, 4.1), dpi=130)
    imagen = ax.imshow(
        matriz,
        cmap=mapa,
        vmin=0,
        vmax=max(int(matriz.max()), 1),
    )

    ax.set_xticks(
        range(len(clases)),
        [f"{spec.etiquetas[c]}\n(clase {c})" for c in clases],
    )
    ax.set_yticks(
        range(len(clases)),
        [f"{spec.etiquetas[c]}\n(clase {c})" for c in clases],
    )
    ax.set_xlabel("Predicción del modelo", fontsize=9)
    ax.set_ylabel("Clase real", fontsize=9)
    ax.grid(False)

    umbral = max(int(matriz.max()), 1) * 0.55
    for i in range(len(clases)):
        for j in range(len(clases)):
            valor = int(matriz[i, j])
            ax.text(
                j, i, str(valor),
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color=BLANCO if valor > umbral else AZUL_MARINO,
            )

    ax.set_title(
        "Matriz de confusión · conjunto de prueba",
        loc="left",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)

    barra = fig.colorbar(imagen, ax=ax, fraction=0.046, pad=0.04)
    barra.ax.tick_params(labelsize=8)
    barra.outline.set_visible(False)

    fig.tight_layout()
    return fig


# ------------------------------------------------------------
# COMPONENTES DE LA INTERFAZ
# ------------------------------------------------------------

def encabezado():
    st.html("""
    <div class="lab-hero">
        <div class="lab-kicker">LABORATORIO DE CLASIFICACIÓN SUPERVISADA</div>
        <h1>KNN Predict Lab</h1>
        <p>
            Explora dos modelos KNN entrenados con los datos de este proyecto:
            satisfacción del cliente e inversión empresarial. Ajusta las variables,
            ejecuta una predicción y observa cómo votan los tres vecinos más cercanos.
        </p>
        <div class="lab-chips">
            <span>K = 3</span>
            <span>30 registros por dataset</span>
            <span>4 variables por modelo</span>
            <span>scikit-learn + TensorFlow</span>
        </div>
    </div>
    """)


def encabezado_modelo(spec: ModelSpec):
    st.html(f"""
    <div class="model-intro">
        <div class="eyebrow">MODELO ACTIVO</div>
        <h2>{html.escape(spec.nombre)}</h2>
        <p>{html.escape(spec.descripcion)}</p>
    </div>
    """)


def metricas_panel(bundle: ModelBundle):
    spec = bundle.spec
    columnas = st.columns(4)
    columnas[0].metric(
        "Exactitud en prueba",
        f"{bundle.exactitud * 100:.2f} %",
        help="accuracy_score sobre los 8 registros del conjunto de prueba.",
    )
    columnas[1].metric(
        "Registros del dataset",
        f"{len(bundle.df)}",
        help="Cantidad real de filas del dataset original del script.",
    )
    columnas[2].metric(
        "Vecinos (K)",
        f"{K_VECINOS}",
        help="Valor configurado en KNeighborsClassifier. No se modifica desde la interfaz.",
    )
    columnas[3].metric(
        "Entrenamiento / prueba",
        f"{len(bundle.X_train)} / {len(bundle.X_test)}",
        help="División 75 / 25 con random_state=42 y stratify=y.",
    )


def control_formulario(feature, clave: str) -> int:
    """Crea el control adecuado para cada variable real del modelo."""
    if feature.control == "slider":
        valor = st.slider(
            feature.label,
            min_value=feature.min_value,
            max_value=feature.max_value,
            value=feature.default,
            step=feature.step,
            help=feature.help,
            key=clave,
        )
    else:
        valor = st.number_input(
            feature.label,
            min_value=feature.min_value,
            max_value=feature.max_value,
            value=feature.default,
            step=feature.step,
            help=feature.help,
            key=clave,
            format="%d",
        )
    st.caption(feature.hint)
    return int(valor)


def bloque_votos(resultado: dict, spec: ModelSpec):
    total = sum(resultado["votos"].values())
    filas = []
    for clase in sorted(resultado["votos"]):
        votos = resultado["votos"][clase]
        porcentaje = (votos / total * 100) if total else 0
        filas.append(f"""
        <div class="vote-head">
            <span>Clase {clase} · {html.escape(spec.etiquetas[clase])}</span>
            <strong>{votos} de {total} votos</strong>
        </div>
        <div class="vote-track">
            <div class="vote-fill" style="width:{porcentaje:.1f}%;
                 background:{spec.colores_clase[clase]};"></div>
        </div>
        """)
    st.html("".join(filas))


def bloque_resultado(bundle: ModelBundle, resultado: dict | None):
    spec = bundle.spec

    if resultado is None:
        st.html("""
        <div class="empty-result">
            <span class="empty-icon">◎</span>
            <strong>Aún no hay una predicción.</strong><br>
            Completa el formulario y pulsa
            <strong>Ejecutar predicción KNN</strong>.<br>
            El resultado aparecerá aquí con la votación de los tres vecinos.
        </div>
        """)
        st.markdown("")
        st.markdown("**Esta sección mostrará**")
        st.markdown(
            "- La clase asignada por el modelo (SÍ o NO).\n"
            "- Los votos de los tres vecinos para cada clase.\n"
            "- Una explicación breve basada en esos votos."
        )
        return

    clase = resultado["clase"]
    es_positivo = clase == 1
    clase_css = "result-si" if es_positivo else "result-no"

    st.html(f"""
    <div class="result-banner {clase_css}">
        <div class="result-eyebrow">RESULTADO DEL MODELO KNN · K = {K_VECINOS}</div>
        <div class="result-value">{"SÍ" if es_positivo else "NO"}</div>
        <div class="result-desc">
            <strong>{html.escape(resultado["etiqueta"])}</strong><br>
            {html.escape(resultado["descripcion"])}
        </div>
    </div>
    """)

    st.caption(
        "Proporción de vecinos que respaldan la clase elegida: "
        f"{resultado['votos'][clase]} de {sum(resultado['votos'].values())} "
        f"({resultado['confianza_votacion'] * 100:.0f} %)."
    )

    if resultado["advertencias"]:
        st.warning(
            "Hay valores fuera del rango observado en el dataset. "
            "La estimación puede ser menos fiable:\n\n- "
            + "\n- ".join(resultado["advertencias"])
        )

    st.markdown("**Votación de los vecinos**")
    bloque_votos(resultado, spec)

    votos = resultado["votos"]
    total = sum(votos.values())
    detalle_votos = " · ".join(
        f"{spec.etiquetas[c]}: {votos[c]}" for c in sorted(votos)
    )
    explicacion = (
        f"El modelo normalizó las {len(spec.features)} características y buscó los "
        f"{total} registros de entrenamiento más cercanos en ese espacio. "
        f"La votación de los vecinos fue: {detalle_votos}. Como la mayoría "
        f"pertenece a la clase {clase} ({resultado['etiqueta']}), el KNN asigna "
        f"esa clase al nuevo {spec.sujeto}."
    )
    st.html(f'<div class="explain">{html.escape(explicacion)}</div>')
    st.caption(
        f"Las distancias se calcularon con las {len(spec.features)} variables "
        "normalizadas. El gráfico de vecinos muestra solo dos de ellas."
    )

    with st.expander("Ver los datos enviados al modelo"):
        tabla_entrada = pd.DataFrame({
            "Característica": [f.label for f in spec.features],
            "Valor ingresado": [
                resultado["valores"][f.key] for f in spec.features
            ],
            "Valor normalizado": [
                round(resultado["valores_escalados"][f.key], 3)
                for f in spec.features
            ],
        })
        st.dataframe(tabla_entrada, hide_index=True, width="stretch")
        st.caption(
            "El escalador entrenado transforma estos valores antes de calcular "
            "las distancias. El orden de las variables es el mismo del dataset."
        )


def seccion_vecinos(bundle: ModelBundle, resultado: dict | None):
    spec = bundle.spec
    st.markdown("## :material/hub: Vecinos más cercanos")
    st.caption(
        "La cercanía se calcula con todas las variables del modelo. "
        "El gráfico usa las dos primeras únicamente como apoyo visual."
    )

    with st.container(border=True):
        figura = figura_vecinos(bundle, resultado)
        st.pyplot(figura, width="stretch")
        plt.close(figura)
        st.caption(
            f"Variables del eje: {spec.features[0].label.lower()} y "
            f"{spec.features[1].label.lower()}, ambas normalizadas. "
            "Los puntos son los registros de entrenamiento; algunos "
            "comparten coordenadas y se superponen."
        )

        if resultado is None:
            st.html("""
            <div class="empty-result">
                <span class="empty-icon">◎</span>
                <strong>Sin vecinos todavía.</strong><br>
                Ejecuta una predicción para ver la clase, la distancia
                y el voto de cada vecino.
            </div>
            """)
        else:
            st.divider()
            st.markdown("**Los tres vecinos del registro**")
            tabla = tabla_vecinos(resultado).set_index("Vecino")
            tabla["Distancia"] = tabla["Distancia"].map(lambda d: f"{d:.4f}")
            st.table(tabla)
            st.caption(
                "Cada vecino vota por su propia clase. La distancia es "
                "euclidiana y usa las "
                f"{len(spec.features)} variables normalizadas."
            )

            with st.expander("Ver características completas de los vecinos"):
                filas = []
                for vecino in resultado["vecinos"]:
                    fila = {
                        "Vecino": f"Vecino {vecino['orden']}",
                        "Clase": vecino["clase"],
                    }
                    fila.update({
                        f.label: vecino["caracteristicas"][f.key]
                        for f in spec.features
                    })
                    fila["Distancia"] = round(vecino["distancia"], 4)
                    filas.append(fila)
                st.dataframe(
                    pd.DataFrame(filas),
                    hide_index=True,
                    width="stretch",
                )
                st.caption(
                    "Valores originales (sin normalizar) de los registros "
                    "de entrenamiento seleccionados por el modelo."
                )


def seccion_evaluacion(bundle: ModelBundle):
    spec = bundle.spec
    st.markdown("## :material/fact_check: Evaluación del modelo")
    st.caption(
        "Métricas reales calculadas una sola vez sobre el conjunto de prueba. "
        "No dependen de los valores del formulario."
    )

    tab_matriz, tab_reporte, tab_datos = st.tabs([
        "Matriz de confusión",
        "Reporte de clasificación",
        "Datos y entrenamiento",
    ])

    aciertos = int(np.trace(bundle.matriz))
    total_prueba = len(bundle.y_test)

    with tab_matriz:
        columna_figura, columna_lectura = st.columns([1.1, 1], gap="large")
        with columna_figura:
            figura = figura_matriz(bundle)
            st.pyplot(figura, width="stretch")
            plt.close(figura)
        with columna_lectura:
            st.metric(
                "Exactitud del modelo",
                f"{bundle.exactitud * 100:.2f} %",
                help="Porcentaje de aciertos sobre el conjunto de prueba.",
            )
            st.metric(
                "Aciertos en prueba",
                f"{aciertos} de {total_prueba}",
            )
            st.markdown("**Cómo leer la matriz**")
            st.markdown(
                "- Las filas son la **clase real** y las columnas la **predicción**.\n"
                "- La diagonal principal contiene los aciertos.\n"
                "- Los valores fuera de la diagonal son errores de clasificación."
            )
            st.caption(
                "La matriz se genera con confusion_matrix(y_test, predicciones) "
                "usando las clases del modelo entrenado."
            )

    with tab_reporte:
        tabla = tabla_reporte(bundle)
        st.dataframe(
            tabla,
            hide_index=True,
            width="stretch",
            column_config={
                "Precisión": st.column_config.NumberColumn(format="%.3f"),
                "Recall": st.column_config.NumberColumn(format="%.3f"),
                "F1-score": st.column_config.NumberColumn(format="%.3f"),
                "Soporte": st.column_config.NumberColumn(format="%d"),
            },
        )
        st.caption(
            "Reporte calculado con classification_report(..., zero_division=0) "
            f"sobre {total_prueba} registros de prueba."
        )
        st.markdown(
            f"**Exactitud global:** {bundle.exactitud * 100:.2f} % "
            f"({aciertos} de {total_prueba} aciertos)."
        )
        st.info(
            "La exactitud se muestra tal como la devuelve el modelo entrenado. "
            "No se introducen porcentajes manuales en la interfaz.",
            icon=":material/verified:",
        )

    with tab_datos:
        st.markdown("**Dataset completo**")
        st.dataframe(
            tabla_dataset(bundle),
            hide_index=True,
            width="stretch",
            height=300,
        )
        st.caption(
            f"{len(bundle.df)} registros originales. Clase 0 = "
            f"{spec.etiquetas[0].lower()}; clase 1 = "
            f"{spec.etiquetas[1].lower()}."
        )
        vista = tabla_dataset(bundle)
        st.download_button(
            "Descargar dataset (CSV)",
            data=vista.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"dataset_{spec.key}.csv",
            mime="text/csv",
            icon=":material/download:",
            width="content",
        )

        st.divider()
        columna_resumen, columna_clases, columna_tf = st.columns(
            [1.05, 0.95, 1.2], gap="large"
        )

        with columna_resumen:
            st.markdown("**Resumen del entrenamiento**")
            st.markdown(
                f"- Registros totales: **{len(bundle.df)}**\n"
                f"- Entrenamiento: **{len(bundle.X_train)}**\n"
                f"- Prueba: **{len(bundle.X_test)}**\n"
                f"- Variables de entrada: **{len(spec.features)}**\n"
                f"- Vecinos K: **{K_VECINOS}**\n"
                f"- División: **75 / 25**, random_state = **42**"
            )
            st.caption(
                "El escalador se ajusta solo con el conjunto de entrenamiento "
                "y se reutiliza en cada predicción."
            )

        with columna_clases:
            st.markdown("**Distribución de clases**")
            conteo = bundle.df[spec.target].value_counts().sort_index()
            for clase, cantidad in conteo.items():
                st.markdown(
                    f"- {spec.etiquetas[int(clase)]} (clase {int(clase)}): "
                    f"**{int(cantidad)}**"
                )
            st.caption(
                "La división es estratificada para conservar esta proporción "
                "en entrenamiento y prueba."
            )

        with columna_tf:
            with st.container(border=True):
                st.markdown("**Complemento TensorFlow**")
                if bundle.tensor_forma is not None:
                    st.caption(
                        "El conjunto de entrenamiento normalizado se convierte en "
                        f"un tensor float32 de forma {bundle.tensor_forma} "
                        f"(TensorFlow {bundle.tensorflow_version}). TensorFlow "
                        "no implementa el clasificador: el modelo es "
                        "KNeighborsClassifier de scikit-learn."
                    )
                else:
                    st.caption(
                        "TensorFlow no está disponible en este entorno. "
                        "El clasificador KNN de scikit-learn sigue funcionando."
                    )
                    if bundle.tensorflow_error:
                        st.caption(f"Detalle: {bundle.tensorflow_error}")


# ------------------------------------------------------------
# APLICACION
# ------------------------------------------------------------

encabezado()

with st.sidebar:
    st.html("""
    <div class="sidebar-brand">KNN Predict Lab</div>
    <div class="sidebar-sub">Laboratorio de predicción · Machine Learning</div>
    """)

    st.markdown("### Modelo a analizar")
    opciones = ["Satisfacción del cliente", "Inversión empresarial"]
    seleccion = st.radio(
        "Selecciona un algoritmo",
        opciones,
        captions=[
            "Clasifica clientes satisfechos o insatisfechos.",
            "Clasifica empresarios que sí o no invertirían.",
        ],
        label_visibility="collapsed",
    )
    model_key = "satisfaccion" if seleccion == opciones[0] else "inversion"
    spec = obtener_spec(model_key)

    # Al cambiar de modelo, el formulario y el resultado son independientes.
    resultado_key = f"resultado_{spec.key}"
    aviso_key = f"nuevo_{spec.key}"

    bundle = cargar_modelo(spec.key)

    st.divider()
    st.markdown("**Ficha del modelo**")
    st.markdown(
        f"- Registros: **{len(bundle.df)}**\n"
        f"- Entrenamiento / prueba: **{len(bundle.X_train)} / {len(bundle.X_test)}**\n"
        f"- Vecinos K: **{K_VECINOS}**\n"
        f"- Exactitud: **{bundle.exactitud * 100:.2f} %**"
    )
    st.caption(
        "El modelo se entrenó una vez con los datos originales y se reutiliza "
        "en cada predicción."
    )

    st.divider()
    st.markdown("**Cómo usar el laboratorio**")
    st.markdown(
        "1. Elige un modelo.\n"
        "2. Ajusta las cuatro variables del formulario.\n"
        "3. Pulsa **Ejecutar predicción KNN**.\n"
        "4. Revisa el resultado, los vecinos y las métricas."
    )

    with st.expander("Ayuda: variables y lectura"):
        for feature in spec.features:
            st.markdown(f"**{feature.label}**")
            st.caption(f"{feature.help} {feature.hint}")

        st.markdown("**Clases del modelo**")
        st.caption(
            f"Clase 0 = {spec.etiquetas[0]}. "
            f"Clase 1 = {spec.etiquetas[1]}."
        )
        st.markdown("**Sobre el gráfico**")
        st.caption(
            "El gráfico muestra dos variables normalizadas; la distancia KNN "
            "se calcula con las cuatro. Por eso una línea corta en el dibujo "
            "no implica necesariamente la distancia más pequeña."
        )

    st.divider()
    if st.button(
        "Restablecer simulación",
        icon=":material/restart_alt:",
        width="stretch",
        help="Borra el resultado actual y devuelve el formulario a los valores del ejemplo.",
    ):
        for clave in list(st.session_state.keys()):
            if clave.startswith(f"widget_{spec.key}_"):
                del st.session_state[clave]
        st.session_state.pop(resultado_key, None)
        st.session_state.pop(aviso_key, None)
        st.rerun()

    st.html("""
    <div class="small-note" style="margin-top:16px;">
        Proyecto académico de Machine Learning · Quinto semestre.<br>
        Los datasets y el valor K provienen de los scripts originales.
    </div>
    """)

if st.session_state.pop(aviso_key, False):
    st.toast("Predicción calculada con el modelo KNN.", icon=":material/check_circle:")

metricas_panel(bundle)
encabezado_modelo(spec)

columna_formulario, columna_resultado = st.columns([1.02, 0.98], gap="large")

with columna_formulario:
    with st.container(border=True):
        st.markdown("#### :material/tune: Simulador de predicción")
        st.caption(
            "Los valores iniciales reproducen el ejemplo de los scripts originales. "
            "Cada campo respeta la escala y el orden real del modelo."
        )

        with st.form(f"formulario_{spec.key}", border=False):
            valores: dict[str, int] = {}
            for feature in spec.features:
                valores[feature.key] = control_formulario(
                    feature,
                    f"widget_{spec.key}_{feature.key}",
                )
            enviado = st.form_submit_button(
                "Ejecutar predicción KNN",
                type="primary",
                icon=":material/play_arrow:",
                width="stretch",
            )

        if enviado:
            try:
                with st.spinner("Aplicando el escalador y buscando los 3 vecinos..."):
                    resultado_nuevo = predecir(bundle, valores)
                st.session_state[resultado_key] = resultado_nuevo
                st.session_state[aviso_key] = True
                st.rerun()
            except Exception as exc:
                st.error(
                    "No se pudo calcular la predicción. Revisa los valores e "
                    f"inténtalo de nuevo. Detalle: {exc}",
                    icon=":material/error:",
                )

with columna_resultado:
    with st.container(border=True):
        st.markdown("#### :material/insights: Resultado de la predicción")
        bloque_resultado(bundle, st.session_state.get(resultado_key))

st.divider()
seccion_evaluacion(bundle)
st.divider()
seccion_vecinos(bundle, st.session_state.get(resultado_key))

st.html("""
<div class="small-note" style="margin-top:28px; border-top:1px solid #E2E8F0; padding-top:14px;">
    KNN Predict Lab · Interfaz construida sobre los scripts
    <code>knn_satisfaccion.py</code> y <code>knn_inversion.py</code>.
    Las predicciones, distancias y métricas se calculan en tiempo de ejecución
    con el modelo y el escalador entrenados; no son valores simulados.
</div>
""")
