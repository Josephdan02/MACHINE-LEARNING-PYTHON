# ============================================================
# KNN PREDICT LAB - NUCLEO COMPARTIDO DE LOS DOS MODELOS
# ============================================================
# Este archivo concentra los datasets, el entrenamiento, las
# metricas y la logica de prediccion. La interfaz web (app.py)
# solo se encarga de mostrar y recoger los datos.
#
# Los datasets y el flujo son exactamente los de:
#   - knn_satisfaccion.py
#   - knn_inversion.py
# K = 3, test_size = 0.25, random_state = 42 y stratify = y.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

K_VECINOS = 3
TEST_SIZE = 0.25
RANDOM_STATE = 42


# ------------------------------------------------------------
# 1. DATASETS ORIGINALES (sin modificar registros ni etiquetas)
# ------------------------------------------------------------

DATOS_SATISFACCION = {
    'calidad_servicio': [5, 4, 2, 1, 5, 3, 2, 4, 1, 5,
                         3, 2, 4, 1, 5, 4, 3, 2, 5, 1,
                         4, 3, 2, 5, 1, 4, 3, 2, 5, 1],
    'atencion_cliente': [5, 4, 2, 1, 5, 3, 2, 4, 1, 5,
                         4, 2, 5, 1, 5, 4, 3, 2, 5, 1,
                         5, 3, 2, 5, 1, 4, 3, 2, 5, 1],
    'tiempo_espera': [1, 2, 5, 5, 1, 3, 4, 2, 5, 1,
                      3, 5, 2, 5, 1, 2, 3, 4, 1, 5,
                      2, 3, 5, 1, 5, 2, 3, 4, 1, 5],
    'precio': [4, 4, 2, 1, 5, 3, 2, 4, 1, 5,
               3, 2, 4, 1, 5, 4, 3, 2, 5, 1,
               4, 3, 2, 5, 1, 4, 3, 2, 5, 1],
    'satisfaccion': [
        1, 1, 0, 0, 1, 1, 0, 1, 0, 1,
        1, 0, 1, 0, 1, 1, 1, 0, 1, 0,
        1, 1, 0, 1, 0, 1, 1, 0, 1, 0
    ]
}

DATOS_INVERSION = {
    'presupuesto': [
        5, 4, 2, 1, 5, 3, 2, 4, 1, 5,
        4, 2, 5, 1, 3, 4, 2, 5, 1, 4,
        5, 3, 2, 4, 1, 5, 3, 2, 4, 1
    ],
    'trabajadores': [
        20, 15, 5, 3, 25, 10, 4, 18, 2, 30,
        15, 5, 25, 3, 12, 20, 6, 28, 2, 18,
        30, 10, 5, 20, 3, 25, 12, 4, 18, 2
    ],
    'procesos_manuales': [
        5, 4, 2, 1, 5, 4, 2, 5, 1, 5,
        4, 2, 5, 1, 3, 5, 2, 5, 1, 4,
        5, 3, 2, 5, 1, 5, 4, 2, 5, 1
    ],
    'costos_operativos': [
        5, 4, 2, 1, 5, 3, 2, 4, 1, 5,
        4, 2, 5, 1, 3, 4, 2, 5, 1, 4,
        5, 3, 2, 4, 1, 5, 3, 2, 4, 1
    ],
    'invierte': [
        1, 1, 0, 0, 1, 1, 0, 1, 0, 1,
        1, 0, 1, 0, 1, 1, 0, 1, 0, 1,
        1, 1, 0, 1, 0, 1, 1, 0, 1, 0
    ]
}


# ------------------------------------------------------------
# 2. DEFINICION DE VARIABLES Y MODELOS
# ------------------------------------------------------------
# El orden de "features" es el mismo que usan los scripts
# originales para construir X. No se debe cambiar.

@dataclass(frozen=True)
class Feature:
    key: str
    label: str
    control: str          # "slider" o "number"
    min_value: int
    max_value: int
    step: int
    default: int
    help: str
    hint: str = ""


@dataclass(frozen=True)
class ModelSpec:
    key: str
    nombre: str
    descripcion_corta: str
    descripcion: str
    target: str
    datos: dict[str, list[int]]
    features: tuple[Feature, ...]
    etiquetas: dict[int, str]
    resultado_positivo: str
    resultado_negativo: str
    colores_clase: dict[int, str]
    nuevo_registro: str
    sujeto: str
    x_label: str
    y_label: str

    @property
    def feature_keys(self) -> list[str]:
        return [f.key for f in self.features]


SATISFACCION = ModelSpec(
    key="satisfaccion",
    nombre="Satisfacción del cliente",
    descripcion_corta="Predice si un cliente quedará satisfecho con el servicio.",
    descripcion=(
        "El modelo clasifica a un cliente como satisfecho o insatisfecho a partir "
        "de cuatro características de su experiencia. Usa los 22 registros de "
        "entrenamiento y el escalador ajustado en el script original."
    ),
    target="satisfaccion",
    datos=DATOS_SATISFACCION,
    features=(
        Feature(
            key="calidad_servicio",
            label="Calidad del servicio",
            control="slider",
            min_value=1, max_value=5, step=1, default=5,
            help="Percepción de la calidad del servicio recibido.",
            hint="1 = muy mala · 5 = excelente",
        ),
        Feature(
            key="atencion_cliente",
            label="Atención al cliente",
            control="slider",
            min_value=1, max_value=5, step=1, default=4,
            help="Nivel de atención recibido por el cliente.",
            hint="1 = muy deficiente · 5 = excelente",
        ),
        Feature(
            key="tiempo_espera",
            label="Tiempo de espera",
            control="number",
            min_value=1, max_value=5, step=1, default=1,
            help="Tiempo de espera percibido por el cliente.",
            hint="1 = espera muy corta · 5 = espera muy larga",
        ),
        Feature(
            key="precio",
            label="Precio",
            control="slider",
            min_value=1, max_value=5, step=1, default=4,
            help="Valoración del precio en la escala del dataset (1 a 5).",
            hint="1 = valoración mínima · 5 = valoración máxima",
        ),
    ),
    etiquetas={0: "Insatisfecho", 1: "Satisfecho"},
    resultado_positivo="El cliente está satisfecho.",
    resultado_negativo="El cliente está insatisfecho.",
    colores_clase={0: "#94A3B8", 1: "#0F766E"},
    nuevo_registro="Nuevo cliente",
    sujeto="cliente",
    x_label="Calidad del servicio (normalizada)",
    y_label="Atención al cliente (normalizada)",
)

INVERSION = ModelSpec(
    key="inversion",
    nombre="Inversión empresarial",
    descripcion_corta="Predice si un empresario invertirá en un software de automatización.",
    descripcion=(
        "El modelo estima si un empresario invertirá en el desarrollo de un software "
        "para automatizar sus procesos, a partir de cuatro características de su "
        "empresa. Usa los 22 registros de entrenamiento y el escalador ajustado "
        "en el script original."
    ),
    target="invierte",
    datos=DATOS_INVERSION,
    features=(
        Feature(
            key="presupuesto",
            label="Presupuesto disponible",
            control="slider",
            min_value=1, max_value=5, step=1, default=5,
            help="Nivel de presupuesto disponible para el proyecto.",
            hint="1 = muy limitado · 5 = amplio",
        ),
        Feature(
            key="trabajadores",
            label="Número de trabajadores",
            control="number",
            min_value=1, max_value=100, step=1, default=25,
            help="Cantidad de trabajadores de la empresa.",
            hint="El dataset original contiene entre 2 y 30 trabajadores.",
        ),
        Feature(
            key="procesos_manuales",
            label="Procesos manuales",
            control="slider",
            min_value=1, max_value=5, step=1, default=5,
            help="Grado de procesos que todavía se realizan de forma manual.",
            hint="1 = pocos procesos manuales · 5 = casi todo manual",
        ),
        Feature(
            key="costos_operativos",
            label="Costos operativos",
            control="slider",
            min_value=1, max_value=5, step=1, default=4,
            help="Nivel de costos operativos de la empresa.",
            hint="1 = muy bajos · 5 = muy altos",
        ),
    ),
    etiquetas={0: "No invierte", 1: "Sí invierte"},
    resultado_positivo="El empresario sí invertirá en el software.",
    resultado_negativo="El empresario no invertirá en el software.",
    colores_clase={0: "#94A3B8", 1: "#0F766E"},
    nuevo_registro="Nuevo empresario",
    sujeto="empresario",
    x_label="Presupuesto (normalizado)",
    y_label="Trabajadores (normalizados)",
)

MODELOS: dict[str, ModelSpec] = {
    SATISFACCION.key: SATISFACCION,
    INVERSION.key: INVERSION,
}


def obtener_spec(model_key: str) -> ModelSpec:
    """Devuelve la definicion del modelo solicitado."""
    if model_key not in MODELOS:
        raise KeyError(f"Modelo desconocido: {model_key}")
    return MODELOS[model_key]


# ------------------------------------------------------------
# 3. RESULTADO DEL ENTRENAMIENTO
# ------------------------------------------------------------

@dataclass
class ModelBundle:
    spec: ModelSpec
    df: pd.DataFrame
    X_train_df: pd.DataFrame
    X_test_df: pd.DataFrame
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: pd.Series
    y_test: pd.Series
    escalador: StandardScaler
    modelo: KNeighborsClassifier
    predicciones: np.ndarray
    exactitud: float
    matriz: np.ndarray
    reporte: dict[str, Any]
    tensor_forma: tuple[int, ...] | None
    tensorflow_version: str | None
    tensorflow_error: str | None


# ------------------------------------------------------------
# 4. COMPLEMENTO TENSORFLOW
# ------------------------------------------------------------
# TensorFlow se usa solo para convertir el conjunto normalizado
# en un tensor, igual que en los scripts. El clasificador KNN es
# de scikit-learn.

@lru_cache(maxsize=1)
def _cargar_tensorflow():
    try:
        import tensorflow as tf
        return tf, tf.__version__, None
    except Exception as exc:  # pragma: no cover - depende del entorno
        return None, None, f"{type(exc).__name__}: {exc}"


# ------------------------------------------------------------
# 5. ENTRENAMIENTO (mismo flujo que los scripts originales)
# ------------------------------------------------------------

def entrenar_modelo(spec: ModelSpec) -> ModelBundle:
    """Entrena el KNN y calcula las metricas reales del modelo."""
    df = pd.DataFrame(spec.datos)

    columnas_esperadas = spec.feature_keys + [spec.target]
    if list(df.columns) != columnas_esperadas:
        raise ValueError(
            "Las columnas del dataset no coinciden con las variables del modelo. "
            f"Esperadas: {columnas_esperadas}"
        )

    # 1. Separar variables
    X = df[spec.feature_keys]
    y = df[spec.target]

    # 2. Dividir datos
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 3. Normalizar con un unico escalador ajustado en entrenamiento
    escalador = StandardScaler()
    X_train = escalador.fit_transform(X_train_df)
    X_test = escalador.transform(X_test_df)

    # 4. Complemento TensorFlow
    tf, tf_version, tf_error = _cargar_tensorflow()
    tensor_forma = None
    if tf is not None:
        tensor_datos = tf.constant(X_train, dtype=tf.float32)
        tensor_forma = tuple(int(d) for d in tensor_datos.shape)

    # 5. Entrenar el modelo KNN
    modelo = KNeighborsClassifier(n_neighbors=K_VECINOS)
    modelo.fit(X_train, y_train)

    # 6. Predecir y evaluar sobre el conjunto de prueba
    predicciones = modelo.predict(X_test)
    exactitud = float(accuracy_score(y_test, predicciones))
    matriz = confusion_matrix(y_test, predicciones, labels=list(modelo.classes_))
    reporte = classification_report(
        y_test, predicciones,
        output_dict=True,
        zero_division=0,
    )

    return ModelBundle(
        spec=spec,
        df=df,
        X_train_df=X_train_df,
        X_test_df=X_test_df,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        escalador=escalador,
        modelo=modelo,
        predicciones=predicciones,
        exactitud=exactitud,
        matriz=matriz,
        reporte=reporte,
        tensor_forma=tensor_forma,
        tensorflow_version=tf_version,
        tensorflow_error=tf_error,
    )


# ------------------------------------------------------------
# 6. PREDICCION DE UN REGISTRO NUEVO
# ------------------------------------------------------------

def predecir(bundle: ModelBundle, valores: dict[str, int]) -> dict[str, Any]:
    """Aplica el escalador y el modelo entrenados, y obtiene los 3 vecinos.

    Las distancias se calculan con TODAS las caracteristicas del modelo,
    aunque el grafico muestre solo dos dimensiones.
    """
    spec = bundle.spec
    claves = spec.feature_keys

    if set(valores) != set(claves):
        raise ValueError("Faltan variables o hay variables adicionales en la entrada.")

    valores_limpios: dict[str, int] = {}
    for feature in spec.features:
        valor = int(valores[feature.key])
        if not feature.min_value <= valor <= feature.max_value:
            raise ValueError(
                f"{feature.label} debe estar entre {feature.min_value} y "
                f"{feature.max_value}. Valor recibido: {valor}."
            )
        valores_limpios[feature.key] = valor

    # Mismo orden de columnas que espera el modelo.
    entrada = pd.DataFrame([valores_limpios], columns=claves)
    entrada_escalada = bundle.escalador.transform(entrada)

    clase = int(bundle.modelo.predict(entrada_escalada)[0])
    probabilidades = bundle.modelo.predict_proba(entrada_escalada)[0]

    distancias, indices = bundle.modelo.kneighbors(
        entrada_escalada,
        n_neighbors=K_VECINOS,
    )

    vecinos = []
    for posicion, indice in enumerate(indices[0]):
        indice = int(indice)
        clase_vecino = int(bundle.y_train.iloc[indice])
        fila_original = bundle.X_train_df.iloc[indice]
        vecinos.append({
            "orden": posicion + 1,
            "indice_entrenamiento": indice,
            "clase": clase_vecino,
            "etiqueta": spec.etiquetas[clase_vecino],
            "distancia": float(distancias[0][posicion]),
            "vota_por": spec.etiquetas[clase_vecino],
            "x": float(bundle.X_train[indice, 0]),
            "y": float(bundle.X_train[indice, 1]),
            "caracteristicas": {
                feature.key: int(fila_original[feature.key])
                for feature in spec.features
            },
        })

    votos = {
        int(clase_modelo): sum(1 for v in vecinos if v["clase"] == int(clase_modelo))
        for clase_modelo in bundle.modelo.classes_
    }

    advertencias = []
    for feature in spec.features:
        minimo = int(bundle.df[feature.key].min())
        maximo = int(bundle.df[feature.key].max())
        if not minimo <= valores_limpios[feature.key] <= maximo:
            advertencias.append(
                f"{feature.label}: {valores_limpios[feature.key]} está fuera del "
                f"rango observado en el dataset ({minimo} a {maximo})."
            )

    return {
        "clase": clase,
        "etiqueta": spec.etiquetas[clase],
        "descripcion": (
            spec.resultado_positivo if clase == 1 else spec.resultado_negativo
        ),
        "probabilidades": {
            int(clase_modelo): float(probabilidad)
            for clase_modelo, probabilidad in zip(
                bundle.modelo.classes_, probabilidades
            )
        },
        "confianza_votacion": votos[clase] / len(vecinos),
        "vecinos": vecinos,
        "votos": votos,
        "valores": dict(valores_limpios),
        "valores_escalados": {
            feature.key: float(entrada_escalada[0, posicion])
            for posicion, feature in enumerate(spec.features)
        },
        "punto_nuevo": {
            "x": float(entrada_escalada[0, 0]),
            "y": float(entrada_escalada[0, 1]),
        },
        "advertencias": advertencias,
    }


# ------------------------------------------------------------
# 7. TABLAS DERIVADAS PARA LA INTERFAZ
# ------------------------------------------------------------

def tabla_reporte(bundle: ModelBundle) -> pd.DataFrame:
    """Convierte el classification_report en una tabla legible."""
    filas = []
    for etiqueta, valores in bundle.reporte.items():
        if not isinstance(valores, dict):
            continue
        if etiqueta in ("accuracy",):
            continue
        if etiqueta in ("macro avg", "weighted avg"):
            nombre = "Promedio macro" if etiqueta == "macro avg" else "Promedio ponderado"
        else:
            clase = int(float(etiqueta))
            nombre = f"Clase {clase} · {bundle.spec.etiquetas[clase]}"
        filas.append({
            "Clase": nombre,
            "Precisión": valores["precision"],
            "Recall": valores["recall"],
            "F1-score": valores["f1-score"],
            "Soporte": int(valores["support"]),
        })
    return pd.DataFrame(filas)


def tabla_dataset(bundle: ModelBundle) -> pd.DataFrame:
    """Dataset completo con la marca de entrenamiento o prueba."""
    vista = bundle.df.copy()
    vista["conjunto"] = "Prueba"
    vista.loc[bundle.X_train_df.index, "conjunto"] = "Entrenamiento"
    vista = vista.rename(columns={
        **{f.key: f.label for f in bundle.spec.features},
        bundle.spec.target: "Clase real",
        "conjunto": "Conjunto",
    })
    return vista


def tabla_vecinos(resultado: dict[str, Any]) -> pd.DataFrame:
    """Tabla de vecinos con clase, distancia y voto."""
    return pd.DataFrame([
        {
            "Vecino": f"Vecino {v['orden']}",
            "Clase": v["clase"],
            "Vota por": v["etiqueta"],
            "Distancia": v["distancia"],
        }
        for v in resultado["vecinos"]
    ])
