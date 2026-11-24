# KNN Predict Lab

Interfaz web interactiva para ejecutar los dos modelos KNN del proyecto de
Machine Learning:

- **Satisfacción del cliente:** clasifica si un cliente está satisfecho o insatisfecho.
- **Inversión empresarial:** estima si un empresario invertirá en un software de
  automatización.

La aplicación reutiliza los datasets, el escalador y los modelos de los scripts
originales. No crea una implementación paralela de KNN ni simula resultados.

## Requisitos

- Python 3.11
- pandas, numpy, matplotlib, scikit-learn, TensorFlow y Streamlit
  (ver `requirements.txt`)

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
streamlit run app.py
```

Abrir la URL que muestra la terminal, normalmente
`http://localhost:8501`.

## Estructura

| Archivo | Descripción |
|---|---|
| `app.py` | Interfaz Streamlit: navegación, formularios, resultados, gráficos y evaluación. |
| `knn_core.py` | Datasets, entrenamiento, escalador, métricas y cálculo de vecinos. |
| `.streamlit/config.toml` | Tema visual con la paleta del laboratorio. |
| `requirements.txt` | Dependencias del proyecto. |
| `knn_satisfaccion.py` | Script original de satisfacción (sin modificaciones). |
| `knn_inversion.py` | Script original de inversión (sin modificaciones). |

## Modelos y variables

Ambos modelos usan `KNeighborsClassifier(n_neighbors=3)`, una división
`test_size=0.25`, `random_state=42`, `stratify=y` y un `StandardScaler`
ajustado únicamente con los 22 registros de entrenamiento.

### Satisfacción del cliente

Orden real de las variables:

1. `calidad_servicio` (1 a 5)
2. `atencion_cliente` (1 a 5)
3. `tiempo_espera` (1 a 5)
4. `precio` (1 a 5)

Clases: `0 = Insatisfecho`, `1 = Satisfecho`.

### Inversión empresarial

Orden real de las variables:

1. `presupuesto` (1 a 5)
2. `trabajadores` (el dataset contiene entre 2 y 30)
3. `procesos_manuales` (1 a 5)
4. `costos_operativos` (1 a 5)

Clases: `0 = No invierte`, `1 = Sí invierte`.

## Verificación incluida

Con los valores de ejemplo de los scripts originales:

- Satisfacción: resultado **Satisfecho**, votos `3 a 0`, distancias
  `0.6783`, `0.9583`, `0.9583`.
- Inversión: resultado **Sí invierte**, votos `3 a 0`, distancias
  `0.7237`, `0.8026`, `0.9264`.
- Exactitud en prueba: **100.00 %** (`8 de 8` aciertos).
- Matriz de confusión: `[[3, 0], [0, 5]]`.
- Dataset: **30 registros**; **22** de entrenamiento y **8** de prueba.

Estos valores se calculan en tiempo de ejecución; no están escritos a mano en
la interfaz.

## Nota sobre TensorFlow

TensorFlow se mantiene como complemento de procesamiento: el conjunto de
entrenamiento normalizado se convierte en un tensor `float32` de forma
`(22, 4)`, igual que en los scripts. TensorFlow **no** implementa el
clasificador; el modelo KNN es de scikit-learn.

## Notas de uso

- El modelo y el escalador se entrenan una sola vez por proceso y se reutilizan
  en cada predicción.
- Los gráficos de dispersión muestran dos variables normalizadas por claridad.
  Las distancias se calculan con las cuatro variables del modelo.
- Varios registros del dataset comparten coordenadas en las dos primeras
  variables, por lo que algunos puntos se superponen en el gráfico.
- Algunos vecinos pueden tener distancia `0.0000` si el registro ingresado
  coincide exactamente con filas del dataset original. Es un resultado real
  del conjunto de datos, no un error.
