
# ============================================
# KNN: PREDICCION DE INVERSION EMPRESARIAL
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. CREAR CONJUNTO DE DATOS
datos = {
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

df = pd.DataFrame(datos)

print("DATOS DE EMPRESARIOS:")
print(df.head(10))

# 2. SEPARAR VARIABLES
X = df.drop('invierte', axis=1)
y = df['invierte']

# 3. DIVIDIR DATOS
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 4. NORMALIZAR LOS DATOS
escalador = StandardScaler()

X_train = escalador.fit_transform(X_train)
X_test = escalador.transform(X_test)

# 5. COMPLEMENTO CON TENSORFLOW
tensor_datos = tf.constant(X_train, dtype=tf.float32)

print("\nTensorFlow - Forma de los datos:", tensor_datos.shape)

# 6. CREAR Y ENTRENAR MODELO KNN
modelo = KNeighborsClassifier(n_neighbors=3)

modelo.fit(X_train, y_train)

# 7. REALIZAR PREDICCIONES
predicciones = modelo.predict(X_test)

# 8. EVALUAR MODELO
precision = accuracy_score(y_test, predicciones)

print("\nRESULTADOS DEL MODELO")
print("Precision:", round(precision * 100, 2), "%")
print("\nMatriz de confusion:")
print(confusion_matrix(y_test, predicciones))
print("\nReporte de clasificacion:")
print(classification_report(y_test, predicciones,
                            zero_division=0))

# 9. PREDICCION PARA UN NUEVO EMPRESARIO
nuevo_empresario = pd.DataFrame([{
    'presupuesto': 5,
    'trabajadores': 25,
    'procesos_manuales': 5,
    'costos_operativos': 4
}])

nuevo_empresario_escalado = escalador.transform(
    nuevo_empresario
)

resultado = modelo.predict(nuevo_empresario_escalado)[0]

print("\nPREDICCION PARA NUEVO EMPRESARIO:")

if resultado == 1:
    print("El empresario SI INVERTIRA en el software.")
else:
    print("El empresario NO INVERTIRA en el software.")

# 10. GRAFICO DE RESULTADOS
conteo = df['invierte'].value_counts().sort_index()

plt.bar(['No invierte', 'Si invierte'],
        conteo.values)

plt.title('Prediccion de inversion empresarial')
plt.xlabel('Decision empresarial')
plt.ylabel('Cantidad de empresarios')
plt.show()

# Grafico de predicciones
plt.figure(figsize=(8, 4))

plt.scatter(range(len(y_test)), y_test,
            label='Valores reales', marker='o')

plt.scatter(range(len(predicciones)), predicciones,
            label='Predicciones KNN', marker='x')

plt.title('Comparacion: resultados reales vs predicciones')
plt.xlabel('Empresario de prueba')
plt.ylabel('Inversion (0 o 1)')
plt.legend()
plt.grid()
plt.show()


# ============================================
# 11. GRAFICO DE LOS 3 VECINOS MAS CERCANOS
# ============================================

# Obtener los 3 vecinos mas cercanos
distancias, indices = modelo.kneighbors(
    nuevo_empresario_escalado,
    n_neighbors=3
)

# Crear grafico
plt.figure(figsize=(10, 7))

# Mostrar empresarios de entrenamiento por clase
for clase, nombre, color in [
    (0, 'No invierte', 'blue'),
    (1, 'Si invierte', 'orange')
]:
    mascara = np.asarray(y_train) == clase

    plt.scatter(
        X_train[mascara, 0],
        X_train[mascara, 1],
        color=color,
        label=nombre,
        alpha=0.7,
        s=70
    )

# Marcar al nuevo empresario
plt.scatter(
    nuevo_empresario_escalado[0, 0],
    nuevo_empresario_escalado[0, 1],
    color='red',
    marker='*',
    s=300,
    label='Nuevo empresario',
    edgecolors='black'
)

# Identificar y resaltar los 3 vecinos
for i, indice in enumerate(indices[0]):

    vecino_x = X_train[indice, 0]
    vecino_y = X_train[indice, 1]

    plt.scatter(
        vecino_x,
        vecino_y,
        s=250,
        facecolors='none',
        edgecolors='green',
        linewidths=2
    )

    # Conectar el empresario con sus vecinos
    plt.plot(
        [nuevo_empresario_escalado[0, 0], vecino_x],
        [nuevo_empresario_escalado[0, 1], vecino_y],
        'g--',
        alpha=0.8
    )

    plt.annotate(
        f'Vecino {i+1}',
        (vecino_x, vecino_y),
        xytext=(5, 8),
        textcoords='offset points'
    )

plt.title('KNN: Los 3 vecinos mas cercanos del empresario')
plt.xlabel('Presupuesto (normalizado)')
plt.ylabel('Numero de trabajadores (normalizado)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Mostrar distancias y clases de los vecinos
print("\nLOS 3 VECINOS MAS CERCANOS:")

for i, indice in enumerate(indices[0]):
    print(
        f"Vecino {i+1}: "
        f"Clase = {y_train.iloc[indice]}, "
        f"Distancia = {distancias[0][i]:.4f}"
    )
