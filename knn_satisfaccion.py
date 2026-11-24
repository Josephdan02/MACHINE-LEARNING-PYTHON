
# ============================================
# KNN: PREDICCION DE SATISFACCION DEL CLIENTE
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

df = pd.DataFrame(datos)

print("DATOS DE CLIENTES:")
print(df.head(10))

# 2. SEPARAR VARIABLES
X = df.drop('satisfaccion', axis=1)
y = df['satisfaccion']

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

# 9. PREDICCION DE UN NUEVO CLIENTE
nuevo_cliente = pd.DataFrame([{
    'calidad_servicio': 5,
    'atencion_cliente': 4,
    'tiempo_espera': 1,
    'precio': 4
}])

nuevo_cliente_escalado = escalador.transform(nuevo_cliente)

resultado = modelo.predict(nuevo_cliente_escalado)[0]

print("\nPREDICCION PARA NUEVO CLIENTE:")

if resultado == 1:
    print("El cliente esta SATISFECHO.")
else:
    print("El cliente esta INSATISFECHO.")

# 10. GRAFICO DE RESULTADOS
conteo = df['satisfaccion'].value_counts().sort_index()

plt.bar(['Insatisfecho', 'Satisfecho'],
        conteo.values)

plt.title('Nivel de satisfaccion de los clientes')
plt.xlabel('Estado del cliente')
plt.ylabel('Cantidad de clientes')
plt.show()

# Grafico de predicciones
plt.figure(figsize=(8, 4))
plt.scatter(range(len(y_test)), y_test,
            label='Valores reales', marker='o')
plt.scatter(range(len(predicciones)), predicciones,
            label='Predicciones KNN', marker='x')

plt.title('Comparacion: resultados reales vs predicciones')
plt.xlabel('Cliente de prueba')
plt.ylabel('Satisfaccion (0 o 1)')
plt.legend()
plt.grid()
plt.show()


# ============================================
# 11. GRAFICO DE LOS 3 VECINOS MAS CERCANOS
# ============================================

# Obtener los 3 vecinos mas cercanos
distancias, indices = modelo.kneighbors(
    nuevo_cliente_escalado,
    n_neighbors=3
)

# Crear grafico
plt.figure(figsize=(10, 7))

# Mostrar clientes de entrenamiento por clase
for clase, nombre, color in [
    (0, 'Insatisfecho', 'blue'),
    (1, 'Satisfecho', 'orange')
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

# Marcar el nuevo cliente
plt.scatter(
    nuevo_cliente_escalado[0, 0],
    nuevo_cliente_escalado[0, 1],
    color='red',
    marker='*',
    s=300,
    label='Nuevo cliente',
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

    # Conectar el nuevo cliente con su vecino
    plt.plot(
        [nuevo_cliente_escalado[0, 0], vecino_x],
        [nuevo_cliente_escalado[0, 1], vecino_y],
        'g--',
        alpha=0.8
    )

    plt.annotate(
        f'Vecino {i+1}',
        (vecino_x, vecino_y),
        xytext=(5, 8),
        textcoords='offset points'
    )

plt.title('KNN: Los 3 vecinos mas cercanos del cliente')
plt.xlabel('Calidad del servicio (normalizada)')
plt.ylabel('Atencion al cliente (normalizada)')
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
