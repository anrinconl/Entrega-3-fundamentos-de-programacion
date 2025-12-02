# main.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp


def analisis_poisson():
    """
    Regresión de Poisson para evaluar cómo la expresión de DYRK1A_N
    se asocia al número de proteínas altamente expresadas.
    """

    # Cargar datos y separar metadatos de proteínas
    df = pd.read_excel("Data_Cortex_Nuclear.xls")
    meta_cols = ["MouseID", "Genotype", "Treatment", "Behavior", "class"]
    protein_cols = [c for c in df.columns if c not in meta_cols]

    # Se eliminan filas incompletas para asegurar análisis válido
    df_clean = df.dropna(subset=meta_cols + protein_cols)

    # Normalización por z-score y conteo de proteínas con alta expresión
    prot_z = (df_clean[protein_cols] - df_clean[protein_cols].mean()) / df_clean[protein_cols].std()
    df_clean["high_protein_count"] = (prot_z > 1.0).sum(axis=1)

    # Selección de la proteína clave del análisis
    protein = "DYRK1A_N"
    if protein not in df_clean.columns:
        raise ValueError(f"{protein} no está en el dataset.")

    # Modelo Poisson usando la expresión de la proteína como predictor continuo
    poisson_model_prot = smf.glm(
        formula=f"high_protein_count ~ {protein}",
        data=df_clean,
        family=sm.families.Poisson()
    ).fit()

    # Secuencia de valores para generar la curva del modelo
    x = np.linspace(df_clean[protein].min(), df_clean[protein].max(), 100)
    y_pred = poisson_model_prot.predict(pd.DataFrame({protein: x}))

    # Gráfica del efecto directo de la proteína
    plt.figure(figsize=(8, 5))
    plt.scatter(df_clean[protein], df_clean["high_protein_count"], alpha=0.4, label="Datos reales")
    plt.plot(x, y_pred, color="red", linewidth=2, label="Modelo Poisson")
    plt.xlabel(f"Expresión {protein}")
    plt.ylabel("Conteo proteínas")
    plt.title(f"{protein} vs conteo")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Modelo Poisson con interacción proteína × genotipo
    modelo_interaccion = smf.glm(
        formula=f"high_protein_count ~ {protein} * Genotype",
        data=df_clean,
        family=sm.families.Poisson()
    ).fit()

    # Curvas de predicción separadas por genotipo
    genotypes = df_clean["Genotype"].unique()

    plt.figure(figsize=(8, 5))
    for gen in genotypes:
        pred = modelo_interaccion.predict(pd.DataFrame({protein: x, "Genotype": gen}))
        plt.plot(x, pred, linewidth=2, label=f"{gen}")

    # Datos reales superpuestos
    plt.scatter(df_clean[protein], df_clean["high_protein_count"], alpha=0.3, color='gray')
    plt.xlabel(f"Expresión {protein}")
    plt.ylabel("Conteo proteínas")
    plt.title("Interacción DYRK1A_N × Genotipo")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()



def modelo_logistico(x, K, A, B):
    # Curva logística usada para modelar respuestas no lineales
    return K / (1 + A * np.exp(-B * x))


def ajuste_modelo_biologico():
    """
    Ajusta un modelo logístico para describir cómo cambia el conteo de proteínas
    según la expresión de DYRK1A_N.
    """

    # Cargar y limpiar datos
    df = pd.read_excel("Data_Cortex_Nuclear.xls")
    meta_cols = ["MouseID", "Genotype", "Treatment", "Behavior", "class"]
    protein_cols = [c for c in df.columns if c not in meta_cols]
    df_clean = df.dropna(subset=meta_cols + protein_cols)

    # Normalizar proteínas y calcular cuántas están altamente expresadas
    prot_z = (df_clean[protein_cols] - df_clean[protein_cols].mean()) / df_clean[protein_cols].std()
    df_clean["high_protein_count"] = (prot_z > 1.0).sum(axis=1)

    # Variable explicativa principal
    protein = "DYRK1A_N"
    x_data = df_clean[protein].values
    y_data = df_clean["high_protein_count"].values

    # Ajuste de la curva logística
    popt, _ = curve_fit(modelo_logistico, x_data, y_data,
                        p0=[max(y_data), 1, 0.1], maxfev=5000)
    K, A, B = popt  # parámetros ajustados

    # Curva suavizada para graficar
    x_curve = np.linspace(min(x_data), max(x_data), 100)
    y_curve = modelo_logistico(x_curve, K, A, B)

    # Gráfica final
    plt.figure(figsize=(8, 5))
    plt.scatter(x_data, y_data, alpha=0.4, label="Datos reales")
    plt.plot(x_curve, y_curve, color="green", linewidth=2, label="Ajuste logístico")
    plt.xlabel(f"Expresión {protein}")
    plt.ylabel("Proteínas expresadas")
    plt.title("Modelo logístico")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()



def analisis_dinamico():
    """
    Sistema dinámico X–Y–Z: un regulador (X), una proteína diana (Y)
    y un inhibidor (Z) interactuando en el tiempo.
    """

    # Definimos cómo cambia cada molécula según la regulación, activación e inhibición
    def sistema(t, vars, k1, d1, k2, K, d2, k3, alpha, d3):
        X, Y, Z = vars
        return [
            k1 / (1 + alpha * Z) - d1 * X,           # X se produce y es inhibida por Z
            (k2 * X / (K + X)) - d2 * Y,             # Y depende de X pero también se degrada
            k3 * Y - d3 * Z                          # Z aumenta por Y y luego se degrada
        ]

    # Parámetros del sistema y resolución numérica
    params = (1, 0.3, 2, 1, 0.4, 1, 1.5, 0.2)
    t_eval = np.linspace(0, 50, 500)
    sol = solve_ivp(lambda t, y: sistema(t, y, *params), (0, 50), [0.1, 0, 0], t_eval=t_eval)

    X, Y, Z = sol.y

    # Gráfica temporal de los tres componentes
    plt.figure(figsize=(8, 5))
    plt.plot(sol.t, X, label="X")
    plt.plot(sol.t, Y, label="Y")
    plt.plot(sol.t, Z, label="Z")
    plt.xlabel("Tiempo")
    plt.ylabel("Nivel molecular")
    plt.title("Dinámica X–Y–Z")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def main():
    print(">>> PROGRAMA INICIADO <<<")
    while True:
        print("\n1. Regresión de Poisson")
        print("2. Ajuste modelo biológico")
        print("3. Análisis dinámico")
        print("0. Salir")

        opcion = input("Selecciona: ")
        if opcion == "1":
            analisis_poisson()
        elif opcion == "2":
            ajuste_modelo_biologico()
        elif opcion == "3":
            analisis_dinamico()
        elif opcion == "0":
            print("Fin del programa.")
            break
        else:
            print("Opción inválida.")


if __name__ == "__main__":
    main()
