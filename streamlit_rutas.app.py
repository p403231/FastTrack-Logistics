
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from scipy.stats import f_oneway, t


# =====================================
# 1. CONFIGURACIÓN Y CARGA DE DATOS
# =====================================

st.set_page_config(
    page_title="FastTrack Logistics",
    page_icon="🚚",
    layout="wide"
)

carpeta = Path(__file__).resolve().parent
ruta_csv = carpeta / "datos_logistica_transporte.csv"

df = pd.read_csv(ruta_csv, encoding="latin-1")


# =====================================
# 2. REGRESIÓN LINEAL MÚLTIPLE
# =====================================

X1 = df["Peso_Carga_Ton"].to_numpy()
X2 = df["Velocidad_Promedio_kmh"].to_numpy()
Y = df["Consumo_Litros"].to_numpy()

# Medias
media_X1 = np.mean(X1)
media_X2 = np.mean(X2)
media_Y = np.mean(Y)

# Sumatorias
s1 = np.sum((X1 - media_X1)**2)
s2 = np.sum((X2 - media_X2)**2)
s3 = np.sum((X1 - media_X1) * (X2 - media_X2))
s4 = np.sum((X1 - media_X1) * (Y - media_Y))
s5 = np.sum((X2 - media_X2) * (Y - media_Y))

# Coeficientes
denominador = s1 * s2 - s3**2

B1 = (s4 * s2 - s5 * s3) / denominador
B2 = (s5 * s1 - s4 * s3) / denominador
B0 = media_Y - B1 * media_X1 - B2 * media_X2


# =====================================
# 3. MÉTRICAS DE REGRESIÓN
# =====================================

Y_pred = B0 + B1 * X1 + B2 * X2
residuales = Y - Y_pred

MSE = np.mean(residuales**2)
RMSE = np.sqrt(MSE)

SS_res = np.sum(residuales**2)
SS_total = np.sum((Y - media_Y)**2)

R2 = 1 - SS_res / SS_total


# =====================================
# 4. SIGNIFICANCIA DE COEFICIENTES
# =====================================

alpha = 0.05
n = len(Y)
gl = n - 3

MSE_residual = SS_res / gl

SE_B1 = np.sqrt(MSE_residual * s2 / denominador)
SE_B2 = np.sqrt(MSE_residual * s1 / denominador)

t_B1 = B1 / SE_B1
t_B2 = B2 / SE_B2

t_critica = t.ppf(1 - alpha / 2, gl)

p_B1 = 2 * t.sf(abs(t_B1), gl)
p_B2 = 2 * t.sf(abs(t_B2), gl)

tabla_coeficientes = pd.DataFrame({
    "Variable": ["Peso de carga", "Velocidad promedio"],
    "Coeficiente": [B1, B2],
    "t calculada": [t_B1, t_B2],
    "Valor-p": [p_B1, p_B2],
    "Decisión": [
        "Se rechaza H0" if p_B1 < alpha else "No se rechaza H0",
        "Se rechaza H0" if p_B2 < alpha else "No se rechaza H0"
    ]
})


# =====================================
# 5. ANOVA DE UN FACTOR
# =====================================

variables_anova = [
    "Consumo_Litros",
    "Peso_Carga_Ton",
    "Velocidad_Promedio_kmh"
]

resultados = []

for variable in variables_anova:

    grupos = [
        grupo[variable].dropna().values
        for _, grupo in df.groupby("Tipo_Ruta")
    ]

    F, p_valor = f_oneway(*grupos)

    decision = (
        "Se rechaza H0"
        if p_valor < alpha
        else "No se rechaza H0"
    )

    resultados.append({
        "Variable": variable,
        "Estadístico F": F,
        "Valor-p": p_valor,
        "Decisión": decision
    })

tabla_anova = pd.DataFrame(resultados)


# =====================================
# 6. ESTADÍSTICAS POR RUTA
# =====================================

consumo_ruta = (
    df.groupby("Tipo_Ruta")["Consumo_Litros"].mean()
)

df["Litros_por_Tonelada"] = (
    df["Consumo_Litros"] / df["Peso_Carga_Ton"]
)

consumo_ton = (
    df.groupby("Tipo_Ruta")["Litros_por_Tonelada"].mean()
)

variables = df[
    ["Peso_Carga_Ton",
     "Velocidad_Promedio_kmh",
     "Consumo_Litros"]
]

correlacion = variables.corr(method="pearson")



# =====================================
# FACTORES DE CORRECCIÓN POR RUTA
# =====================================

# Corregir nombre de montaña
df["Tipo_Ruta"] = df["Tipo_Ruta"].replace({
    "Monta¤a": "Montaña"
})

# Predicción original para cada viaje
df["Prediccion_Original"] = (
    B0
    + B1 * df["Peso_Carga_Ton"]
    + B2 * df["Velocidad_Promedio_kmh"]
)

# Calcular promedios por ruta
comparacion_rutas = df.groupby("Tipo_Ruta").agg(
    Consumo_Real=("Consumo_Litros", "mean"),
    Consumo_Predicho=("Prediccion_Original", "mean")
)

# Calcular factores de corrección
comparacion_rutas["Factor"] = (
    comparacion_rutas["Consumo_Real"]
    / comparacion_rutas["Consumo_Predicho"]
)




# =====================================
# 7. MENÚ LATERAL
# =====================================

pagina = st.sidebar.radio(
    "Selecciona una sección:",
    [
        "Inicio",
        "Análisis descriptivo",
        "Análisis estadístico",
        "Regresión lineal",
        "Simulador",
        "Toma de decisiones"
    ]
)


# =====================================
# 8. INICIO
# =====================================

if pagina == "Inicio":

    st.title("FastTrack Logistics")

    st.subheader(
        "Sistema de análisis y predicción de combustible"
    )

    st.write (" FastTrack Logistics es una empresa de transporte de carga terrestre de mediano tamaño que opera en el centro del país," 
    " la empresa se especializa en entregas Just-in-Time (Justo a Tiempo) para el sector industrial. Durante el último trimestre fiscal,"
    " el Director Financiero ha notado una tendencia alarmante: los costos operativos han subido un 18%, impulsados casi exclusivamente por el gasto en combustible (Diésel). "
    "La empresa quiere entender qué factores influyen más en el consumo de combustible de su flota de camiones.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Viajes analizados", len(df))

    with col2:
        st.metric("Tipos de ruta", df["Tipo_Ruta"].nunique())

    with col3:
        st.metric(
            "Consumo promedio",
            f"{df['Consumo_Litros'].mean():.2f} L"
        )

    st.subheader("Objetivos del proyecto")

    st.write(
        "- Comparar el consumo entre tipos de ruta.\n"
        "- Analizar las relaciones entre peso, velocidad y consumo.\n"
        "- Evaluar la significancia estadística.\n"
        "- Predecir el consumo mediante regresión lineal múltiple."
    )


# =====================================
# 9. ANÁLISIS DESCRIPTIVO
# =====================================



elif pagina == "Análisis descriptivo":

    st.title("Análisis descriptivo")

    # GRÁFICA 1
    st.subheader("Consumo promedio por tipo de ruta")

    fig, ax = plt.subplots(figsize=(8, 4))

    barras = ax.bar(
        consumo_ruta.index,
        consumo_ruta.values,
        color=["#16A085", "#E67E22", "#3B82F6"],
        width=0.5
    )

    ax.bar_label(barras, fmt="%.2f L", padding=5)

    ax.set_xlabel("Tipo de ruta")
    ax.set_ylabel("Consumo promedio (L)")
    ax.set_title("Consumo promedio de combustible por ruta")
    ax.set_ylim(0, consumo_ruta.max() * 1.2)

    st.pyplot(fig)
    plt.close(fig)

    # GRÁFICA 2
    st.subheader("Consumo promedio por tonelada")

    fig, ax = plt.subplots(figsize=(8, 4))

    barras = ax.bar(
        consumo_ton.index,
        consumo_ton.values,
        color=["#16A085", "#E67E22", "#3B82F6"],
        width=0.5
    )

    ax.bar_label(barras, fmt="%.2f L/ton", padding=5)

    ax.set_xlabel("Tipo de ruta")
    ax.set_ylabel("Litros por tonelada")
    ax.set_title("Consumo promedio por tonelada según ruta")
    ax.set_ylim(0, consumo_ton.max() * 1.2)

    st.pyplot(fig)
    plt.close(fig)


    # =====================================
    # VELOCIDAD PROMEDIO POR TIPO DE RUTA
    # =====================================

    st.subheader("Velocidad promedio por tipo de ruta")

    velocidad_ruta = df.groupby("Tipo_Ruta")[
        "Velocidad_Promedio_kmh"
    ].mean()

    fig, ax = plt.subplots(figsize=(8, 4))

    barras = ax.bar(
        velocidad_ruta.index,
        velocidad_ruta.values,
        color=["#16A085", "#E67E22", "#3B82F6"],
        width=0.5
    )

    ax.bar_label(
        barras,
        fmt="%.2f km/h",
        padding=5
    )

    ax.set_xlabel("Tipo de ruta")
    ax.set_ylabel("Velocidad promedio (km/h)")
    ax.set_title("Velocidad promedio por tipo de ruta")

    ax.set_ylim(0, velocidad_ruta.max() * 1.2)

    st.pyplot(fig)
    plt.close(fig)




    # =====================================
    # PESO PROMEDIO POR TIPO DE RUTA
    # =====================================

    st.subheader("Peso promedio de carga por tipo de ruta")

    peso_ruta = df.groupby("Tipo_Ruta")[
        "Peso_Carga_Ton"
    ].mean()

    fig, ax = plt.subplots(figsize=(8, 4))

    barras = ax.bar(
        peso_ruta.index,
        peso_ruta.values,
        color=["#16A085", "#E67E22", "#3B82F6"],
        width=0.5
    )

    ax.bar_label(
        barras,
        fmt="%.2f ton",
        padding=5
    )

    ax.set_xlabel("Tipo de ruta")
    ax.set_ylabel("Peso promedio (toneladas)")
    ax.set_title("Peso promedio transportado por tipo de ruta")

    ax.set_ylim(0, peso_ruta.max() * 1.2)

    st.pyplot(fig)
    plt.close(fig)




    # =====================================
    # MEDIDAS DE TENDENCIA CENTRAL Y DISPERSIÓN
    # =====================================

    st.subheader("Medidas de tendencia central y dispersión")

    # Seleccionar variable
    variable = st.selectbox(
        "Selecciona la variable que deseas analizar:",
        [
            "Peso_Carga_Ton",
            "Velocidad_Promedio_kmh",
            "Consumo_Litros"
        ],
        format_func=lambda x: {
            "Peso_Carga_Ton": "Peso de carga (toneladas)",
            "Velocidad_Promedio_kmh": "Velocidad promedio (km/h)",
            "Consumo_Litros": "Consumo de combustible (L)"
        }[x]
    )

    # Función para calcular la moda
    def calcular_moda(datos):
        modas = datos.mode()

        # Si todos los valores son únicos, no hay moda
        if datos.dropna().value_counts().max() == 1:
            return np.nan

        return modas.iloc[0] if not modas.empty else np.nan

    # Agrupar por tipo de ruta
    grupos = df.groupby("Tipo_Ruta")[variable]

    # Calcular estadísticas
    tabla_estadisticas = grupos.agg(
        Media="mean",
        Mediana="median",
        Varianza="var",
        Desviacion="std",
        Minimo="min",
        Maximo="max"
    )

    # Calcular rango
    tabla_estadisticas["Rango"] = (
        tabla_estadisticas["Maximo"]
        - tabla_estadisticas["Minimo"]
    )

    # Renombrar columnas
    tabla_estadisticas = tabla_estadisticas.rename(columns={
        "Desviacion": "Desviación estándar",
        "Minimo": "Mínimo",
        "Maximo": "Máximo"
    })

    # Mostrar resultados
    st.dataframe(
        tabla_estadisticas.round(2),
        use_container_width=True
    )




# =====================================
# 10. ANÁLISIS ESTADÍSTICO
# =====================================

elif pagina == "Análisis estadístico":

    st.title("Análisis estadístico")

    # PEARSON
    st.subheader("Matriz de correlación de Pearson")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.heatmap(
        correlacion,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        ax=ax
    )

    ax.set_title("Correlación entre peso, velocidad y consumo")

    st.pyplot(fig)
    plt.close(fig)

    # ANOVA
    st.subheader("ANOVA de un factor")

    st.dataframe(
        tabla_anova,
        hide_index=True,
        column_config={
            "Estadístico F": st.column_config.NumberColumn(
                format="%.4f"
            ),
            "Valor-p": st.column_config.NumberColumn(
                format="%.3e"
            )
        }
    )

    st.write("Nivel de significancia: α = 0.05")


# =====================================
# 11. REGRESIÓN LINEAL
# =====================================

elif pagina == "Regresión lineal":

    st.title("Regresión lineal múltiple")

    st.subheader("Ecuación del modelo")

    st.latex(
        rf"\hat{{Y}} = {B0:.2f} + ({B1:.2f})X_1 + ({B2:.2f})X_2"
    )

    st.write("X₁: Peso de carga (toneladas)")
    st.write("X₂: Velocidad promedio (km/h)")

    st.subheader("Evaluación del modelo")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("MSE", f"{MSE:.2f}")

    with col2:
        st.metric("RMSE", f"{RMSE:.2f} L")

    with col3:
        st.metric("R²", f"{R2:.4f}")


    st.subheader("Significancia de los coeficientes")

    st.dataframe(
        tabla_coeficientes,
        hide_index=True,
        column_config={
            "Coeficiente": st.column_config.NumberColumn(
                format="%.4f"
            ),
            "t calculada": st.column_config.NumberColumn(
                format="%.4f"
            ),
            "Valor-p": st.column_config.NumberColumn(
                format="%.3e"
            )
        }
    )

    st.write(f"Valor t crítico: ±{t_critica:.4f}")


# =====================================
# 12. SIMULADOR
# =====================================


elif pagina == "Simulador":

    st.title("Simulador de consumo de combustible")

    peso = st.number_input(
        "Peso de carga (toneladas)",
        min_value=0.0,
        value=15.0
    )

    velocidad = st.number_input(
        "Velocidad promedio (km/h)",
        min_value=0.0,
        value=80.0
    )

    ruta = st.selectbox(
        "Tipo de ruta",
        ["Autopista", "Urbana", "Montaña"]
    )

    # Predicción con regresión lineal múltiple
    consumo_original = B0 + B1 * peso + B2 * velocidad

    # Obtener factor de corrección
    factor = comparacion_rutas.loc[ruta, "Factor"]

    # Aplicar factor
    consumo_ajustado = consumo_original * factor

    # Mostrar resultados
    col1, col2 = st.columns(2)

    with col2:
        st.metric(
            "Prediccion de consumo",
            f"{consumo_ajustado:.2f} L"
        )

    st.subheader("Ejemplos de viajes realizados")

    ejemplos = df.groupby(
        "Tipo_Ruta", sort=True
    ).head(2)

    ejemplos = ejemplos[
        [
            "ID_Viaje",
            "Tipo_Ruta",
            "Peso_Carga_Ton",
            "Velocidad_Promedio_kmh",
            "Consumo_Litros"
        ]
    ]

    ejemplos = ejemplos.rename(columns={
        "ID_Viaje": "ID del viaje",
        "Tipo_Ruta": "Tipo de ruta",
        "Peso_Carga_Ton": "Peso (ton)",
        "Velocidad_Promedio_kmh": "Velocidad (km/h)",
        "Consumo_Litros": "Consumo (L)"
    })

    st.dataframe(
        ejemplos,
        hide_index=True,
        use_container_width=True
    )

    # Corregir el nombre de montaña en una copia
    ejemplos = df.copy()
    ejemplos["Tipo_Ruta"] = ejemplos["Tipo_Ruta"].replace(
        {"Monta¤a": "Montaña"}
    )



# =====================================
# 13. TOMA DE DECISIONES
# =====================================

elif pagina == "Toma de decisiones":

    st.title("Toma de decisiones")

    st.write(
        "A continuación, se presentan las conclusiones "
        "obtenidas a partir del análisis estadístico "
        "de FastTrack Logistics."
    )

    # =====================================
    # PREGUNTA 1
    # =====================================

    st.subheader(
        "1. ¿Debería la empresa evitar ciertas rutas "
        "para cargas pesadas?"
    )

    st.write(
        "Los resultados sugieren que las rutas de montaña "
        "presentan el mayor consumo promedio de combustible. "
        "Sin embargo, también son las rutas que transportan "
        "mayor cantidad de carga en promedio."
    )

    st.write(
        "Al comparar el consumo por tonelada transportada, "
        "las rutas urbanas y de montaña presentan resultados "
        "muy similares, mientras que las rutas de autopista "
        "muestran un consumo considerablemente menor."
    )


    st.divider()

    # =====================================
    # PREGUNTA 2
    # =====================================

    st.subheader(
        "2. ¿En qué tipo de ruta se consume más combustible "
        "y la diferencia es estadísticamente significativa?"
    )

    st.write(
        "Las rutas de montaña presentan el mayor consumo "
        "promedio de combustible, con aproximadamente "
        "75.01 litros por viaje."
    )

    st.write(
        "En comparación, las rutas urbanas registran "
        "un promedio de 42.27 litros y las rutas de "
        "autopista de 41.25 litros."
    )

    st.write(
        "Sin embargo, al comparar el consumo por tonelada, "
        "las rutas urbanas y de montaña presentan valores "
        "similares: 4.14 y 4.20 L/ton, respectivamente."
    )

    st.write(
        "La prueba ANOVA obtuvo un valor-p inferior a 0.05, "
        "por lo que se rechaza la hipótesis nula de igualdad "
        "de medias de consumo entre los tipos de ruta."
    )


    st.divider()

    # =====================================
    # PREGUNTA 3
    # =====================================

    st.subheader(
        "3. ¿Debería la empresa controlar la velocidad "
        "o reducir el peso de carga?"
    )

    st.write(
        "Los resultados sugieren que el peso de carga "
        "es un factor especialmente importante para "
        "el consumo de combustible."
    )

    st.write(
        "En la matriz de correlación de Pearson, "
        "el peso presenta una correlación positiva "
        "fuerte con el consumo (r = 0.84), mientras "
        "que la velocidad presenta una correlación "
        "lineal mucho más débil (r = -0.09)."
    )

    st.write(
        f"En nuestro modelo de regresión múltiple, "
        f"el coeficiente del peso fue {B1:.4f}, "
        f"mientras que el coeficiente de velocidad "
        f"fue {B2:.4f}."
    )


    st.divider()

    # =====================================
    # CONCLUSIÓN GENERAL
    # =====================================

    st.subheader("Conclusión general")

    st.write(
        "El análisis realizado permite identificar "
        "diferencias importantes en el consumo de "
        "combustible según el tipo de ruta y el "
        "peso transportado."
    )

    st.write(
        "Las rutas de montaña presentan el mayor "
        "consumo promedio por viaje, mientras que "
        "las autopistas muestran el menor consumo "
        "por tonelada transportada."
    )

    st.write(
        "El modelo de regresión lineal múltiple "
        "permite estimar el consumo utilizando "
        "el peso y la velocidad promedio. "
    )

