# TAM PROJECT: Filtro de Kalman Adaptativo con LSTM Dual-Head para Trading Algorítmico (EUR/JPY)

**Universidad Nacional de Colombia - sede Manizales**  
**Departamento de Ingeniería Eléctrica, Electrónica, y Computación**  

**Presentado por:**  
- Santiago Alexander Zambrano Chicunque (Ingeniería Electrónica)  
- Santiago Bustamante Montoya (Ingeniería Electrónica)  

**Profesor:**  
- Diego Armando Pérez Rosero, PhD.C  

---

## 📌 Visión General del Proyecto
Este proyecto propone un sistema predictivo híbrido que elimina el ruido del precio mediante un **Filtro de Kalman Adaptativo Bidireccional (Zero-Lag)** y emplea una red neuronal **LSTM Dual-Head** para modelar la microestructura residual. El resultado es un canal predictivo estocástico ($\pm 2\sigma$) para el par EUR/JPY (temporalidad H1).

### 📊 Presentación Académica (`index.html`)
Se incluye una presentación interactiva detallada fase a fase (Descomposición Aditiva, Explorador de Resonancia Temporal, LSTM, Canal Predictivo y Backtest), con todo el código documentado.

### 💻 Cuaderno de Investigación (`test_final.ipynb`)
El simulador ha sido completamente portado a **Matplotlib** puro para generación de gráficos (`.png`), eliminando dependencias pesadas de Chromium/Kaleido, garantizando su correcta ejecución y exportación directa desde **Google Colab**.

---

## Arquitectura y Dinámica Interna de la Red Neuronal Recurrente LSTM Híbrida de Salida Dual
### Especificación de Hardware Lógico, Flujo de Señales y Ecuaciones Dinámicas de Control Multitarea

Este documento detalla la arquitectura de la red neuronal **LSTM (Long Short-Term Memory) de Salida Dual** implementada para el par de Forex **EUR/JPY** en temporalidad **H1**. El sistema se describe desde la perspectiva del modelado de sistemas de control, procesamiento de señales y dinámica de redes neuronales recurrentes (RNN).

---

## 1. Flujo de Señales y Estructura de Capas

El modelo procesa una ventana de tiempo discreto de longitud $T = 24$ (representando las últimas 24 horas de negociación) y extrae características normalizadas para predecir de forma simultánea dos variables a tiempo $t+1$: el residuo microestructural (AC) y la desviación estándar de la predicción (volatilidad local).

### Diagrama del Espectro de Señales de la Red

```mermaid
graph TD
    X["Tensor de Entrada X[n] (Dim: 24 x 2)"] --> LSTM1["Capa Recurrente 1 (LSTM 1)<br/>64 Unidades - return_seq=True"]
    LSTM1 -->|H^(1) (Dim: 24 x 64)| Drop1["Capa de Dropout 1<br/>Frecuencia: 20%"]
    Drop1 -->|D^(1) (Dim: 24 x 64)| LSTM2["Capa Recurrente 2 (LSTM 2)<br/>32 Unidades - return_seq=False"]
    LSTM2 -->|h_24^(2) (Dim: 32)| Drop2["Capa de Dropout 2<br/>Frecuencia: 20%"]
    Drop2 -->|d_24^(2) (Dim: 32)| Shared["Capa Densa Latente Compartida<br/>32 Neuronas - Activación tanh"]
    Shared -->|h_shared (Dim: 32)| HeadRes["Cabezal 1: Residuo AC (Res)<br/>1 Neurona - Activación Lineal"]
    Shared -->|h_shared (Dim: 32)| HeadStd["Cabezal 2: Volatilidad (Std)<br/>1 Neurona - Activación ReLU"]
    HeadRes -->|r_norm_hat[n+1]| UnscaleZ["Des-escalador Z-Score"]
    HeadStd -->|std_norm_hat[n+1]| UnscaleMinMax["Des-escalador MinMax"]
    UnscaleZ -->|r_hat[n+1]| Output["r_hat[n+1] (Residuo Físico AC)"]
    UnscaleMinMax -->|sigma_hat[n+1]| OutputVol["sigma_hat[n+1] (Volatilidad Física)"]

    style X fill:#34495E,stroke:#2C3E50,stroke-width:2px,color:#FFF
    style LSTM1 fill:#2980B9,stroke:#1F3A60,stroke-width:2px,color:#FFF
    style LSTM2 fill:#2980B9,stroke:#1F3A60,stroke-width:2px,color:#FFF
    style Drop1 fill:#7F8C8D,stroke:#5D6D7E,stroke-width:2px,color:#FFF
    style Drop2 fill:#7F8C8D,stroke:#5D6D7E,stroke-width:2px,color:#FFF
    style Shared fill:#8E44AD,stroke:#5B2C6F,stroke-width:2px,color:#FFF
    style HeadRes fill:#27AE60,stroke:#196F3D,stroke-width:2px,color:#FFF
    style HeadStd fill:#E67E22,stroke:#9E5207,stroke-width:2px,color:#FFF
    style UnscaleZ fill:#1ABC9C,stroke:#16A085,stroke-width:2px,color:#FFF
    style UnscaleMinMax fill:#1ABC9C,stroke:#16A085,stroke-width:2px,color:#FFF
    style Output fill:#2C3E50,stroke:#2C3E50,stroke-width:2px,color:#FFF
    style OutputVol fill:#2C3E50,stroke:#2C3E50,stroke-width:2px,color:#FFF
```

---

## 2. Descripción Capa por Capa

### 2.1. Capa de Entrada (Input Tensor)
El tensor de entrada a tiempo de predicción $n$ es una matriz de dimensiones fijas:
$$X[n] \in \mathbb{R}^{T \times F} \quad \text{con } T = 24, \, F = 2$$

Donde:
* **Característica 1 ($f_1$):** El precio suavizado obtenido del Filtro de Kalman Adaptativo de Fase Cero ($s[n]$), normalizado mediante Z-Score.
* **Característica 2 ($f_2$):** La volatilidad local calculada sobre el precio real ($\text{StdDev}[n]$), normalizada mediante Z-Score.

$$X[n] = \begin{bmatrix} s_{\text{norm}}[n-23] & \text{StdDev}_{\text{norm}}[n-23] \\ s_{\text{norm}}[n-22] & \text{StdDev}_{\text{norm}}[n-22] \\ \vdots & \vdots \\ s_{\text{norm}}[n] & \text{StdDev}_{\text{norm}}[n] \end{bmatrix}$$

### 2.2. Capa Recurrente 1 (LSTM 1)
* **Configuración:** 64 unidades recurrentes independientes, `return_sequences=True`.
* **Comportamiento:** Procesa secuencialmente el tensor de entrada y produce un estado oculto para cada paso temporal. La salida es un tensor tridimensional de activaciones temporales:
  $$H^{(1)}[n] \in \mathbb{R}^{T \times 64}$$
* **Funciones de Activación:** Tangente hiperbólica ($\tanh$) para el estado interno y sigmoide logística ($\sigma_g$) para las compuertas lógicas.

### 2.3. Capa de Regularización (Dropout 1)
* **Configuración:** Tasa de atenuación estocástica de $0.2$ (20%).
* **Comportamiento:** Durante el entrenamiento, apaga aleatoriamente el 20% de las conexiones en cada paso del tensor $H^{(1)}[n]$, actuando como un filtro de ruido que fuerza a la red a no depender de canales de transmisión específicos (co-adaptación), evitando el sobreajuste.

### 2.4. Capa Recurrente 2 (LSTM 2)
* **Configuración:** 32 unidades recurrentes, `return_sequences=False`.
* **Comportamiento:** Recibe el tensor regularizado $D^{(1)}[n] \in \mathbb{R}^{24 \times 64}$. Al tener la bandera de retorno secuencial en falso, la capa solo emite el último vector del estado oculto a tiempo final $k=24$:
  $$h_{24}^{(2)} \in \mathbb{R}^{32}$$
  Este vector consolida la memoria de largo y corto plazo filtrada a través de las dos capas recurrentes.

### 2.5. Capa de Regularización (Dropout 2)
* **Configuración:** Tasa de atenuación estocástica de $0.2$.
* **Comportamiento:** Regulariza el vector de salida de la segunda LSTM, produciendo el vector $d_{24}^{(2)} \in \mathbb{R}^{32}$.

### 2.6. Capa Densa Latente Compartida (Shared Feature Layer)
* **Configuración:** Capa Densa (Fully Connected) de 32 neuronas con activación $\tanh$.
* **Comportamiento:** Extrae una representación abstracta final que consolida las características necesarias para ambas tareas del sistema de predicción. Su salida es el estado compartido $h_{\text{shared}}$:
  $$h_{\text{shared}} = \tanh\left( W_s d_{24}^{(2)} + b_s \right) \quad \text{con } W_s \in \mathbb{R}^{32 \times 32}, \, b_s \in \mathbb{R}^{32}$$

### 2.7. Cabezales de Salida (Dual Heads)

La arquitectura se bifurca en dos proyecciones independientes a partir del vector latente compartido $h_{\text{shared}}$:

#### Cabezal 1: Consenso del Residuo AC ($\hat{r}[n+1]$)
* **Neuronas:** 1.
* **Activación:** Lineal (Identidad).
* **Propósito:** Predecir el valor del residuo microestructural (AC) del par de Forex para la siguiente hora.
* **Ecuación:**
  $$\hat{r}^{\text{norm}}[n+1] = W_r h_{\text{shared}} + b_r \quad \text{con } W_r \in \mathbb{R}^{1 \times 32}, \, b_r \in \mathbb{R}$$
  $$\hat{r}[n+1] = \hat{r}^{\text{norm}}[n+1] \cdot \sigma_{\text{residual}} + \mu_{\text{residual}}$$

#### Cabezal 2: Varianza Local / Volatilidad Predictiva ($\hat{\sigma}[n+1]$)
* **Neuronas:** 1.
* **Activación:** **ReLU (Rectified Linear Unit)**. La activación ReLU está dada por $f(x) = \max(0, x)$. Esto asegura físicamente que la desviación estándar predicha nunca sea negativa, lo cual invalidaría las ecuaciones probabilísticas.
* **Propósito:** Predecir el margen de volatilidad estadística para el siguiente periodo.
* **Ecuación:**
  $$\hat{\sigma}^{\text{norm}}[n+1] = \max\left(0, W_{\sigma} h_{\text{shared}} + b_{\sigma}\right) \quad \text{con } W_{\sigma} \in \mathbb{R}^{1 \times 32}, \, b_{\sigma} \in \mathbb{R}$$
  $$\hat{\sigma}[n+1] = \hat{\sigma}^{\text{norm}}[n+1] \cdot (\sigma_{\text{max}} - \sigma_{\text{min}}) + \sigma_{\text{min}}$$

---

## 3. Funcionamiento Interno de la Celda LSTM

Desde la perspectiva de la teoría de control y modelado analógico, una celda LSTM puede ser analizada como un **integrador dinámico no lineal con bucles de realimentación y compuertas de conmutación**.

El siguiente diagrama detalla la microarquitectura de la celda en un paso temporal de la secuencia $k \in [1, 24]$:

```mermaid
graph TD
    subgraph Celda_LSTM [Microarquitectura de la Celda LSTM (Paso k)]
        h_prev["Hidden State Previo h[k-1]"]
        u_k["Input actual u[k]"]
        c_prev["Cell State Previo c[k-1]"]

        %% Gate calculations
        f_gate["Compuerta de Olvido f[k]<br/>(sigmoid)"]
        i_gate["Compuerta de Entrada i[k]<br/>(sigmoid)"]
        c_cand["Estado Candidato c~[k]<br/>(tanh)"]
        o_gate["Compuerta de Salida o[k]<br/>(sigmoid)"]

        %% Operations
        mult_f["Multiplicación (x)"]
        mult_i["Multiplicación (x)"]
        sum_c["Suma (+)"]
        tanh_c["Saturación tanh"]
        mult_o["Multiplicación (x)"]

        %% Output
        c_k["Cell State c[k]"]
        h_k["Hidden State h[k]"]

        %% Connections
        h_prev --> f_gate
        h_prev --> i_gate
        h_prev --> c_cand
        h_prev --> o_gate

        u_k --> f_gate
        u_k --> i_gate
        u_k --> c_cand
        u_k --> o_gate

        c_prev --> mult_f
        f_gate --> mult_f

        i_gate --> mult_i
        c_cand --> mult_i

        mult_f --> sum_c
        mult_i --> sum_c

        sum_c --> c_k
        c_k --> tanh_c
        tanh_c --> mult_o
        o_gate --> mult_o
        mult_o --> h_k
    end

    style Celda_LSTM fill:#F8F9FA,stroke:#343A40,stroke-width:2px
    style h_prev fill:#34495E,stroke:#2C3E50,stroke-width:1px,color:#FFF
    style u_k fill:#34495E,stroke:#2C3E50,stroke-width:1px,color:#FFF
    style c_prev fill:#34495E,stroke:#2C3E50,stroke-width:1px,color:#FFF
    style f_gate fill:#2980B9,stroke:#1F3A60,stroke-width:1px,color:#FFF
    style i_gate fill:#2980B9,stroke:#1F3A60,stroke-width:1px,color:#FFF
    style c_cand fill:#2980B9,stroke:#1F3A60,stroke-width:1px,color:#FFF
    style o_gate fill:#2980B9,stroke:#1F3A60,stroke-width:1px,color:#FFF
    style mult_f fill:#E67E22,stroke:#D35400,stroke-width:1px,color:#FFF
    style mult_i fill:#E67E22,stroke:#D35400,stroke-width:1px,color:#FFF
    style sum_c fill:#E67E22,stroke:#D35400,stroke-width:1px,color:#FFF
    style tanh_c fill:#8E44AD,stroke:#7D3C98,stroke-width:1px,color:#FFF
    style mult_o fill:#E67E22,stroke:#D35400,stroke-width:1px,color:#FFF
    style c_k fill:#27AE60,stroke:#1E8449,stroke-width:1px,color:#FFF
    style h_k fill:#27AE60,stroke:#1E8449,stroke-width:1px,color:#FFF
```

A continuación se describe la función física de cada compuerta dentro de la celda recurrente para una muestra secuencial $u[k] \in \mathbb{R}^{F_{\text{in}}}$:

### 3.1. Compuerta de Olvido ($f[k]$) - Coeficiente de Fuga
Actúa como un atenuador de la memoria del sistema. Controla la retención de carga en el integrador.
$$f[k] = \sigma_g\left(W_f u[k] + U_f h[k-1] + b_f\right)$$

* **Interpretación:** La función sigmoide restringe $f[k] \in [0, 1]$. Si $f[k] = 1$, la memoria histórica del acumulador se retiene por completo (resistencia de fuga infinita). Si $f[k] = 0$, la carga del integrador se descarga completamente a cero.

### 3.2. Compuerta de Entrada ($i[k]$) - Ganancia de Acoplamiento
Controla la cantidad de corriente de entrada que se inyecta en el acumulador.
$$i[k] = \sigma_g\left(W_i u[k] + U_i h[k-1] + b_i\right)$$

### 3.3. Estado Candidato ($\tilde{c}[k]$) - Amplificador de Entrada
Pre-procesa las señales de entrada mediante una ganancia no lineal de saturación simétrica:
$$\tilde{c}[k] = \tanh\left(W_c u[k] + U_c h[k-1] + b_c\right)$$

### 3.4. Actualización del Estado de Celda ($c[k]$) - El Integrador Activo
El estado de la celda $c[k]$ representa la memoria acumulada a largo plazo del sistema. Su dinámica está regida por la siguiente ecuación diferencial en diferencias de primer orden:
$$c[k] = f[k] \odot c[k-1] + i[k] \odot \tilde{c}[k]$$

Donde $\odot$ es el producto elemento a elemento (Hadamard). Esta ecuación lineal variable en el tiempo permite que el gradiente fluya libremente hacia atrás en el tiempo sin desvanecerse exponencialmente (el problema clásico de la desaparición del gradiente en RNN estándar).

### 3.5. Compuerta de Salida ($o[k]$) y Hidden State ($h[k]$) - Buffer de Salida
La compuerta de salida regula la cantidad de señal interna del integrador que se propaga como Hidden State hacia el siguiente paso temporal y hacia las siguientes capas:
$$o[k] = \sigma_g\left(W_o u[k] + U_o h[k-1] + b_o\right)$$
$$h[k] = o[k] \odot \tanh(c[k])$$

---

## 4. Dinámica del Aprendizaje Multitarea (Backpropagation Through Time)

El modelo aprende a ajustar su matriz de parámetros $\mathcal{W} = \{W, U, b\}$ minimizando de forma simultánea los errores de predicción del residuo y de la varianza en una función de pérdida conjunta (Loss Multitarea):

### 4.1. Formulación Matemática de la Loss
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{res}} + w \cdot \mathcal{L}_{\text{std}}$$

Donde:
* **Loss del Residuo (MSE de la componente AC):**
  $$\mathcal{L}_{\text{res}} = \frac{1}{B} \sum_{i=1}^B \left( r_i - \hat{r}_i^{\text{norm}} \right)^2$$
* **Loss del Riesgo (MSE de la desviación estándar):**
  $$\mathcal{L}_{\text{std}} = \frac{1}{B} \sum_{i=1}^B \left( \sigma_i^{\text{norm}} - \hat{\sigma}_i^{\text{norm}} \right)^2$$
* **Parámetro de Ponderación Operacional ($w = 0.3$):** 
  El coeficiente de ponderación de la varianza se fija en $w=0.3$. Esto se debe a que la predicción del ruido e incertidumbre ($\sigma_i$) tiene una dinámica altamente estocástica y ruidosa en comparación con el consenso central ($r_i$). Un peso de $0.3$ evita que las derivadas de la volatilidad dominen el gradiente total, protegiendo al extractor de características compartidas de desestabilizar la convergencia de la trayectoria central del precio.
* **$B$:** Tamaño del lote de datos (Batch Size = 32).

### 4.2. Flujo de Gradientes vía BPTT
El entrenamiento calcula las derivadas parciales de la función de pérdida total respecto a cada peso mediante el algoritmo de **Retropropagación a través del Tiempo (BPTT)**. Para un peso general $W_{ij}$ de las compuertas recurrentes, el gradiente se propaga en dirección contraria al tiempo:

$$\frac{\partial \mathcal{L}_{\text{total}}}{\partial W_{ij}} = \sum_{k=1}^{T} \frac{\partial \mathcal{L}_{\text{total}}}{\partial h[k]} \cdot \frac{\partial h[k]}{\partial c[k]} \cdot \frac{\partial c[k]}{\partial W_{ij}}$$

La ecuación del integrador activo $c[k] = f[k] \odot c[k-1] + i[k] \odot \tilde{c}[k]$ garantiza que:
$$\frac{\partial c[k]}{\partial c[k-1]} = f[k]$$

Si la compuerta de olvido está activa ($f[k] \approx 1$), el gradiente fluye intacto hacia pasos de tiempo lejanos, eliminando la atenuación exponencial de la señal de error. El optimizador **Adam** ajusta dinámicamente los momentos de primer y segundo orden de estos gradientes para actualizar los pesos en cada época:

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

Donde $\eta = 0.001$ es la tasa de aprendizaje, $\hat{m}_t$ es el promedio móvil corregido del gradiente (primer momento) y $\hat{v}_t$ es la varianza móvil corregida (segundo momento).

---

## 5. Síntesis Final de la Predicción (Reconstrucción del Canal)

El modelo se ejecuta en producción sumando la predicción del residuo predicho a la tendencia de Kalman para reconstruir el precio absoluto proyectado:

$$\widehat{y}[n+1] = \text{Kalman\_ZeroLag}[n+1] + \hat{r}[n+1]$$

A su vez, se proyecta la envolvente de fluctuación gaussiana de la predicción, modelada por la salida dual del cabezal de riesgo:

$$\text{Canal Predictivo}[n+1] = \widehat{y}[n+1] \pm 2\hat{\sigma}[n+1]$$

Este canal actúa como la frontera dinámica de decisión:
* Un precio real $y[n] > \widehat{y}[n] + 2\hat{\sigma}[n]$ representa una sobretensión (sobrecompra estadísticamente saturada) que gatilla órdenes SHORT.
* Un precio real $y[n] < \widehat{y}[n] - 2\hat{\sigma}[n]$ representa una caída de tensión (sobreventa estadísticamente saturada) que gatilla órdenes LONG.
