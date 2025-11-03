

# Motor de Inferencia por Enumeración en Redes Bayesianas (Python, OOP)

Proyecto académico: implementación completa, genérica y documentada de un **motor de inferencia por enumeración** sobre **Redes Bayesianas** con carga de **estructura** y **tablas de probabilidad (CPTs)** desde archivos.

---

## ✅ Objetivos del proyecto

1. **Estructura OOP** de una Red Bayesiana:
   - `BayesianNode`: nodos booleanos y sus CPT.
   - `BayesianNetwork`: grafo, arcos, orden topológico, descripciones legibles.
   - `BNFileLoader`: lectura de `estructura.txt` y `cpts.json`.
2. **Motor de inferencia por enumeración**:
   - Algoritmo `enumerate_ask` (con `enumerate_all`) con **traza paso a paso**.
   - Evidencia arbitraria, consultas genéricas.
3. **Validación** con la red clásica `Alarm` (Burglary–Earthquake–Alarm–John–Mary).
4. **Generalidad**: reutilizable con **cualquier** red (mientras sea DAG).

---

## 🗂️ Estructura del repositorio

```

Proyecto3_RedBayesiana/
├── bayes_enum.py       # Código completo (OOP + inferencia + demo)
├── estructura.txt      # Estructura de la red (arcos)
└── cpts.json           # CPTs (tablas de probabilidad)

```

---

## 🧩 Archivos de entrada

### 1️⃣ `estructura.txt`

Formato:  
- Nodos **sin padres**: `- -> Nodo`
- Nodos **con padres**: `Padre1,Padre2 -> Hijo` (el **orden** importa)

**Ejemplo (Alarm):**
```

* -> Burglary
* -> Earthquake
  Burglary,Earthquake -> Alarm
  Alarm -> JohnCalls
  Alarm -> MaryCalls

````

### 2️⃣ `cpts.json`

Archivo JSON donde:
- `parents`: lista de padres (en el mismo orden que `estructura.txt`).
- `table`: clave = asignación de padres (`"A=T,B=F"`), valor = `P(Nodo=True | padres)`.
- Si no hay padres, usar clave vacía `""`.

**Ejemplo:**
```json
{
  "Burglary": {
    "parents": [],
    "table": { "": 0.001 }
  },
  "Earthquake": {
    "parents": [],
    "table": { "": 0.002 }
  },
  "Alarm": {
    "parents": ["Burglary", "Earthquake"],
    "table": {
      "Burglary=T,Earthquake=T": 0.95,
      "Burglary=T,Earthquake=F": 0.94,
      "Burglary=F,Earthquake=T": 0.29,
      "Burglary=F,Earthquake=F": 0.001
    }
  },
  "JohnCalls": {
    "parents": ["Alarm"],
    "table": {
      "Alarm=T": 0.90,
      "Alarm=F": 0.05
    }
  },
  "MaryCalls": {
    "parents": ["Alarm"],
    "table": {
      "Alarm=T": 0.70,
      "Alarm=F": 0.01
    }
  }
}
````

---

## 🧠 Diseño orientado a objetos

### 🟦 `BayesianNode`

Representa una variable **booleana** con:

* `name`: nombre del nodo.
* `parents`: lista ordenada de padres.
* `children`: lista de hijos.
* `cpt_true`: tabla de probabilidades condicionadas (`dict`).

**Métodos principales:**

* `p_true_given(parent_assignment)` → retorna `P(True | padres)`.
* `p_value_given(value, parent_assignment)` → retorna `P(value | padres)`.

---

### 🟩 `BayesianNetwork`

Gestiona **todos los nodos** y las **relaciones**.

**Métodos principales:**

* `ensure_node`, `add_node`, `get_node`, `connect` → construcción de la red.
* `topological_order()` → devuelve orden de variables para inferencia.
* `describe_structure()`, `describe_cpts()` → muestran estructura y CPTs legibles.

---

### 🟨 `BNFileLoader`

Carga desde archivos externos:

* `load_structure(path, bn)` → lee `estructura.txt`.
* `load_cpts(path, bn)` → lee `cpts.json`.

**Valida** que el orden de padres coincida entre ambos archivos.

---

### 🟥 `InferenceEngine`

Implementa el **algoritmo de Enumeración** (`enumerate_ask`):

* `query(query_var, evidence, trace=True)` → retorna `{True: p, False: p}`.
* `_enumerate_all(vars, evidence, trace)` → recursión principal.

Incluye **traza paso a paso** (multiplicaciones, sumas, normalización).

---

## ▶️ Cómo ejecutar

1. Tener instalado **Python 3.10+**
2. Abrir una terminal en la carpeta del proyecto:

   ```bash
   cd Proyecto3_RedBayesiana
   ```
3. Ejecutar:

   ```bash
   python bayes_enum.py
   ```
4. Se imprimirá:

   * La estructura de la red.
   * Las tablas de probabilidad.
   * La traza detallada del proceso de inferencia.
   * El resultado normalizado de la consulta:

     ```
     P(Burglary | JohnCalls=T, MaryCalls=T)
     ```

---

## 🧪 Ejemplos de consultas

En `demo_alarm_inference()` puedes modificar la consulta según necesites:

```python
engine = InferenceEngine(bn)

# Consulta 1 (por defecto)
result = engine.query("Burglary", {"JohnCalls": True, "MaryCalls": True}, trace=True)

# Consulta 2
# result = engine.query("Alarm", {"JohnCalls": True}, trace=True)

# Consulta 3
# result = engine.query("Earthquake", {"MaryCalls": True}, trace=True)
```

Para evitar la traza, cambia `trace=True` por `trace=False`.

---

## 🧮 ¿Qué hace el motor?

Implementa la inferencia por **Enumeración**, basada en:

[
P(X | e) \propto \sum_y P(X, y, e)
]

El algoritmo recorre las variables:

* Si una variable está **en evidencia**, multiplica por ( P(Y=y | padres) ).
* Si no está en evidencia, **suma** sobre ambas posibilidades ( Y \in {T, F} ).
* Al final **normaliza** la distribución para que sume 1.

---

## 🔧 Cómo usar con otra red

1. Edita `estructura.txt` con tus nuevos **nodos y arcos**.
2. Crea un nuevo `cpts.json` con las **tablas de probabilidad**.
3. Asegúrate que el orden de `parents` y sus claves coincida.
4. Ejecuta el mismo script:

   ```bash
   python bayes_enum.py
   ```
5. Cambia la consulta según tus nuevas variables.

---

## 🧱 Suposiciones y limitaciones

* Solo soporta **variables booleanas (True/False)**.
* La red debe ser un **DAG (acíclica)**.
* CPTs completas y válidas (valores en [0,1]).
* Enumeración exacta → complejidad **exponencial** en número de variables ocultas.

**Posibles extensiones:**

* Variables con más de dos estados.
* Optimización con **memoization**.
* Lectura de CPTs desde CSV.
* Validación automática de CPTs.

---

## 🧰 Requisitos

* Python 3.10 o superior.
* Sin dependencias externas.

---

## ⚠️ Errores comunes

| Error                        | Causa                                     | Solución                                                       |
| ---------------------------- | ----------------------------------------- | -------------------------------------------------------------- |
| `KeyError` en CPT            | El orden o nombres de padres no coinciden | Asegura que `parents` en JSON coincidan con `estructura.txt`   |
| `RuntimeError: no es un DAG` | Existe un ciclo en la estructura          | Revisa `estructura.txt` y corrige el grafo                     |
| `FileNotFoundError`          | Archivos no encontrados                   | Confirma nombres y ubicación de `estructura.txt` y `cpts.json` |

---

## 📊 Ejemplo de salida esperada

```
=== Estructura de la Red Bayesiana ===
- Earthquake: padres -> (sin padres)
- Burglary: padres -> (sin padres)
- Alarm: padres -> Burglary, Earthquake
- MaryCalls: padres -> Alarm
- JohnCalls: padres -> Alarm
=== Tablas de Probabilidad (P(nodo=True | padres)) ===
...

====== Distribución normalizada ======
P(Burglary=T | evidencia) = 0.284172
P(Burglary=F | evidencia) = 0.715828
=====================================
```

---

## 👩‍🏫 Recomendaciones para la sustentación

* Explica brevemente qué es una **Red Bayesiana** y cómo se usa la **enumeración**.
* Muestra la **estructura** y las **CPTs**.
* Ejecuta el programa y comenta:

  * Las líneas donde se **multiplican probabilidades condicionadas**.
  * Dónde se **suma sobre variables ocultas**.
  * Cómo se **normaliza** el resultado final.
* Cambia evidencia en vivo para demostrar la flexibilidad del modelo.

---

## ✍️ Autoría

* **Autor(a):** (Tu nombre completo)
* **Curso:** (Nombre del curso / grupo)
* **Universidad:** Pontificia Universidad Javeriana – Ingeniería de Sistemas
* **Lenguaje:** Python 3.10+

---

```




