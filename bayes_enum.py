# bayes_enum.py
# ------------------------------------------------------------
# Motor de Inferencia por Enumeración para Redes Bayesianas
# 
# Estructura del código:
#   - BayesianNode: representa un nodo (variable booleana) con su CPT
#   - BayesianNetwork: representa la red completa (nodos y arcos)
#   - BNFileLoader: utilidades para cargar estructura y CPTs desde archivos
#   - InferenceEngine: implementación de enumeración-ask con trazas
#
# Archivos de entrada:
#   - estructura.txt  (formato: "A,B -> C" o "- -> Root")
#   - cpts.json       (ver ejemplo en las instrucciones)
#
# Uso rápido (ver al final en "if __name__ == '__main__'"):
#   python bayes_enum.py
# ------------------------------------------------------------

from __future__ import annotations  # Habilita referenciar clases que se definen después (anotaciones adelantadas)
from dataclasses import dataclass, field        # dataclass simplifica clases de datos; field para defaults mutables
from typing import Dict, List, Tuple, Optional  # Tipos para claridad: diccionarios, listas, tuplas, opcionales
import json                                     # Para leer y parsear cpts.json (tablas de prob.)
import itertools                                # (No se usa finalmente, pero sirve para combinaciones si amplías)
import os                                       # Para verificar existencia de archivos en el sistema


# ------------------------------------------------------------
# Clase: BayesianNode
# Representa un nodo booleano en la Red Bayesiana.
# - name: nombre del nodo (string)
# - parents: lista de nombres de padres en orden (importante para la CPT)
# - children: lista de nombres de hijos (se llena al construir la red)
# - cpt_true: dict que mapea "clave de padres" -> P(node=True | padres)
#
# La "clave de padres" es una cadena tipo "A=T,B=F" respetando el orden de self.parents.
# Si no hay padres, la clave es "" (cadena vacía).
# ------------------------------------------------------------
@dataclass  # Convierte la clase en contenedor de datos con __init__ y otros métodos auto-generados
class BayesianNode:
    name: str  # Nombre único del nodo (variable) en la red
    parents: List[str] = field(default_factory=list)   # Lista ordenada de nombres de padres (vacía si no tiene)
    children: List[str] = field(default_factory=list)  # Lista de nombres de hijos (se llena al conectar)
    cpt_true: Dict[str, float] = field(default_factory=dict)  # CPT: clave de padres -> P(nodo=True | padres)

    def p_true_given(self, parent_assignment: Dict[str, bool]) -> float:
        """
        Retorna P(self=True | parent_assignment) usando la clave de padres
        en el mismo orden declarado en self.parents.

        parent_assignment: dict {parent_name: bool}  # Asignación booleana de cada padre

        Ejemplo:
            Si parents = ["A","B"] y parent_assignment = {"A": True, "B": False},
            la clave formada será "A=T,B=F".
        """
        # Construir la clave en el orden exacto de self.parents
        if not self.parents:  # Si no hay padres...
            # Sin padres -> clave vacía
            key = ""  # key será "" para consultar la probabilidad marginal del nodo
        else:
            # Unimos "Nombre=Valor" con comas según el orden fijo
            # Nota: 'T' para True y 'F' para False
            parts = []  # parts acumula cada "Padre=T/F" respetando el orden de self.parents
            for p in self.parents:           # Recorremos los padres en su orden
                val = parent_assignment[p]   # val es el bool asignado al padre p
                parts.append(f"{p}={'T' if val else 'F'}")  # Agregamos "p=T" o "p=F" según val
            key = ",".join(parts)  # key final: "A=T,B=F" (mismo orden que self.parents)

        # Buscar probabilidad en la CPT
        if key not in self.cpt_true:  # Validamos que exista la fila correspondiente en la CPT
            raise KeyError(
                f"No existe una entrada CPT para el nodo '{self.name}' con clave de padres '{key}'. "
                f"Verifica el archivo cpts.json y el orden de 'parents'."
            )
        return self.cpt_true[key]  # Retornamos P(node=True | padres) correspondiente a esa clave

    def p_value_given(self, value: bool, parent_assignment: Dict[str, bool]) -> float:
        """
        Retorna P(self=value | padres). Si 'value' es False, usa 1 - P(True|...).
        """
        p_true = self.p_true_given(parent_assignment)  # Calcula P(True | padres)
        return p_true if value else (1.0 - p_true)     # Si pedimos False, retornamos el complemento


# ------------------------------------------------------------
# Clase: BayesianNetwork
# Contiene todos los nodos y provee utilidades:
# - add_node / get_node
# - conectar padres e hijos según estructura
# - ordenar topológicamente
# - impresiones legibles de estructura y CPT
# ------------------------------------------------------------
class BayesianNetwork:
    def __init__(self):
        # Diccionario: nombre -> BayesianNode
        self.nodes: Dict[str, BayesianNode] = {}  # Almacena cada nodo de la red por su nombre

    # --- creación / acceso básico ---
    def add_node(self, node: BayesianNode) -> None:
        if node.name in self.nodes:  # Evita duplicados de nombre
            raise ValueError(f"El nodo '{node.name}' ya existe en la red.")
        self.nodes[node.name] = node  # Inserta el nodo en el diccionario

    def get_node(self, name: str) -> BayesianNode:
        if name not in self.nodes:  # Valida que exista
            raise KeyError(f"El nodo '{name}' no existe en la red.")
        return self.nodes[name]      # Retorna el objeto BayesianNode

    def ensure_node(self, name: str) -> BayesianNode:
        """
        Crea el nodo vacío si no existe (para poder conectar arcos primero).
        """
        if name not in self.nodes:                 # Si no lo tenemos aún...
            self.nodes[name] = BayesianNode(name=name)  # ...creamos un nodo con ese nombre
        return self.nodes[name]                    # Retornamos el nodo (existente o recién creado)

    # --- conectar estructura ---
    def connect(self, parents: List[str], child: str) -> None:
        """
        Define que 'child' tiene 'parents' (en ese orden).
        También actualiza la lista de children en cada padre.
        """
        child_node = self.ensure_node(child)  # child_node es el objeto del hijo; lo crea si no existe
        # Establecemos el orden explícito de padres en el nodo hijo
        child_node.parents = parents[:]       # Copiamos la lista para fijar el orden de padres
        # Registramos el hijo en cada padre
        for p in parents:                                 # Para cada padre p...
            parent_node = self.ensure_node(p)             # Aseguramos que el nodo padre exista
            if child not in parent_node.children:         # Evitamos duplicarlo si ya estaba
                parent_node.children.append(child)        # Añadimos el hijo a la lista de hijos del padre

    # --- orden topológico (necesario para la enumeración) ---
    def topological_order(self) -> List[str]:
        """
        Devuelve una lista de nombres de nodos en orden topológico:
        todo padre aparece antes que sus hijos.
        """
        # Conteo de entradas (in-degree) basado en cantidad de padres
        in_degree = {name: len(node.parents) for name, node in self.nodes.items()}  # Grado de entrada por nodo
        # Cola inicial: nodos sin padres
        frontier = [name for name, deg in in_degree.items() if deg == 0]  # Nodos raíz (deg=0)
        order: List[str] = []  # Aquí iremos colocando el orden resultante

        # Algoritmo de Kahn
        while frontier:                # Mientras haya nodos sin dependencias por procesar...
            n = frontier.pop()         # Tomamos uno (LIFO aquí, podría ser FIFO igual)
            order.append(n)            # Lo agregamos al orden
            for child in self.nodes[n].children:  # Para cada hijo del nodo n...
                in_degree[child] -= 1             # Reducimos su grado de entrada (hemos resuelto uno de sus padres)
                if in_degree[child] == 0:         # Si ya no tiene padres pendientes...
                    frontier.append(child)        # Lo agregamos a la frontera

        if len(order) != len(self.nodes):  # Si no cubrimos todos, hay ciclo (no es DAG válido)
            raise RuntimeError("La red parece contener ciclos (no es un DAG). Revisa la estructura.")
        return order  # Retornamos la lista en orden topológico válido

    # --- utilidades de impresión ---
    def describe_structure(self) -> str:
        """
        Devuelve un string con la estructura:
        para cada nodo, lista sus padres (predecesores).
        """
        lines = ["=== Estructura de la Red Bayesiana ==="]  # Encabezado del reporte
        for name in self.topological_order():               # Recorremos en orden topológico
            node = self.nodes[name]                         # node es el BayesianNode actual
            if node.parents:                                # Si tiene padres...
                parents_str = ", ".join(node.parents)       # Formateamos "A, B, C"
            else:
                parents_str = "(sin padres)"                # Caso raíz
            lines.append(f"- {name}: padres -> {parents_str}")  # Añadimos la línea descriptiva
        return "\n".join(lines)                             # Unimos todas las líneas con saltos

    def describe_cpts(self) -> str:
        """
        Devuelve un string legible con todas las tablas de probabilidad P(node=True | padres).
        """
        lines = ["=== Tablas de Probabilidad (P(nodo=True | padres)) ==="]  # Encabezado
        for name in self.topological_order():        # Recorremos nodos en orden topológico
            node = self.nodes[name]                  # Obtenemos el BayesianNode
            lines.append(f"[{name}]")               # Sección para ese nodo
            if not node.parents:                    # Si no tiene padres...
                p = node.cpt_true.get("", None)     # Probabilidad marginal P(name=True) está en clave ""
                lines.append(f"  (sin padres)  P({name}=T) = {p}")  # Imprimimos esa prob.
            else:
                lines.append(f"  padres: {', '.join(node.parents)}")  # Listamos padres en orden
                # Ordenar por clave para una salida estable
                for key in sorted(node.cpt_true.keys()):              # Iteramos filas de CPT ordenadas por clave
                    lines.append(f"  {key}  ->  P({name}=T) = {node.cpt_true[key]}")  # Mostramos cada fila
            lines.append("")  # línea en blanco para separar nodos
        return "\n".join(lines)  # Unimos todo en un string


# ------------------------------------------------------------
# Clase: BNFileLoader
# Encargada de leer los archivos 'estructura.txt' y 'cpts.json' y
# construir/llenar la BayesianNetwork.
# ------------------------------------------------------------
class BNFileLoader:
    @staticmethod  # Método estático: no necesita instancia de BNFileLoader
    def load_structure(path: str, bn: BayesianNetwork) -> None:
        """
        Lee 'estructura.txt' con líneas del tipo:
            - -> Root
            A,B -> C
        Ignora líneas vacías o que empiezan por '#'.
        """
        if not os.path.exists(path):  # Verificamos que el archivo exista
            raise FileNotFoundError(f"No se encontró el archivo de estructura: {path}")

        with open(path, "r", encoding="utf-8") as f:  # Abrimos el archivo en modo lectura
            for raw in f:                             # raw es la línea cruda con saltos
                line = raw.strip()                    # line limpia espacios en extremos
                if not line or line.startswith("#"):  # Saltamos líneas vacías o comentarios
                    continue
                # Separar "izquierda -> derecha"
                if "->" not in line:                  # Validamos el separador requerido
                    raise ValueError(f"Línea inválida en estructura: '{line}' (falta '->')")
                left, right = [part.strip() for part in line.split("->")]  # Split por '->' y limpiamos espacios
                child = right                     # child es el nombre del nodo hijo a conectar
                if left == "-" or left == "":     # Si a la izquierda hay '-' o vacío, no hay padres
                    parents: List[str] = []       # Lista de padres vacía
                else:
                    parents = [p.strip() for p in left.split(",") if p.strip()]  # Parseamos múltiples padres
                # Conectar en la red
                bn.connect(parents, child)        # Llamamos a connect para fijar relaciones

    @staticmethod
    def load_cpts(path: str, bn: BayesianNetwork) -> None:
        """
        Lee 'cpts.json' con estructura:
            { "Node": { "parents": [...], "table": { "A=T,B=F": 0.9, ... } }, ... }
        Valida consistencia entre 'parents' del JSON y los definidos en estructura.
        """
        if not os.path.exists(path):  # Validamos que el archivo exista
            raise FileNotFoundError(f"No se encontró el archivo de CPTs: {path}")

        with open(path, "r", encoding="utf-8") as f:  # Abrimos el JSON
            data = json.load(f)                       # data es un dict con nodos -> especificación

        for node_name, spec in data.items():     # Recorremos cada entrada del JSON
            # Debe existir el nodo en la red (creado al leer la estructura)
            node = bn.ensure_node(node_name)     # Aseguramos que el nodo esté en la red

            # Padres en JSON (para validar)
            json_parents = spec.get("parents", [])  # Lista de padres declarada en el JSON
            # Tabla con P(node=True | padres)
            table = spec.get("table", {})          # Diccionario clave->probabilidad

            # Validación: que coincida el orden y contenido de padres
            # (Importante porque las claves de CPT usan este orden)
            if node.parents != json_parents:    # Comparamos con los padres definidos por la estructura
                raise ValueError(
                    f"Los padres de '{node_name}' en cpts.json {json_parents} "
                    f"no coinciden con los de la estructura {node.parents}.\n"
                    f"Asegúrate de que el orden y los nombres coincidan."
                )

            # Guardar la CPT en el nodo
            # Nota: no convertimos claves; las usamos como vienen ("A=T,B=F" o "")
            node.cpt_true = table  # Asignamos la tabla tal cual (P(True|...))


# ------------------------------------------------------------
# Clase: InferenceEngine
# Implementa enumeración-ask con traza.
#
# Fórmula clásica:
#   enumerate_ask(X, e) devuelve distribución normalizada sobre X:
#       P(X|e) ∝ ∑_y P(x, y, e)
#   usando recursion enumerate_all(vars, e)
#
# Esta implementación:
#  - Variables booleanas
#  - Evidence: dict {var: bool}
#  - query_var: nombre de la variable consultada
#  - trace: si True, imprime pasos de cálculo (claros y simples)
# ------------------------------------------------------------
class InferenceEngine:
    def __init__(self, bn: BayesianNetwork):
        self.bn = bn                             # Guardamos la referencia a la red bayesiana
        # guardamos orden topológico una vez
        self.order = self.bn.topological_order() # self.order: lista de nombres en orden válido para enumeración

    def query(self, query_var: str, evidence: Dict[str, bool], trace: bool = True) -> Dict[bool, float]:
        """
        Devuelve la distribución P(query_var | evidence) como {True: p, False: p}.
        Si trace=True, imprime la traza paso a paso.
        """
        if query_var not in self.bn.nodes:                        # Validamos que exista la variable consultada
            raise KeyError(f"La variable de consulta '{query_var}' no existe en la red.")

        # Construimos el vector de resultados para X=True y X=False
        distro = {}  # distro almacenará prob. no normalizadas para {True: val, False: val}
        for x_val in [True, False]:  # Evaluamos ambos casos de la variable consulta
            # evidence_extiende = evidence ∪ {X=x_val}
            extended_evidence = evidence.copy()     # Copiamos la evidencia original para no mutarla
            extended_evidence[query_var] = x_val    # Fijamos la variable consultada a True o False

            if trace:  # Si queremos ver la traza, imprimimos encabezado del caso
                print("----------------------------------------------------")
                print(f"Caso {query_var} = {'T' if x_val else 'F'} con evidencia {self._fmt_ev(evidence)}")
                print("----------------------------------------------------")

            # Enumerar sobre todas las variables en orden
            p = self._enumerate_all(self.order, extended_evidence, trace)  # p es el valor proporcional P(x,e)
            if trace:
                print(f"Resultado parcial: P({query_var}={'T' if x_val else 'F'} | evidencia) ∝ {p}\n")
            distro[x_val] = p  # Guardamos el resultado proporcional para este valor de X

        # Normalizar
        total = distro[True] + distro[False]  # Suma total para normalizar
        if total == 0.0:                      # Prevención: no debería pasar con CPTs válidas
            raise ZeroDivisionError("La suma de probabilidades es 0; revisa CPTs/evidencia.")
        distro[True] /= total   # Normalizamos el caso True
        distro[False] /= total  # Normalizamos el caso False

        if trace:  # Imprimimos los resultados finales ya normalizados
            print("====== Distribución normalizada ======")
            print(f"P({query_var}=T | evidencia) = {distro[True]:.6f}")
            print(f"P({query_var}=F | evidencia) = {distro[False]:.6f}")
            print("=====================================\n")

        return distro  # Retornamos el diccionario {True: p, False: p}

    def _enumerate_all(self, vars_in_order: List[str], evidence: Dict[str, bool], trace: bool) -> float:
        """
        Implementa la función recursiva enumerate_all.
        vars_in_order: lista de nombres en orden topológico.
        evidence: dict con asignaciones actuales (parciales o completas).
        """
        if not vars_in_order:  # Caso base: si no quedan variables por procesar
            # Caso base: no hay más variables -> multiplicador neutro
            return 1.0         # Devolvemos 1 para no afectar el producto

        # Tomar la primera variable y el resto
        Y = vars_in_order[0]   # Y es la siguiente variable a procesar
        rest = vars_in_order[1:]  # rest son el resto de variables por procesar después de Y
        node = self.bn.get_node(Y)  # node es el objeto BayesianNode asociado a Y

        # Preparar asignación de padres de Y desde 'evidence'
        parent_assignment = {p: evidence[p] for p in node.parents}  # Tomamos los valores actuales de los padres

        if Y in evidence:  # Si Y ya está fijada por evidencia (original o extendida)
            # Y ya está fijada (en evidencia extendida)
            y_val = evidence[Y]                           # y_val es el valor booleano concreto de Y
            # P(Y=y | padres)
            prob = node.p_value_given(y_val, parent_assignment)  # prob = P(Y=y_val|padres)

            if trace:
                print(f"[Fijada] {Y} = {'T' if y_val else 'F'}  =>  "
                      f"P({Y}={'T' if y_val else 'F'} | {self._fmt_parent_assign(parent_assignment)}) = {prob}")

            # Continuar con el resto
            return prob * self._enumerate_all(rest, evidence, trace)  # Multiplicamos y descendemos recursivamente
        else:
            # Y no está fijada -> sumar sobre Y ∈ {True, False}
            total = 0.0  # total acumulará la suma de ambas ramas (True y False)
            if trace:
                print(f"[Suma] {Y} no está en evidencia; sumamos sobre T y F dado {self._fmt_parent_assign(parent_assignment)}")

            for y_val in [True, False]:  # Exploramos ambas asignaciones posibles de Y
                prob = node.p_value_given(y_val, parent_assignment)  # P(Y=y_val|padres)
                # Extender evidencia temporalmente con Y=y_val
                evidence[Y] = y_val  # Fijamos temporalmente Y en la evidencia para la recursión
                # Llamada recursiva
                sub = self._enumerate_all(rest, evidence, trace)  # sub es el producto de abajo con Y fijada
                # Retirar Y para no contaminar otras ramas
                del evidence[Y]  # Eliminamos Y para dejar la evidencia como estaba

                contrib = prob * sub  # contrib es la contribución de esta rama a la suma total
                total += contrib      # Acumulamos

                if trace:
                    print(f"  - {Y}={'T' if y_val else 'F'}: "
                          f"P={prob} * sub={sub}  => contribución={contrib}")

            if trace:
                print(f"[Total] Suma para {Y}: {total}\n")

            return total  # Retornamos la suma de las dos ramas

    @staticmethod
    def _fmt_ev(ev: Dict[str, bool]) -> str:
        """Imprime evidencia tipo {A=T, B=F} (orden alfabético solo para estética)."""
        if not ev:                 # Si el dict está vacío...
            return "{}"            # Mostramos llaves vacías
        parts = [f"{k}={'T' if v else 'F'}" for k, v in sorted(ev.items())]  # Creamos pares ordenados alfabéticamente
        return "{" + ", ".join(parts) + "}"  # Formateamos como {A=T, B=F}

    @staticmethod
    def _fmt_parent_assign(pa: Dict[str, bool]) -> str:
        """Imprime asignación de padres tipo A=T,B=F (orden alfabético para claridad)."""
        if not pa:                      # Si no hay padres...
            return "(sin padres)"       # Texto claro
        parts = [f"{k}={'T' if v else 'F'}" for k, v in sorted(pa.items())]  # Ordenamos y formateamos
        return ",".join(parts)          # "A=T,B=F"


# ------------------------------------------------------------
# Ejemplo / Validación
#   Carga la red Alarm desde estructura.txt y cpts.json,
#   imprime estructura y CPTs,
#   y realiza una inferencia de ejemplo con traza.
# ------------------------------------------------------------
def build_network_from_files(struct_path: str, cpts_path: str) -> BayesianNetwork:
    bn = BayesianNetwork()                              # Creamos una red vacía
    BNFileLoader.load_structure(struct_path, bn)        # Cargamos y conectamos la estructura desde archivo
    BNFileLoader.load_cpts(cpts_path, bn)               # Cargamos las CPTs y las asignamos a cada nodo
    return bn                                           # Devolvemos la red ya lista


def demo_alarm_inference():
    # Rutas esperadas (puedes cambiarlas o pasarlas por argumentos)
    struct_file = "estructura.txt"   # Nombre del archivo de estructura en el mismo directorio
    cpts_file = "cpts.json"          # Nombre del archivo con las tablas de probabilidad

    # Construir la red
    bn = build_network_from_files(struct_file, cpts_file)  # bn es la red completa a partir de archivos

    # Mostrar estructura y CPTs
    print(bn.describe_structure())  # Imprime la estructura en texto plano
    print(bn.describe_cpts())       # Imprime todas las CPTs en formato legible

    # Crear motor de inferencia
    engine = InferenceEngine(bn)    # engine realizará consultas por enumeración

    # Ejemplo clásico de validación:
    # Consulta: P(Burglary | JohnCalls=T, MaryCalls=T)
    evidence = {"JohnCalls": True, "MaryCalls": True}                 # Evidencia de llamadas
    result = engine.query("Burglary", evidence=evidence, trace=True)  # Ejecuta la inferencia con traza

    # Imprimir resultado final (también sale en la traza normalizada)
    print("Resultado final:")
    print(f"P(Burglary=T | JohnCalls=T, MaryCalls=T) = {result[True]:.6f}")   # Probabilidad final para True
    print(f"P(Burglary=F | JohnCalls=T, MaryCalls=T) = {result[False]:.6f}")  # Probabilidad final para False


if __name__ == "__main__":   # Punto de entrada cuando se ejecuta como script
    # Ejecuta el demo si corres: python bayes_enum.py
    demo_alarm_inference()    # Llama al flujo de demostración/validación
