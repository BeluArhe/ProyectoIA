"""
Sistema de Diagnóstico Médico usando Redes Bayesianas
Versión corregida y funcional
"""

from pomegranate import *
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Configuración inicial
np.set_printoptions(precision=3, suppress=True)  # Formato de salida numérica

# ======================
# DEFINICIÓN DE LA RED
# ======================

def crear_red_bayesiana():
    """
    Crea y devuelve la red bayesiana para diagnóstico médico
    """
    
    # 1. Distribución de probabilidad para Predisposición Genética (PG)
    pg = DiscreteDistribution({
        'Alta': 0.3,
        'Media': 0.4,
        'Baja': 0.3
    })
    
    # 2. Distribuciones condicionales
    # Gripe dado PG
    gripe = ConditionalProbabilityTable([
        ['Alta', 'Sí', 0.6],
        ['Alta', 'No', 0.4],
        ['Media', 'Sí', 0.4],
        ['Media', 'No', 0.6],
        ['Baja', 'Sí', 0.2],
        ['Baja', 'No', 0.8]
    ], [pg])
    
    # Neumonía dado PG
    neumonia = ConditionalProbabilityTable([
        ['Alta', 'Sí', 0.7],
        ['Alta', 'No', 0.3],
        ['Media', 'Sí', 0.5],
        ['Media', 'No', 0.5],
        ['Baja', 'Sí', 0.3],
        ['Baja', 'No', 0.7]
    ], [pg])
    
    # Fiebre dado Gripe y Neumonía
    fiebre = ConditionalProbabilityTable([
        ['Sí', 'Sí', 'Sí', 0.9],
        ['Sí', 'Sí', 'No', 0.1],
        ['Sí', 'No', 'Sí', 0.8],
        ['Sí', 'No', 'No', 0.2],
        ['No', 'Sí', 'Sí', 0.7],
        ['No', 'Sí', 'No', 0.3],
        ['No', 'No', 'Sí', 0.1],
        ['No', 'No', 'No', 0.9]
    ], [gripe, neumonia])
    
    # Tos dado Neumonía
    tos = ConditionalProbabilityTable([
        ['Sí', 'Sí', 0.8],
        ['Sí', 'No', 0.2],
        ['No', 'Sí', 0.3],
        ['No', 'No', 0.7]
    ], [neumonia])
    
    # Dolor de cabeza dado Gripe
    dolor_cabeza = ConditionalProbabilityTable([
        ['Sí', 'Sí', 0.7],
        ['Sí', 'No', 0.3],
        ['No', 'Sí', 0.2],
        ['No', 'No', 0.8]
    ], [gripe])
    
    # 3. Crear nodos
    nodo_pg = Node(pg, name="PG")
    nodo_gripe = Node(gripe, name="Gripe")
    nodo_neumonia = Node(neumonia, name="Neumonia")
    nodo_fiebre = Node(fiebre, name="Fiebre")
    nodo_tos = Node(tos, name="Tos")
    nodo_dolor = Node(dolor_cabeza, name="Dolor_cabeza")
    
    # 4. Construir la red bayesiana
    modelo = BayesianNetwork("Diagnóstico Médico")
    modelo.add_states(nodo_pg, nodo_gripe, nodo_neumonia, 
                     nodo_fiebre, nodo_tos, nodo_dolor)
    
    # 5. Añadir conexiones
    modelo.add_edge(nodo_pg, nodo_gripe)
    modelo.add_edge(nodo_pg, nodo_neumonia)
    modelo.add_edge(nodo_gripe, nodo_fiebre)
    modelo.add_edge(nodo_neumonia, nodo_fiebre)
    modelo.add_edge(nodo_neumonia, nodo_tos)
    modelo.add_edge(nodo_gripe, nodo_dolor)
    
    # 6. Compilar el modelo
    modelo.bake()
    
    return modelo

# ======================
# FUNCIONES DE DIAGNÓSTICO
# ======================

def diagnosticar(modelo, sintomas):
    """
    Realiza un diagnóstico basado en los síntomas observados
    """
    try:
        # Preparar las evidencias para pomegranate
        evidencias = {}
        for nombre, valor in sintomas.items():
            # Mapear nombres de síntomas a nombres de nodos
            if nombre == 'Fiebre':
                evidencias['Fiebre'] = valor
            elif nombre == 'Tos':
                evidencias['Tos'] = valor
            elif nombre == 'Dolor_cabeza':
                evidencias['Dolor_cabeza'] = valor
        
        # Realizar inferencia
        result = modelo.predict_proba(evidencias)
        
        # Obtener resultados relevantes
        diagnostico = {
            'PG': result[0].parameters[0],
            'Gripe': result[1].parameters[0],
            'Neumonia': result[2].parameters[0]
        }
        
        return diagnostico
    
    except Exception as e:
        print(f"Error en diagnóstico: {str(e)}")
        return None

def mostrar_diagnostico(diagnostico):
    """Muestra los resultados del diagnóstico"""
    print("\n=== RESULTADOS DEL DIAGNÓSTICO ===")
    for enfermedad, prob in diagnostico.items():
        print(f"\n{enfermedad}:")
        for estado, valor in prob.items():
            print(f"  P({estado}) = {valor:.3f}")

# ======================
# INTERFAZ PRINCIPAL
# ======================

def main():
    """Función principal del sistema de diagnóstico"""
    
    print("""
    ====================================
    SISTEMA DE DIAGNÓSTICO MÉDICO
    ====================================
    """)
    
    # 1. Crear la red bayesiana
    print("Construyendo red bayesiana...")
    modelo = crear_red_bayesiana()
    print("✅ Red bayesiana creada exitosamente!")
    
    # 2. Visualizar la estructura de la red
    try:
        G = nx.DiGraph()
        edges = [(edge[0].name, edge[1].name) for edge in modelo.edges]
        G.add_edges_from(edges)
        
        pos = nx.spring_layout(G)
        plt.figure(figsize=(10, 6))
        nx.draw(G, pos, with_labels=True, node_size=3000, 
               node_color='lightblue', font_size=12, 
               font_weight='bold', arrowsize=20)
        plt.title("Estructura de la Red Bayesiana", fontsize=14)
        plt.show()
    except Exception as e:
        print(f"⚠️ No se pudo generar el gráfico: {str(e)}")
    
    # 3. Ejemplos de diagnóstico
    print("\nEjecutando casos de prueba...")
    
    # Caso 1: Paciente con fiebre y tos
    print("\nCaso 1: Fiebre='Sí', Tos='Sí', Dolor_cabeza='No'")
    diag = diagnosticar(modelo, {'Fiebre': 'Sí', 'Tos': 'Sí', 'Dolor_cabeza': 'No'})
    mostrar_diagnostico(diag)
    
    # Caso 2: Paciente solo con dolor de cabeza
    print("\nCaso 2: Fiebre='No', Tos='No', Dolor_cabeza='Sí'")
    diag = diagnosticar(modelo, {'Fiebre': 'No', 'Tos': 'No', 'Dolor_cabeza': 'Sí'})
    mostrar_diagnostico(diag)
    
    # 4. Modo interactivo
    print("\n=== MODO INTERACTIVO ===")
    print("Ingrese los síntomas del paciente (deje vacío para omitir)")
    
    sintomas = {}
    for sintoma in ['Fiebre', 'Tos', 'Dolor_cabeza']:
        valor = input(f"{sintoma} (Sí/No): ").strip().capitalize()
        if valor in ['Sí', 'No']:
            sintomas[sintoma] = valor
    
    if sintomas:
        diag = diagnosticar(modelo, sintomas)
        mostrar_diagnostico(diag)
    else:
        print("No se ingresaron síntomas válidos")

if __name__ == "__main__":
    main()