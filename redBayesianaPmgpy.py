from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Configuración inicial
np.set_printoptions(precision=3, suppress=True)  # Formato de salida numérica

# ======================
# DEFINICIÓN DE LA RED
# ======================

def crear_red_pgmpy():
    modelo = BayesianNetwork([
        ('PG', 'Gripe'),
        ('PG', 'Neumonia'),
        ('Gripe', 'Fiebre'),
        ('Neumonia', 'Fiebre'),
        ('Neumonia', 'Tos'),
        ('Gripe', 'Dolor_cabeza')
    ])
    
    # CPDs
    cpd_pg = TabularCPD(variable='PG', variable_card=3,
                        values=[[0.3], [0.4], [0.3]],
                        state_names={'PG': ['Alta', 'Media', 'Baja']})

    cpd_gripe = TabularCPD(variable='Gripe', variable_card=2,
                           values=[[0.6, 0.4, 0.2],
                                   [0.4, 0.6, 0.8]],
                           evidence=['PG'], evidence_card=[3],
                           state_names={'Gripe': ['Sí', 'No'], 'PG': ['Alta', 'Media', 'Baja']})

    cpd_neumonia = TabularCPD(variable='Neumonia', variable_card=2,
                               values=[[0.7, 0.5, 0.3],  # Sí
                                       [0.3, 0.5, 0.7]], # No
                               evidence=['PG'], evidence_card=[3],
                               state_names={'Neumonia': ['Sí', 'No'], 'PG': ['Alta', 'Media', 'Baja']})

    cpd_fiebre = TabularCPD(variable='Fiebre', variable_card=2,
                            values=[
                                [0.9, 0.8, 0.7, 0.1],  # Sí
                                [0.1, 0.2, 0.3, 0.9]   # No
                            ],
                            evidence=['Gripe', 'Neumonia'],
                            evidence_card=[2, 2],
                            state_names={'Fiebre': ['Sí', 'No'], 'Gripe': ['Sí', 'No'], 'Neumonia': ['Sí', 'No']})

    cpd_tos = TabularCPD(variable='Tos', variable_card=2,
                         values=[
                             [0.8, 0.2],  # Sí
                             [0.2, 0.8]   # No
                         ],
                         evidence=['Neumonia'], evidence_card=[2],
                         state_names={'Tos': ['Sí', 'No'], 'Neumonia': ['Sí', 'No']})

    cpd_dolor = TabularCPD(variable='Dolor_cabeza', variable_card=2,
                           values=[
                               [0.7, 0.3],  # Sí
                               [0.3, 0.7]   # No
                           ],
                           evidence=['Gripe'], evidence_card=[2],
                           state_names={'Dolor_cabeza': ['Sí', 'No'], 'Gripe': ['Sí', 'No']})

    modelo.add_cpds(cpd_pg, cpd_gripe, cpd_neumonia, cpd_fiebre, cpd_tos, cpd_dolor)

    # Validar modelo
    assert modelo.check_model()
    
    return modelo

# ======================
# FUNCIONES DE DIAGNÓSTICO
# ======================

def diagnosticar_pgmpy(modelo, sintomas):
    """
    Realiza un diagnóstico usando pgmpy basado en los síntomas observados.
    """
    try:
        # Convertir nombres y valores observados a evidencia
        evidencias = {}
        enfermedades = ['PG','Gripe','Neumonia','Tos','Fiebre','Dolor_cabeza']
        for nombre, valor in sintomas.items():
            if nombre in ['Fiebre', 'Tos', 'Dolor_cabeza','Gripe','Neumonia','PG']:
                evidencias[nombre] = valor
                enfermedades.remove(nombre)

        # Crear motor de inferencia
        infer = VariableElimination(modelo)

        # Diccionario para guardar resultados
        diagnostico = {}

        # Consultar la distribución posterior de cada enfermedad
        for enf in enfermedades:
            resultado = infer.query(variables=[enf], evidence=evidencias)
            diagnostico[enf] = resultado.values  # Esto es un array con las probabilidades
        return diagnostico

    except Exception as e:
        print(f"Error en diagnóstico: {str(e)}")
        return None

def mostrar_diagnostico_pgmpy(diagnostico, estados_posibles):
    """Muestra los resultados del diagnóstico (pgmpy)"""
    print("\n=== RESULTADOS DEL DIAGNÓSTICO ===")
    for enfermedad, probs in diagnostico.items():
        print(f"\n{enfermedad}:")
        for estado, valor in zip(estados_posibles[enfermedad], probs):
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
    modelo = crear_red_pgmpy()
    print("✅ Red bayesiana creada exitosamente!")

    # 2. Visualizar la estructura de la red
    try:
        G = nx.DiGraph()
        edges = list(modelo.edges())
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

    # 3. Ejemplo de diagnóstico
    print("\nEjecutando caso de prueba...")

    print("\nCaso 1: Fiebre='Sí', Tos='Sí', Dolor_cabeza='No'")
    estados_posibles = {
        'PG': ['Alta', 'Media', 'Baja'],
        'Gripe': ['Sí', 'No'],
        'Neumonia': ['Sí', 'No']
    }
    diag = diagnosticar_pgmpy(modelo, {'Fiebre': 'Sí', 'Tos': 'Sí', 'Dolor_cabeza': 'No'})
    mostrar_diagnostico_pgmpy(diag, estados_posibles)

    print("\nCaso 2: Tos='Sí', Gripe='Sí'")
    estados_posibles = {
        'PG': ['Alta', 'Media', 'Baja'],
        'Neumonia': ['Sí', 'No'],
        'Fiebre': ['Sí','No'],
        'Dolor_cabeza': ['Sí', 'No']
    }
    diag = diagnosticar_pgmpy(modelo, {'Tos': 'Sí', 'Gripe': 'Sí'})
    mostrar_diagnostico_pgmpy(diag, estados_posibles)

    # 4. Modo interactivo
    print("\n=== MODO INTERACTIVO ===")
    print("Ingrese los síntomas del paciente (deje vacío para omitir)")
    
    estados_posibles = {
        'PG': ['Alta', 'Media', 'Baja'],
        'Gripe': ['Sí', 'No'],
        'Neumonia': ['Sí', 'No']
    }

    sintomas = {}
    for sintoma in ['Fiebre', 'Tos', 'Dolor_cabeza']:
        valor = input(f"{sintoma} (Sí/No): ").strip().capitalize()
        if valor in ['Sí', 'No']:
            sintomas[sintoma] = valor
    
    if sintomas:
        diag = diagnosticar_pgmpy(modelo, sintomas)
        mostrar_diagnostico_pgmpy(diag, estados_posibles)
    else:
        print("No se ingresaron síntomas válidos")

if __name__ == "__main__":
    main()