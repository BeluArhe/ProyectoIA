"""
Sistema de Diagnóstico Médico usando Redes Bayesianas con PyMC
Versión completamente funcional
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import arviz as az
import pymc as pm

# Configuración inicial
np.set_printoptions(precision=3, suppress=True)

# ======================
# DEFINICIÓN DE LA RED
# ======================

def crear_modelo_bayesiano(sintomas_observados=None):
    """
    Crea y devuelve el modelo bayesiano para diagnóstico médico con PyMC
    Versión completamente funcional
    """
    with pm.Model() as modelo:
        # 1. Distribución de probabilidad para Predisposición Genética (PG)
        pg_probs = np.array([0.3, 0.4, 0.3])  # Alta, Media, Baja
        pg = pm.Categorical('PG', p=pg_probs)
        
        # Gripe dado PG
        p_gripe = pm.math.switch(
            pm.math.eq(pg, 0), 0.6,  # PG Alta
            pm.math.switch(
                pm.math.eq(pg, 1), 0.4,  # PG Media
                0.2  # PG Baja
            )
        )
        gripe = pm.Bernoulli('Gripe', p=p_gripe)
        
        # Neumonía dado PG
        p_neumonia = pm.math.switch(
            pm.math.eq(pg, 0), 0.7,  # PG Alta
            pm.math.switch(
                pm.math.eq(pg, 1), 0.5,  # PG Media
                0.3  # PG Baja
            )
        )
        neumonia = pm.Bernoulli('Neumonia', p=p_neumonia)
        
        # Fiebre dado Gripe y Neumonía
        p_fiebre = pm.math.switch(
            pm.math.and_(pm.math.eq(gripe, 1), pm.math.eq(neumonia, 1)), 0.9,
            pm.math.switch(
                pm.math.and_(pm.math.eq(gripe, 1), pm.math.eq(neumonia, 0)), 0.8,
                pm.math.switch(
                    pm.math.and_(pm.math.eq(gripe, 0), pm.math.eq(neumonia, 1)), 0.7,
                    0.1  # Ninguna
                )
            )
        )
        fiebre = pm.Bernoulli('Fiebre', p=p_fiebre,
                        observed=sintomas_observados.get('Fiebre') if sintomas_observados and 'Fiebre' in sintomas_observados else None)

        # Tos dado Neumonía
        p_tos = pm.math.switch(
            pm.math.eq(neumonia, 1), 0.8,  # Neumonía=Sí
            0.3  # Neumonía=No
        )
        tos = pm.Bernoulli('Tos', p=p_tos,
                        observed=sintomas_observados.get('Tos') if sintomas_observados and 'Tos' in sintomas_observados else None)

        # Dolor de cabeza dado Gripe
        p_dolor = pm.math.switch(
            pm.math.eq(gripe, 1), 0.7,  # Gripe=Sí
            0.2  # Gripe=No
        )
        dolor_cabeza = pm.Bernoulli('Dolor_cabeza', p=p_dolor,
                        observed=sintomas_observados.get('Dolor_cabeza') if sintomas_observados and 'Dolor_cabeza' in sintomas_observados else None)
    
    return modelo

# ======================
# FUNCIONES DE DIAGNÓSTICO
# ======================

def diagnosticar(sintomas):
    """
    Realiza un diagnóstico basado en los síntomas observados
    Versión corregida y funcional
    """
    try:
        # Define los valores posibles por variable
        valores_posibles = {
            'PG': ['Alta', 'Media', 'Baja'],       # codificados como 0,1,2
            'Gripe': ['No', 'Sí'],                 # codificados como 0,1
            'Neumonia': ['No', 'Sí'],              
            'Fiebre': ['No', 'Sí'],                 
            'Tos': ['No', 'Sí'],
            'Dolor_cabeza': ['No', 'Sí']
        }

        # Convertir síntomas a valores numéricos (1=Sí, 0=No)
        sintomas_numericos = {}
        for nombre, v in sintomas.items():
            if v.lower() in ['sí', 'si', 's']:
                sintomas_numericos[nombre] = 1
                valores_posibles.pop(nombre, None)
            elif v.lower() in ['no', 'n']:
                sintomas_numericos[nombre] = 0
                valores_posibles.pop(nombre, None)
        
        # Crear modelo con las observaciones
        modelo = crear_modelo_bayesiano(sintomas_numericos)
        
        # Realizar inferencia con configuración más robusta
        with modelo:
            trace = pm.sample(
                draws=2000,
                tune=1000,
                chains=2,
                return_inferencedata=True,
                progressbar=True
            )
    
        # Calcular probabilidades manualmente
        resultados = {}
        for enf in valores_posibles:
            muestras = trace.posterior[enf].values.flatten()
            probs = {}
            for idx, estado in enumerate(valores_posibles[enf]):
                probs[estado] = np.mean(muestras == idx)
            resultados[enf] = probs

        return resultados, trace
    
    except Exception as e:
        print(f"\nError durante el diagnóstico: {str(e)}")
        print("Posibles causas:")
        print("- Síntomas no ingresados correctamente (deben ser 'Sí' o 'No')")
        print("- Problemas de convergencia del modelo")
        return None, None

# Resto del código (visualizar_red, mostrar_diagnostico, main) permanece igual...
def mostrar_diagnostico(resultados):
    """Muestra los resultados del diagnóstico de forma clara"""
    if resultados is None:
        print("\nNo se pudo realizar el diagnóstico. Verifique los síntomas ingresados.")
        return
    
    print("\n=== RESULTADOS DEL DIAGNÓSTICO ===")

    for nombre_variable, estados in resultados.items():
        print(f"{nombre_variable}:")
        for estado, prob in estados.items():
            bar = '█' * int(prob * 20)
            print(f"  {estado:<12}: {prob:.3f} |{bar:<20}| {prob * 100:.1f}%")
        print()

def visualizar_red():
    """Visualiza la estructura de la red bayesiana con mejor formato"""
    G = nx.DiGraph()
    
    # Añadir nodos y conexiones
    edges = [
        ("Predisposición\nGenética", "Gripe"),
        ("Predisposición\nGenética", "Neumonía"),
        ("Gripe", "Fiebre"),
        ("Neumonía", "Fiebre"),
        ("Neumonía", "Tos"),
        ("Gripe", "Dolor de\ncabeza")
    ]
    G.add_edges_from(edges)
    
    pos = {
        "Predisposición\nGenética": (0, 1),
        "Gripe": (-1, 0),
        "Neumonía": (1, 0),
        "Fiebre": (0, -1),
        "Tos": (1.5, -0.5),
        "Dolor de\ncabeza": (-1.5, -0.5)
    }
    
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=5000, 
           node_color='lightgreen', font_size=10, 
           font_weight='bold', arrowsize=20, 
           edge_color='gray', width=2)
    plt.title("Red Bayesiana de Diagnóstico Médico", fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()

def main():
    """Función principal con interfaz mejorada"""
    
    # Banner de inicio
    print("""
    ╔══════════════════════════════════╗
    ║  SISTEMA DE DIAGNÓSTICO MÉDICO   ║
    ║       (Modelo Bayesiano)         ║
    ╚══════════════════════════════════╝
    """)
    
    # Visualizar la red al inicio
    print("\n🖥️  Visualizando estructura de la red bayesiana...")
    visualizar_red()
    
    # Casos de prueba demostrativos
    print("\n🔬 Ejecutando casos de prueba demostrativos...")
    
    # Caso 1
    print("\nCaso 1 : Paciente con fiebre y tos")
    print("Síntomas: Fiebre='Sí', Tos='Sí', Dolor_cabeza='No'")
    diag, trace = diagnosticar({'Fiebre': 'Sí', 'Tos': 'Sí', 'Dolor_cabeza': 'No'})
    if diag:
        mostrar_diagnostico(diag)
    else:
        print("No se pudo realizar el diagnóstico. Verifique los síntomas ingresados.")
    
    # Caso 2
    print("\nCaso 2 : Paciente con tos y gripe")
    print("Síntomas: Tos='Sí', Gripe='Sí'")
    diag, trace = diagnosticar({'Tos': 'Sí', 'Gripe': 'Sí'})
    if diag:
        mostrar_diagnostico(diag)
    else:
        print("No se pudo realizar el diagnóstico. Verifique los síntomas ingresados.")
    
    # Modo interactivo mejorado
    # 4. Modo interactivo
    print("\n=== MODO INTERACTIVO ===")
    print("Ingrese los síntomas del paciente (responda 'sí' o 'no' para cada uno)")
    
    while True:
        print("\n" + "═"*50)
        sintomas = {}
        
        # Solicitar síntomas con validación
        for sintoma in ['Fiebre', 'Tos', 'Dolor_cabeza']:
            while True:
                valor = input(f"¿El paciente tiene {sintoma.lower()}? [sí/no]: ").strip()
                if valor.lower() in ['sí', 'si', 's', 'n', 'no']:
                    sintomas[sintoma] = 'Sí' if valor.lower() in ['sí', 'si', 's'] else 'No'
                    break
                print("⚠️ Por favor ingrese 'sí' o 'no'")
        
        # Realizar diagnóstico
        print("\n🔍 Analizando síntomas...")
        diag, trace = diagnosticar(sintomas)
        mostrar_diagnostico(diag)

         # Preguntar por otro diagnóstico
        if input("\n¿Desea realizar otro diagnóstico? [s/n]: ").lower() != 's':
            print("\n👋 ¡Gracias por usar el sistema de diagnóstico!")
            break

if __name__ == "__main__":
    # Configuración adicional para mejor visualización
    plt.style.use('seaborn')
    np.random.seed(42)  # Para reproducibilidad
    
    main()