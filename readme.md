Directorio para el Trabajo de Fin de Máster de Inteligencia Artificial y Big Data, por IMF + Universidad Católica de Ávila - Santiago Duque Porras

Estructura:

- [] defensas: directorio con las clases para generar cada modelo con defensa adversarial
	-ataques.py
	-deepfool.py
	-hopskipjump.py
- [] ataques: directorio con las clases para cada método de generación de ataques adversariales
	-modelo_destilado.py
	-modelo_detector.py
	-modelo_ensemble.py
	-modelo_ensemble_destilado.py
	-modelo_naive.py
	-modelo_pgd.py
	
- benchmarking.py - funciones para evaluar modelos y ataques
- data_loading.py - funciones para cargar los datos ya guardados
- data_raw.py - funciones para pre-procesar el dataset original
- model_loading - funciones para cargar los modelos ya guardados
- requirements.txt - fichero de requerimientos de librerías. Para instalar: pip install -r requirements.txt
- utils - funciones auxiliares

- Los siguientes directorios no están incluidos por logística de memoria:
- [] adv_attacks: directorio con los ataques adversariales ya guardados; formato .npy
- [] modelos: directorio con los modelos ya guardados; formatos .keras (Keras.Model) y .index (pesos guardados)