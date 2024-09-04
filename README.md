# Implementación de un agente inteligente para jugar a videojuegos mediante técnicas de Aprendizaje por Refuerzo

Este repositorio contiene la implementación del TFG de **Alejandro Torres Rodríguez**. Este TFG pretende introducir al estudiante en el campo del Aprendizaje por Refuerzo con el objetivo de implementar un agente inteligente capaz de jugar a videjuegos sin información previa sobre el juego.

## Librerías utilizados

Para la realización de este trabajo se han utilizado las siguientes librerías:

1. [Numpy](https://numpy.org/)
2. [Pytorch](https://pytorch.org/)
3. [Matplotlib](https://matplotlib.org/)
4. [Gymnasium](https://gymnasium.farama.org/)

## Instalación

Para instalar y configurar el proyecto sigue los siguientes pasos:

1. Clona este repositorio:
```bash
git clone https://github.com/atorresrod/TFG.git
```
2. Navega a la carpeta del proyecto:
```bash
cd TFG
```
3. Instala las dependencias necesarias:
```bash
pip install -r requirements.txt
```

## Uso

Para entrenar los agentes ejecutar los scripts con prefio ```train``` de la carpeta **scripts**:

```bash
./scripts/train_breakout.sh
```

Para ejecutar la búsqueda de hiperparámetros de los algoritmos tabulares ejecutar el siguiente script:

```bash
./scripts/hyperparameter_tuning.sh
```

Para ejecutar la evaluación de los algoritmos ejecutar los siguientes scripts:

```bash
./scripts/evaluate_dqn.sh
./scripts/evaluate_tabular.sh
```

Para ver a los agentes entrenados jugando a los juegos ejecutar el siguiente script de python:

```bash
./src/play.py <Nombre del entorno> <Ruta del modelo entrenado> <1 si es DQN, 0 si es tabular> <opcional --num_episodes para indicar el número de episodios> <opcional --epsilon para indicar el epsilon a utilizar, por defecto es 0>
```
