import os
from surprise import SVD
from surprise import Dataset
from surprise import Reader

from surprise.model_selection import GridSearchCV


def best_params():
    # Dataset de reviews a utilizar
    file_path = os.path.expanduser('./data/reviews_stars.csv')
    reader = Reader(line_format='user item rating', sep=',')
    data = Dataset.load_from_file(file_path, reader=reader)

    # Se crea una lista de posibles valores de factores
    n_factors_values = []
    n_factors_initial_value = 2
    # Se prueban 15 distintos factores en intervalos de 2
    for i in range(0, 15):
        n_factors_values.append(n_factors_initial_value + (n_factors_initial_value * i))

    # Se crea una lista de posibles epochs
    n_epochs_values = []
    n_epochs_initial_value = 5
    # Se prueba 10 valores distintos en intervalos de 5
    for i in range(0, 10):
        n_epochs_values.append(n_epochs_initial_value + (n_epochs_initial_value * i))

    # Se crea una lista de posibles parámetros de regularización
    reg_all_values = []
    reg_all_initial_value = 0.2
    # Se prueban 5 valores distintos en intervalos de 0.2
    for i in range(0, 5):
        reg_all_values.append(reg_all_initial_value + (reg_all_initial_value * i))

    # Se crea una lista de posibles learning rates
    lr_all_values = []
    lr_all_initial_value = 0.002
    # Se prueban 5 valores distintos en intervalos de 0.002
    for i in range(0, 5):
        lr_all_values.append(lr_all_initial_value + (lr_all_initial_value * i))

    # Se crea el diccionario de parámetros
    param_grid = {
        'n_factors': n_factors_values,
        'n_epochs': n_epochs_values,
        'lr_all': lr_all_values,
        'reg_all': reg_all_values,
        'biased': [True]
    }
    # Se prueban los parámetros utilizando MAE y RMSE
    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)

    gs.fit(data)

    # Se escribe en un archivo los resultados de los mejores parámetros RMSE y MAE
    with open('./data/results.txt', 'a') as file:
        file.write('Score rmse: ' + str(gs.best_score['rmse'])+ '\n')
        file.write('Best parameters rmse: ' + str(gs.best_params['rmse']) + '\n')
        file.write('Score mae: ' + str(gs.best_score['mae']) + '\n')
        file.write('Best parameters mae: ' + str(gs.best_params['mae']) + '\n')


if __name__ == '__main__':
    best_params()

