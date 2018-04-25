import os
from hybrid_algorithm import HybridAlgorithm
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import GridSearchCV


def best_params():
    file_path = os.path.expanduser('./data/reviews_stars.csv')
    reader = Reader(line_format='user item rating', sep=',')
    data = Dataset.load_from_file(file_path, reader=reader)

    param_grid = {'collaborative_weight': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}

    gs = GridSearchCV(HybridAlgorithm, param_grid, measures=['rmse', 'mae'], cv=3)
    gs.fit(data)

    with open('./data/results_hybrid.txt', 'a') as file:
        file.write('Score rmse: ' + str(gs.best_score['rmse'])+ '\n')
        file.write('Best parameters rmse: ' + str(gs.best_params['rmse']) + '\n')
        file.write('Score mae: ' + str(gs.best_score['mae']) + '\n')
        file.write('Best parameters mae: ' + str(gs.best_params['mae']) + '\n')


if __name__ == '__main__':
    best_params()