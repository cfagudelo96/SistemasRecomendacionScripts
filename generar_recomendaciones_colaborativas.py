import os
import pandas as pd
from surprise import SVD
from surprise import Dataset
from surprise import Reader


def generate_recommendations():
    file_path = os.path.expanduser('./data/reviews_stars.csv')
    reader = Reader(line_format='user item rating', sep=',')
    data = Dataset.load_from_file(file_path, reader=reader)

    trainset = data.build_full_trainset()

    algo = SVD(n_factors=5, n_epochs=25, lr_all=0.006, reg_all=0.2, biased=True)
    algo.fit(trainset)

    df = pd.read_csv('./data/reviews_stars.csv', header=None, names=['user', 'business', 'review'])
    businesses = df['business'].unique()
    users = df.groupby('user')['business'].nunique()

    with open('./data/collaborative_recomendations.csv', 'w') as file:
        for user, user_count in users.items():
            for business in businesses:
                if user_count > 12:
                    pred = algo.predict(user, business)
                    if pred.est > 4.2:
                        file.write(user + ',' + business + ',' + str(pred.est) + '\n')
            print(user)

generate_recommendations()
