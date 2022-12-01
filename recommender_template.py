import numpy as np
import pandas as pd
import recommender_functions as rec_func
import sys  # can use sys to take command line arguments


class Recommender():
    '''
    This Recommender uses FunkSVD to make predictions of exact ratings.  And uses either FunkSVD or a Knowledge Based recommendation (highest ranked) to make recommendations for users.  Finally, if given a movie, the recommender will provide movies that are most similar as a Content Based Recommender.
    '''

    def __init__(self, movies_filepath, reviews_filepath):
        '''
        Attributes:
            movies_filepath(csv file): file containing movies contents
            reviews_filepath(csv file): file containing movies ratings
        '''
        self.movies = pd.read_csv(movies_filepath)
        self.reviews = pd.read_csv(reviews_filepath)

    def fit(self, latent_features=5, learning_rate=0.001, iters=100):
        '''This function performs matrix factorization using a basic form of FunkSVD with no regularization.
        INPUT:
            latent_features(int): number of latent features to take into consideration
            learning_rate(float): value of alpha learning_rate
            iters(int): number of iterations for the recommender

        OUTPUT:
            user_mat(array): numpy array of the U matrix 
            movie_mat( array): numpy array of the V matrix

        '''
        # create user-movie pair matrix and rating values matrix
        arrange = self.reviews[['user_id', 'movie_id', 'rating', 'timestamp']]
        self.user_movie_mat = arrange.groupby(['user_id', 'movie_id'])[
            'rating'].max().unstack()
        ratings_mat = np.array(self.user_movie_mat)

        self.n_users = ratings_mat.shape[0]
        self.n_movies = ratings_mat.shape[1]
        self.ratings_num = np.count_nonzero(~np.isnan(ratings_mat))

        # create random matrix in u matrix and v matrix using latent feature
        user_mat = np.random.rand(self.n_users, latent_features)
        movie_mat = np.random.rand(latent_features, self.n_movies)

        # initialize sse at 0 for first iteration
        sse_accum = 0

        # keep track of iteration and MSE
        print("Optimization Statistics")
        print("Iteration | Mean Square Error")

        for iteration in range(iters):

            # update our sse
            old_sse = sse_accum
            sse_accum = 0

            # for each user-movie pair
            for i in range(self.n_users):
                for j in range(self.n_movies):

                    # if rating exists
                    if ratings_mat[i, j] > 0:
                        # compute error as the actual minus dot product of the user and movie
                        diff = ratings_mat[i, j] - \
                            np.dot(user_mat[i, :], movie_mat[:, j])

                        # keep track of the sum of square errors for the matrix
                        sse_accum += diff**2

                        # update the values in each matrix in the direction of the gradient
                        for k in range(latent_features):
                            user_mat[i, k] += learning_rate * \
                                (2 * diff * movie_mat[k, j])
                            movie_mat[k, j] += learning_rate * \
                                (2 * diff * user_mat[i, k])

            print("%d\t\t %f" % (iteration+1, sse_accum / self.ratings_num))

        self.user_mat = user_mat
        self.movie_mat = movie_mat

    def predict_rating(self, user_id, movie_id):
        '''function to output the prediction of the recommender

        INPUT:
            user_id: the user_id from the reviews df
            movie_id: the movie_id from movies df
        OUTPUT:
            pred: the predicted rating for user_id-movie_id according to FunkSVD
        '''
        # create the ids of users and movies in right order
        user_ids_series = np.array(self.user_movie_mat.index)
        movie_ids_series = np.array(self.user_movie_mat.columns)

        # find the row and column index
        user_row = np.where(user_ids_series == user_id)[0][0]
        movie_col = np.where(movie_ids_series == movie_id)[0][0]

        # make a predict
        pred = np.dot(self.user_mat[user_row, :], self.movie_mat[:, movie_col])

        movie_name = str(
            self.movies[self.movies['movie_id'] == movie_id]['movie'])[5:]
        movie_name = movie_name.replace('\nName: movie, dtype: object', '')
        print("For user {} we predict a {} rating for the movie {}.".format(
            user_id, round(pred, 2), str(movie_name)))

        return pred

    def make_recommendations(self, _id, id_type='movie', rec_num=5):
        ''' function to make movies recommendation for user_id and movie_id

        INPUT:
            _id(int): either a user or movie id
            id_type(string): movie or user
            rec_num(int): nunmber of recommendation to return

        OUTPUT:
             recs(array): list or numpy array of reccommended movies for a given movie or user id
        '''
        try:

            # create index of user_id and get ranked movies df
            val_users = self.user_movie_mat.index
            val_movies = self.user_movie_mat.columns
            self.ranked_movies = rec_func.create_ranked_df(
                self.movies, self.reviews)

            if id_type == 'user':
                if _id in val_users:
                    # get the index of which row the user is in
                    idx = np.where(val_users == _id)[0][0]

                    # make a predict
                    preds = np.dot(self.user_mat[idx, :], self.movie_mat)

                    # pull the top movies from the predictions
                    indices = preds.argsort()[-rec_num:][::-1]
                    rec_ids = self.user_movie_mat.columns[indices]
                    recs = rec_func.get_movie_names(rec_ids, self.movies)

                else:
                    # get top recs movies if user is not found
                    recs = rec_func.popular_recommendations(
                        _id, rec_num, self.ranked_movies)
                    print(
                        "Because this user wasn't in our database, we are giving back the top movie recommendations for all users")

            # Find similar movies if it is a movie that is passed
            else:
                if _id in val_movies:
                    recs = list(rec_func.find_similar_movies(
                        _id, self.movies))[:rec_num]
                else:
                    print(
                        "That movie doesn't exist in our database.  Sorry, we don't have any recommendations for you.")

            return recs

        except:
            print("I'm sorry, but a prediction cannot be made for this user-movie pair.  It looks like one of these items does not exist in our current database.")

            return None


if __name__ == '__main__':
    import recommender_template as rt

    # instantiate recommender
    rec = rt.Recommender('movies_clean.csv', 'train_data.csv')

    # fit recommender
    rec.fit(learning_rate=.01, iters=1)

    # predict
    rec.predict_rating(user_id=8, movie_id=2844)

    # make recommendations
    print(rec.make_recommendations(8, 'user'))  # user in the dataset
    print(rec.make_recommendations(1, 'user'))  # user not in dataset
    print(rec.make_recommendations(1853728))  # movie in the dataset
    print(rec.make_recommendations(1))  # movie not in dataset
    print(rec.n_users)
    print(rec.n_movies)
    print(rec.ratings_num)
