# MovieRatings-recommendation
##  Project Description
MovieTweetings is a dataset consisting of ratings on movies that were contained in well-structured tweets on Twitter. These dataset comprise of movies and reviews of this movies which was wrangled and interested data were collected into train_data.csv.
The idea of this project was to provide a technique for movies recommendation on knownledge based, collaborative filtering and content based recommendations. The simple matrix factorization which is Singular Value Decomposition(SVD) couldn't work because nan value poses challenge for SVD. FunkSVD technique perfectly work around the nan value converge to provide U matrix and V matrix of 5 latent features.

## Files
1. movies_clean.csv: contain movies dataset
1. recommender_function.py: python file for knowledge based and content based recommendation stretegy
1. recommender_template.py: python file for Recommender class that take care of matrix factorization and make predictiona and movies recommendation
1. [result_img.png](/Result_img.png): Snapshot of the Recommender in making predictions and recommendations
train_data.csv: reviews dataset that contain movies ratings

![image result](/Result_img.png)