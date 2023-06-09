import numpy as np
RANDOM_SEED = 5


class NaiveBayes(object):
    def __init__(self):
        pass

    def likelihood_ratio(self, ratings):  # [5pts]
        '''
        Args:
            rating_stars: a python list of numpy arrays that is length <number of labels> x 1
            
            Example rating_stars for Five-label NB model:
    
            ratings_stars = [ratings_1, ratings_2, ratings_3, ratings_4, ratings_5] -- length 5
            ratings_1: N_ratings_1 x D
                where N_ratings_1 is the number of reviews that gave an Amazon
                product a 1-star rating and D is the number of features (we use the word count as the feature)
            ratings_2: N_ratings_2 x D
                where N_ratings_2 is the number of reviews that gave an Amazon
                product a 2-star rating and D is the number of features (we use the word count as the feature)
            ratings_3: N_ratings_3 x D
                where N_ratings_3 is the number of reviews that gave an Amazon
                product a 3-star rating and D is the number of features (we use the word count as the feature)
            ratings_4: N_ratings_4 x D
                where N_ratings_4 is the number of reviews that gave an Amazon
                product a 4-star rating and D is the number of features (we use the word count as the feature)
            ratings_5: N_ratings_5 x D
                where N_ratings_5 is the number of reviews that gave an Amazon
                product a 5-star rating and D is the number of features (we use the word count as the feature)
            
            If you look at the end of this python file, you will see a docstring that contains more examples!
            
        Return:
            likelihood_ratio: (<number of labels>, D) numpy array, the likelihood ratio of different words for the different classes of ratings.
        '''
        _, D = np.shape(ratings[0])
        likelihood_ratio = np.zeros((len(ratings),D))
        for i in range(len(ratings)): 
            denominator = (np.sum(ratings[i]) + D)
            for j in range(D):
                nominator = np.sum(ratings[i][:,j]) + 1
                likelihood_ratio[i][j] = nominator/denominator
            
        return likelihood_ratio


    def priors_prob(self, ratings):  # [5pts]
        '''
        Args:
            rating_stars: a python list of numpy arrays that is length <number of labels> x 1
            
            Example rating_stars for Five-label NB model:
    
            ratings_stars = [ratings_1, ratings_2, ratings_3, ratings_4, ratings_5] -- length 5
            ratings_1: N_ratings_1 x D
                where N_ratings_1 is the number of reviews that gave an Amazon
                product a 1-star rating and D is the number of features (we use the word count as the feature)
            ratings_2: N_ratings_2 x D
                where N_ratings_2 is the number of reviews that gave an Amazon
                product a 2-star rating and D is the number of features (we use the word count as the feature)
            ratings_3: N_ratings_3 x D
                where N_ratings_3 is the number of reviews that gave an Amazon
                product a 3-star rating and D is the number of features (we use the word count as the feature)
            ratings_4: N_ratings_4 x D
                where N_ratings_4 is the number of reviews that gave an Amazon
                product a 4-star rating and D is the number of features (we use the word count as the feature)
            ratings_5: N_ratings_5 x D
                where N_ratings_5 is the number of reviews that gave an Amazon
                product a 5-star rating and D is the number of features (we use the word count as the feature)
            
            If you look at the end of this python file, you will see a docstring that contains more examples!
            
        Return:
            priors_prob: (1, <number of labels>) numpy array, where each entry denotes the prior probability for each class
        '''
        
        priors_prob = np.zeros((len(ratings),1))
        summ = 0
        for i in range(len(ratings)):
            summ += np.sum(ratings[i])

        for i in range(len(ratings)):
            nominator = np.sum(ratings[i])
            priors_prob[i] = nominator/summ

        return priors_prob.T


    # [5pts]
    def analyze_star_rating(self, likelihood_ratio, priors_prob, X_test):
        '''
        Args:
            likelihood_ratio: (<number of labels>, D) numpy array, the likelihood ratio of different words for different classes of ratings
            priors_prob: (1, <number of labels>) numpy array, where each entry denotes the prior probability for each class
            X_test: (N_test, D) numpy array, a bag of words representation of the N_test number of ratings that we need to analyze
        Return:
            ratings: (N_test,) numpy array, where each entry is a class label specific for the Naïve Bayes model
        '''
        ratings = np.zeros((len(X_test),))
        for i in range(len(X_test)):
            t = np.prod(likelihood_ratio ** X_test[i], axis= 1) * priors_prob 
            ratings[i] = np.argmax(t)
        return ratings


'''
ADDITIONAL EXAMPLES for ratings_stars
ratings_stars: Python list that contains the labels per corresponding Naive Bayes models.
The length of ratings will change depending on which Naive Bayes model we are training.
You are highly encouraged to use a for-loop to iterate over ratings!
------------------------------------------------------------------------------------------------------------------------
Two-label NB model:
ratings_stars = [ratings_less_than_or_equal_to_2, ratings_greater_or_equal_to_3] -- length 2
ratings_less_than_or_equal_to_2: N_ratings_less_than_or_equal_to_2 x D
    where N_ratings_less_than_or_equal_to_2 is the number of reviews that gave an Amazon
    product a 1 or 2-star rating and D is the number of features (we use the word count as the feature)
ratings_greater_or_equal_to_3: N_ratings_greater_or_equal_to_3 x D
    where N_ratings_greater_or_equal_to_3 is the number of reviews that gave an Amazon
    product a 3, 4, or 5-star rating and D is the number of features (we use the word count as the feature)
------------------------------------------------------------------------------------------------------------------------
Three-label NB model:
ratings_stars = [ratings_less_than_or_equal_to_2, ratings_3, ratings_greater_or_equal_to_4] -- length 3
ratings_less_than_or_equal_to_2: N_ratings_less_than_or_equal_to_2 x D
    where N_ratings_less_than_or_equal_to_2 is the number of reviews that gave an Amazon
    product a 1 or 2-star rating and D is the number of features (we use the word count as the feature)
ratings_3: N_ratings_3 x D
    where N_ratings_3 is the number of reviews that gave an Amazon
    product a rating a 3-star and D is the number of features (we use the word count as the feature)
ratings_greater_or_equal_to_4: N_ratings_greater_or_equal_to_4 x D
    where N_ratings_greater_or_equal_to_4 is the number of reviews that gave an Amazon
    product a 4 or 5-star rating and D is the number of features (we use the word count as the feature)
------------------------------------------------------------------------------------------------------------------------
Four-label NB model:
ratings_stars = [ratings_less_than_or_equal_to_2, ratings_3, ratings_4, ratings_5] -- length 4
ratings_less_than_or_equal_to_2: N_ratings_less_than_or_equal_to_2 x D
    where N_ratings_less_than_or_equal_to_2 is the number of reviews that gave an Amazon
    product a 1 or 2-star rating and D is the number of features (we use the word count as the feature)
ratings_3: N_ratings_3 x D
    where N_ratings_3 is the number of reviews that gave an Amazon
    product a 3-star rating and D is the number of features (we use the word count as the feature)
ratings_4: N_ratings_4 x D
    where N_ratings_4 is the number of reviews that gave an Amazon
    product a 4-star rating and D is the number of features (we use the word count as the feature)
ratings_5: N_ratings_5 x D
    where N_ratings_5 is the number of reviews that gave an Amazon
    product a 5-star rating and D is the number of features (we use the word count as the feature)
------------------------------------------------------------------------------------------------------------------------
Five-label NB model:
ratings_stars = [ratings_1, ratings_2, ratings_3, ratings_4, ratings_5] -- length 5
ratings_1: N_ratings_1 x D
    where N_ratings_1 is the number of reviews that gave an Amazon
    product a 1-star rating and D is the number of features (we use the word count as the feature)
ratings_2: N_ratings_2 x D
    where N_ratings_2 is the number of reviews that gave an Amazon
    product a 2-star rating and D is the number of features (we use the word count as the feature)
ratings_3: N_ratings_3 x D
    where N_ratings_3 is the number of reviews that gave an Amazon
    product a 3-star rating and D is the number of features (we use the word count as the feature)
ratings_4: N_ratings_4 x D
    where N_ratings_4 is the number of reviews that gave an Amazon
    product a 4-star rating and D is the number of features (we use the word count as the feature)
ratings_5: N_ratings_5 x D
    where N_ratings_5 is the number of reviews that gave an Amazon
    product a 5-star rating and D is the number of features (we use the word count as the feature)
------------------------------------------------------------------------------------------------------------------------
*** Note, the variables inside the list are just placeholders. Do not reference with these variable names! ***
'''
