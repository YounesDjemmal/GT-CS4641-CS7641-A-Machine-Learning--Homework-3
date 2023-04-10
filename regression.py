import numpy as np

class Regression(object):
    
    def __init__(self):
        pass
    
    def rmse(self, pred, label): # [5pts]
        """
        Calculate the root mean square error.

        Args:
            pred: (N, 1) numpy array, the predicted labels
            label: (N, 1) numpy array, the ground truth labels
        Return:
            A float value
        """
        rmse = np.sqrt(np.mean((pred - label) ** 2))
        return rmse
    
    def construct_polynomial_feats(self, x, degree): # [5pts]
        """
        Args:
            x: N x D numpy array, where N is number of instances and D is the
               dimensionality of each instance.
            degree: the max polynomial degree
        Return:
            feat: 
                For 1-D array, numpy array of shape Nx(degree+1), remember to include
                the bias term. feat is in the format of:
                [[1.0, x1, x1^2, x1^3, ....,],
                 [1.0, x2, x2^2, x2^3, ....,],
                 ......
                ]
                
                For D-dimensional array: numpy array of shape N x (degree+1) x D, remember to include
                  the bias term.

                Example: print(feat)
                For an input where N=3, D=2, and degree=3...

                [[[ 1.0        1.0]
                  [ x_{1,1}    x_{1,1}]
                  [ x_{1,1}^2  x_{1,2}^2]
                  [ x_{1,1}^3  x_{1,2}^3]]

                 [[ 1.0        1.0]
                  [ x_{2,1}    x_{2,2}]
                  [ x_{2,1}^2  x_{2,2}^2]
                  [ x_{2,1}^3  x_{2,2}^3]]

                 [[ 1.0        1.0]
                  [ x_{3,1}    x_{3,2}]
                  [ x_{3,1}^2  x_{3,2}^2]
                  [ x_{3,1}^3  x_{3,2}^3]]]

        """
        N = x.shape[0]
        if(x.ndim == 1):
            feat = np.ones((len(x), degree + 1))
            for d in range(1,degree + 1):
                feat[:, d] = x ** d
        else:
            N,D = np.shape(x)
            feat = np.ones((N, degree + 1, D))
            for d in range(1,degree + 1):
                feat[:, d, :] = x ** d
        return feat

    def predict(self, xtest, weight): # [5pts]
        """
        Args:
            xtest: (N,D) numpy array, where N is the number 
                   of instances and D is the dimensionality
                   of each instance
            weight: (D,1) numpy array, the weights of linear regression model
        Return:
            prediction: (N,1) numpy array, the predicted labels
        """
        
        return np.dot(xtest, weight)

    # =================
    # LINEAR REGRESSION
    # Hints:
    # - In the fit function, use close form solution of the linear regression to get weights. 
    # - For inverse, you can use numpy linear algebra function (np.linalg.pinv) 
    # - For the predict, you need to use linear combination of data points and their weights (y = theta0*1+theta1*X1+...)

    def linear_fit_closed(self, xtrain, ytrain): # [5pts]
        """
        Args:
            xtrain: (N,D) numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: (N,1) numpy array, the true labels
        Return:
            weight: (D,1) numpy array, the weights of linear regression model
        """
        temp = np.linalg.pinv(np.dot(xtrain.T, xtrain))
        t = np.dot(temp, xtrain.T)
        weight = np.dot(t, ytrain)
        return weight
        

    def linear_fit_GD(self, xtrain, ytrain, epochs=5, learning_rate=0.001): # [5pts]
        """
        Args:
            xtrain: (N,D) numpy array, where N is number 
                    of instances and D is the dimensionality of each 
                    instance
            ytrain: (N,1) numpy array, the true labels
        Return:
            weight: (D,1) numpy array, the weights of linear regression model
            loss_per_epoch: (epochs,) list of floats, rmse of each epoch
        """
        N, D = np.shape(xtrain)
        weight = np.zeros((D, 1))
        loss_per_epoch = []
        for epoch in range(epochs):
            weight = weight + learning_rate * (np.dot(xtrain.T, (ytrain - np.dot(xtrain, weight)))) / N
            pred = self.predict(xtrain, weight)
            error = self.rmse(pred, ytrain)
            loss_per_epoch.append(error)
        return weight, loss_per_epoch

    def linear_fit_SGD(self, xtrain, ytrain, epochs=100, learning_rate=0.001): # [5pts]
        """
        Args:
            xtrain: (N,D) numpy array, where N is number 
                    of instances and D is the dimensionality of each 
                    instance
            ytrain: (N,1) numpy array, the true labels
        Return:
            weight: (D,1) numpy array, the weights of linear regression model
            loss_per_step: (N*epochs,) list of floats, rmse calculated after each update step

        NOTE: For autograder purposes, iterate through the dataset SEQUENTIALLY, NOT stochastically.

        Note: Keep in mind that the number of epochs is the number of
        complete passes through the training dataset. SGD updates the
        weight for one datapoint at a time, but for each epoch, you'll
        need to go through all of the points.
        """
        N, D = np.shape(xtrain)
        weight = np.zeros((D, 1))
        loss_per_epoch = []
        for epoch in range(epochs):
            for i in range(N):
                delta = (xtrain[i].reshape((D,1)) * (ytrain[i] - np.dot(xtrain[i].T, weight))/ N ).reshape((D,1))
                weight = (weight + learning_rate * delta).reshape((D,1))
                pred = self.predict(xtrain, weight)
                error = self.rmse(pred, ytrain)
                loss_per_epoch.append(error)

        return weight, loss_per_epoch


    # =================
    # RIDGE REGRESSION
        
    def ridge_fit_closed(self, xtrain, ytrain, c_lambda): # [5pts]
        """
        Args:
            xtrain: (N,D) numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: (N,1) numpy array, the true labels
            c_lambda: float value
        Return:
            weight: (D,1) numpy array, the weights of ridge regression model
        """
        identity = np.identity(xtrain.shape[1])
        identity[0][0] = 0.0 
        weight = np.dot(np.dot(np.linalg.pinv(np.dot( xtrain.T, xtrain) +c_lambda * identity), xtrain.T), ytrain)
        return weight
         

        
    def ridge_fit_GD(self, xtrain, ytrain, c_lambda, epochs=500, learning_rate=1e-7): # [5pts]
        """
        Args:
            xtrain: (N,D) numpy array, where N is number 
                    of instances and D is the dimensionality of each 
                    instance
            ytrain: (N,1) numpy array, the true labels
            c_lambda: float value
        Return:
            weight: (D,1) numpy array, the weights of linear regression model
            loss_per_epoch: (epochs,) list of floats, rmse of each epoch
        """
        N, D = np.shape(xtrain)
        weight = np.zeros((D, 1))
        loss_per_epoch = []
        for epoch in range(epochs):
            # grad =  2/N * xtrain.T.dot(xtrain.dot(weight) - ytrain) + 2 * c_lambda * weight
            # weight = weight - learning_rate * grad

            grad = xtrain.T @ (ytrain - (xtrain @ weight)) / N + (2 * c_lambda * learning_rate) 
            weight += weight * (2 * c_lambda * learning_rate) + learning_rate * grad 

            pred = self.predict(xtrain, weight)
            error = self.rmse(pred, ytrain)
            loss_per_epoch.append(error)

        return weight, loss_per_epoch

    def ridge_fit_SGD(self, xtrain, ytrain, c_lambda, epochs=100, learning_rate=0.001): # [5pts]
        """
        Args:
            xtrain: (N,D) numpy array, where N is number 
                    of instances and D is the dimensionality of each 
                    instance
            ytrain: (N,1) numpy array, the true labels
        Return:
            weight: (D,1) numpy array, the weights of linear regression model
            loss_per_step: (N*epochs,) list of floats, rmse calculated after each update step

        NOTE: For autograder purposes, iterate through the dataset SEQUENTIALLY, NOT stochastically.
            
        Note: Keep in mind that the number of epochs is the number of
        complete passes through the training dataset. SGD updates the
        weight for one datapoint at a time, but for each epoch, you'll
        need to go through all of the points.
        """
        N, D = np.shape(xtrain)
        weight = np.zeros((D, 1))      
        loss_per_epoch = []  
        for e in range(epochs):
            i = np.random.choice(N, N)
            x = xtrain[i,:]
            y = ytrain[i,:]

            grad = 2/xtrain.shape[0] * x.T.dot(x.dot(weight) - y) + 2*c_lambda*weight
            grad *= learning_rate
            weight -= grad * learning_rate
            pred = self.predict(xtrain, weight)
            error = self.rmse(pred, ytrain)
            loss_per_epoch.append(error)

        return weight, loss_per_epoch

    def ridge_cross_validation(self, X, y, kfold=10, c_lambda=100): # [5 pts]
        """
        Args: 
            X : (N,D) numpy array, where N is the number of instances and D is the dimensionality of each instance
            y : (N,1) numpy array, true labels
            kfold: Number of folds you should take while implementing cross validation.
            c_lambda: Value of regularization constant
        Returns:
            meanErrors: float, average rmse error
        Hint: np.concatenate might be helpful.
        Look at 3.5 to see how this function is being used.
        # For cross validation, use 10-fold method and only use it for your training data (you already have the train_indices to get training data).
        # For the training data, split them in 10 folds which means that use 10 percent of training data for test and 90 percent for training.
        """
        N, D = np.shape(X)
        fold = int(N / kfold)
        meanErrors = 0.0
        for i in range(kfold):
            x1 = X[ : i*fold, :]
            x2 = X[(i + 1)*fold:, :]
            xtrain = np.concatenate((x1, x2))
            y1 = y[:i*fold]
            y2 = y[(i + 1)*fold:]
            ytrain = np.concatenate((y1, y2))

            weight = self.ridge_fit_closed(xtrain, ytrain, c_lambda)

            xtest = X[i * fold:(i + 1) * fold, :]
            ytest = y[i * fold:(i + 1) * fold]
            pred = self.predict(xtest, weight)
            error = self.rmse(pred, ytest)
            meanErrors += error

        meanErrors /= kfold
        return meanErrors