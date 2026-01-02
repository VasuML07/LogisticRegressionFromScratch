import numpy as np
#used for mathematical computations
class LogisticRegression:
    def __init__(self,learning_rate = 0.01,iter = 100):
        #constructor 
        self.learning_rate = learning_rate
        self.iter = iter
        self.weights = None
        self.bias = None
        #setting the hyperparameters
    def sigmoid(self,z):
            return 1/(1+np.exp(-z))
        #sigmoid function which is activation function for logistic regression
    def fit(self,X,y):
            #X is matrix of features
            #y is array of outputs
            m,n = X.shape
            self.weights = np.zeros(n)
            #we intialize all parameters to 0 before training
            self.bias = 0
            for i in range(self.iter):
                linear_model = np.dot(X,self.weights)+self.bias
                #y = w*x+b
                y_pred = self.sigmoid(linear_model)
                #applying the activation function for predictions
                #BackwardPropagation for adjusting paramters by caluclating gradients
                dw  = (1/m)*np.dot(X.T, (y_pred-y))
                db = (1/m)*np.sum(y_pred-y)
                #updating the parameters
                self.weights = self.weights-self.learning_rate*dw
                self.bias = self.bias-self.learning_rate*db
                #printing loss for every 100 iterations
                if i % 100 == 0:
                    cost = -(1/m)*np.sum(y*np.log(y_pred)+(1-y)*np.log(1-y_pred))
    def predict(self, X):
            linear_model = np.dot(X,self.weights)+self.bias
            y_pred = self.sigmoid(linear_model)
            #converting proabilities into classes
            y_pred_cls = [1 if i > 0.5 else 0 for i in y_pred]
            return np.array(y_pred_cls)
if __name__ == "__main__":
    X = np.array([[10.0,10.0],[11.0,15.0],[12.0,12.0],[19.0,15.0],[18.0,20.0]])
    y = np.array([0,0,0,1,1])
    model = LogisticRegression(learning_rate = 0.01,iter = 500)
    #training the model
    model.fit(X,y)
    test_data = np.array([[17.5,22.0]])
    prediction = model.predict(test_data)
    print(f"weight is {model.weights} and bias is {model.bias}")
    print(f"Prediction is Class{prediction[0]}")
