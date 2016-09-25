from rest_framework.response import Response
from .serializers import *
from rest_framework import generics
from .models import *
import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.naive_bayes import BernoulliNB
import numpy as np
import math, pickle, simplejson,ast




class TrainModel(generics.CreateAPIView):
    """
    Generate a precision from the model
    """
    
    serializer_class = TrainingDataSerializer
    queryset = ClassificationModel

    def k_fold_selection(self,training_dataframe, selected_column, ignored_columns):
        """
        Compute the best model using K-fold cross validation
        """
        
        # getting all the outcomes (i.e. from the selected column)
        outcomes = training_dataframe[selected_column].factorize()[0]
        # remove selected column from dataframe
        training_dataframe.drop(selected_column, axis=1, inplace=True)
        
        # remove all 'ignored columns' from dataframe
        ignored_columns = ast.literal_eval(ignored_columns)
        for column in ignored_columns:
            training_dataframe.drop(column, axis=1, inplace=True)
            
        column_states = {}
        # factorize the columns and store the state names of each feature as a dict
        for column in training_dataframe.columns:
            factorized_column = training_dataframe[column].factorize()
            training_dataframe[column] = factorized_column[0]
            column_states[column] = factorized_column[1].tolist()
        
        # intialize the k-fold cross validation object
        kf = KFold(len(outcomes), n_folds=10, shuffle=True, random_state=14)
        
        for iter_id, item in enumerate(kf):
            # for the first iteration, the classification model needs to be initialized at the end
            if iter_id == 0:
                # training segment
                features_train = np.array([list(training_dataframe.ix[i]) for i in item[0]])
                outcomes_train = np.array([outcomes[i] for i in item[0]])
                
                # evaluation segment
                features_test = np.array([list(training_dataframe.ix[i]) for i in item[1]])
                outcomes_test = np.array([outcomes[i] for i in item[1]])
                
                # create the classifier
                best_clf = BernoulliNB()
                best_clf.fit(features_train, outcomes_train)
                best_accuracy = math.ceil((best_clf.score(features_test,outcomes_test, sample_weight=None)*100)*100)/100
                continue
            
            # training segment
            features_train = np.array([list(training_dataframe.ix[i]) for i in item[0]])
            outcomes_train = np.array([outcomes[i] for i in item[0]])
            
            # evaluation segment
            features_test = np.array([list(training_dataframe.ix[i]) for i in item[1]])
            outcomes_test = np.array([outcomes[i] for i in item[1]])
            
            # create the classifier
            clf = BernoulliNB()
            clf.fit(features_train, outcomes_train)
            accuracy = math.ceil((clf.score(features_test,outcomes_test,sample_weight=None)*100)*100)/100
            
            # hold the best model in best_clf
            if accuracy > best_accuracy:
                accuracy = best_accuracy
                best_clf = clf
            
        return best_accuracy, best_clf, column_states, training_dataframe.columns.tolist()


   
    def post(self, request, format=None):
        """
        Handle CSV files uploaded on POST requests
        """
        
        training_dataframe = pd.read_csv(request.FILES['training_file'].file)
        selected_column = request.POST['selected_column']
        ignored_columns = request.POST['ignore_columns']

        accuracy, clf, column_states, features = self.k_fold_selection(training_dataframe,
                                                                       selected_column,
                                                                       ignored_columns
                                                                      )
        new_model = ClassificationModel.objects.create(model_name=request.POST['model_name'],
                                                       features = simplejson.dumps(features),
                                                       selected_column=selected_column,
                                                       )
        
        with open(settings.BASE_DIR + '/data/trained_models/' + str(new_model.id) + '.pickle', 'wb') as f:
            pickle.dump(clf, f)

        return Response([accuracy, new_model.id, column_states])
    
    
    
# {"email":1,"company_name":0,"purpose_of_use":1,"job_title":1,"activity":0}
class CalculatePrediction(generics.CreateAPIView):
    """
    Calculate the prediction for a user
    """
    
    serializer_class = PredictionSerializer
    
    def post(self, request, format=None):
        model_id = request.data['model_id']
        parameters = request.data['parameters']
        classification_model = ClassificationModel.objects.get(id=model_id)
        
        with open(settings.BASE_DIR + '/data/trained_models/' + str(classification_model.id) + '.pickle', 'rb') as f:
            clf = pickle.load(f)
        
        features = ast.literal_eval(classification_model.features)
        parameters = simplejson.loads(parameters)
        
        uncertain_data = []
        
        for feature in features:
            uncertain_data.append(parameters[feature])
        
        complete_prediction = clf.predict_proba(uncertain_data)
        prediction = math.ceil((complete_prediction[0][1]*100)*100)/100
        return Response(prediction)
