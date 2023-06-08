import numpy as np
import pandas as pd
from enum import IntEnum
from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model import LabelModel
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from pydantic import BaseModel


class ChatLabel(IntEnum):
    ACTION = 0
    REVIEW = 1
    RECYCLE = 2
    ABSTAIN = -1


class ValuesNotEmpty(BaseModel):
    vectorizer: CountVectorizer
    svm: SVC
    mlp: MLPClassifier

    class Config:
        arbitrary_types_allowed = True


class WeakLearner:

    def __init__(self, vectorizer=None, svm=None, mlp=None, label_model=None):
        self.vectorizer = vectorizer
        self.svm = svm
        self.mlp = mlp
        self.label_model = label_model

    def create_label_applier(self):
        ValuesNotEmpty(vectorizer=self.vectorizer, svm=self.svm, mlp=self.mlp)

        # create the weak learners
        @labeling_function()
        def lf_svm_rbf(x):
            x = self.vectorizer.transform(x).toarray()
            y_pred = self.svm.predict(x)
            y_prob = self.svm.predict_proba(x)
            mx = np.max(y_prob, axis=1)
            idx = np.where(mx < 0.75)
            y_pred[idx[0]] = int(ChatLabel.ABSTAIN)
            return y_pred[0]

        @labeling_function()
        def lf_mlp(x):
            x = self.vectorizer.transform(x)
            if (x.shape[0] > 1):
                return None

            y_pred = self.mlp.predict(x)
            y_prob = self.mlp.predict_proba(x)
            y_pred = y_pred[0]
            mx = np.max(y_prob, axis=1)
            if (mx[0] < 0.75):
                y_pred = int(ChatLabel.ABSTAIN)
            return y_pred

        @labeling_function()
        def lf_channel(x):
            x = x['message']
            if any([
                (x.find('joined the channel') >= 0),
                (x.find('added to the channel') >= 0)
            ]):
                return ChatLabel.RECYCLE
            return ChatLabel.ABSTAIN

        @labeling_function()
        def lf_length(x):
            if (len(x['message']) < 6):
                return ChatLabel.RECYCLE
            return ChatLabel.ABSTAIN

        @labeling_function()
        def lf_hello(x):
            if any([
                (x['message'].find('hello') >= 0),
                (x['message'].find('hola') >= 0),
                (x['message'].find('good morning') >= 0),
                (x['message'].find('good evening') >= 0),
                (x['message'].find('good night') >= 0)
            ]):
                return ChatLabel.RECYCLE
            return ChatLabel.ABSTAIN

        @labeling_function()
        def lf_roger(x):
            if any([
                (x['message'].find('rgr') >= 0),
                (x['message'].find('roger') >= 0)
            ]):
                return ChatLabel.RECYCLE
            return ChatLabel.ABSTAIN

        @labeling_function()
        def lf_lunch(x):
            if any([
                (x['message'].find('lunch') >= 0),
                (x['message'].find('dinner') >= 0),
                (x['message'].find('food') >= 0)
            ]):
                return ChatLabel.RECYCLE
            return ChatLabel.ABSTAIN

        labeling_functions = [lf_svm_rbf, lf_mlp, lf_channel,
                              lf_length, lf_hello, lf_roger, lf_lunch]
        label_applier = PandasLFApplier(labeling_functions)
        return labeling_functions, label_applier

    def train_weak_learners(self, filepath_train):

        # train the classifiers
        data_train = pd.read_csv(filepath_train)
        data_train['message'] = data_train['message'].astype(str)
        data_train = data_train[data_train['createat'].notnull()]
        self.vectorizer = CountVectorizer(stop_words="english")
        x_train = self.vectorizer.fit_transform(data_train['message'])
        y_train = data_train['labels']

        self.svm = SVC(gamma=2, C=1, probability=True)
        self.mlp = MLPClassifier(alpha=1, max_iter=1000)
        self.svm.fit(x_train.toarray(), y_train)
        self.mlp.fit(x_train.toarray(), y_train)

        labeling_functions, label_applier = self.create_label_applier()
        L_train = label_applier.apply(pd.DataFrame(data_train['message']))

        # train the weak learner and apply it to the dataset
        LFAnalysis(L=L_train, lfs=labeling_functions).lf_summary()
        self.label_model = LabelModel(cardinality=6, verbose=True)
        self.label_model.fit(L_train=L_train, n_epochs=500,
                             log_freq=100, seed=123)
        return (self.vectorizer, self.svm, self.mlp, self.label_model)

    def get_label_model(self):
        return self.label_model
