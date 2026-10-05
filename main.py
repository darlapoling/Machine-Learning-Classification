#Darla Poling, Oct 5 2026, https://www.digitalocean.com/community/tutorials/how-to-build-a-machine-learning-classifier-in-python-with-scikit-learn
import sklearn
import csv
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#import dependencies

import pandas as pd
import numpy as np
import plotly.express as px

data = load_breast_cancer()
#for row, diagnosis in zip(data.data, data.target):
    #print(dict(zip(data.feature_names, row)), data.target_names[diagnosis])

def clean_data():
    #Organize our data
    label_names = data['target_names']
    labels = data['target']
    feature_names = data['feature_names']
    features = data['data']

    # Look at our data
    print(label_names)
    print(labels[0])
    print(feature_names[0])
    print(features[0])

    #Investigate outliers
    fig = px.scatter(x=data.data[:, 3], y=data.data[:, 23], color=data.target.astype(str))

    fig.show()

    # Split our data
    train, test, train_labels, test_labels = train_test_split(features,
        labels,
        test_size=0.33,
        random_state=42)
    return train, test, train_labels, test_labels

def classify_data(train, test, train_labels, test_labels):
    # Initialize our classifier
    gnb = GaussianNB()

    # Train our classifier
    model = gnb.fit(train, train_labels)


    # Make predictions
    preds = gnb.predict(test)
    print(preds)
    return preds

def evaluate_accuracy(test_labels, preds):
    print(accuracy_score(test_labels, preds))
    # Evaluate accuracy)
    return accuracy_score(test_labels, preds)

def plot_data(accuracy):
    sizes = [accuracy, 1 - accuracy]
    labels = ['Accuracy', 'Inaccuracy']
    
    plt.pie(sizes, labels=labels, startangle=20)
    plt.title('Accuracy Rate')
    plt.axis('equal')  # ensures the pie chart is a perfect circle
    plt.show()

def main():
    train, test, train_labels, test_labels = clean_data()
    preds = classify_data(train, test, train_labels, test_labels)
    accuracy = evaluate_accuracy(test_labels, preds)
    print("Accuracy:", accuracy)
    plot_data(accuracy)

if __name__ == "__main__":
    main()