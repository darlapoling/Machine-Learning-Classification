import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = load_breast_cancer()


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

def main():
    train, test, train_labels, test_labels = clean_data()
    preds = classify_data(train, test, train_labels, test_labels)
    accuracy = evaluate_accuracy(test_labels, preds)
    print("Accuracy:", accuracy)

if __name__ == "__main__":
    main()