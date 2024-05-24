import click
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = str(text)
    lm = WordNetLemmatizer()
    text = re.sub(r"[^\w\s]", " ", text.lower()).split()
    text = " ".join([lm.lemmatize(word) for word in text if word not in stop_words])
    return text


def get_XY_traint_test(path_to_data, split, path_to_test):
    # return x_train, x_test, y_train, y_test
    try:
        data = pd.read_csv(path_to_data)
        df = data[['rating']].copy()
        df['text'] = (data[["title"]].squeeze() + " " + data[["text"]].squeeze())
        df['preprocessed_text'] = df['text'].apply(preprocess_text)
        if split:
            return train_test_split(df['preprocessed_text'], df['rating'],
                                    test_size=float(split), random_state=42)
        elif path_to_test:
            test_data = pd.read_csv(path_to_test)
            df_test = test_data[['rating']].copy()
            df_test['text'] = (test_data[["title"]].squeeze() + " " + test_data[["text"]].squeeze())
            df_test['preprocessed_text'] = df_test['text'].apply(preprocess_text)
            return df['preprocessed_text'], df_test['preprocessed_text'], df['rating'], df_test['rating']
        else:
            return df['preprocessed_text'], None, df['rating'], None

    except Exception as e:
        raise Exception(e)


def train_model(path_to_data, path_to_model, split, path_to_test):
    if split and path_to_test:
        raise Exception("You cannot use a separate dataframe and split at the same time for testing")
    try:
        x_train, x_test, y_train, y_test = get_XY_traint_test(path_to_data, split, path_to_test)
        model = Pipeline(
            [
                ("vectorizer", TfidfVectorizer()),
                ("model", SVC(kernel='linear')),
            ]
        )
        model.fit(x_train, y_train)
        if x_test is not None and y_test is not None:
            y_pred = model.predict(x_test)
            score = f1_score(y_test, y_pred, average='micro')
            print(f"F1 score on test: {score}")
            print(f"Tests values is: {y_pred}")
        with open(path_to_model, 'w+b') as file:
            pickle.dump(model, file)
    except Exception as e:
        raise Exception(e)


def predict(path_to_model, data):
    try:
        with open(path_to_model, 'rb') as file:
            model = pickle.load(file)
            if len(data) > 4 and data[-4:] == ".csv":
                got_data = pd.read_csv(data)
                df = got_data[['text']].copy()
                df['text'] = (got_data[["title"]].squeeze() + " " + got_data[["text"]].squeeze())
                df['preprocessed_text'] = df['text'].apply(preprocess_text)
                res = model.predict(df['preprocessed_text'])
                for el in res:
                    print(el)
            else:
                print(model.predict([preprocess_text(data)]))

    except Exception as e:
        raise Exception(e)


@click.command()
@click.option('--data', required=False, help='Number of greetings.')
@click.option('--test', required=False, help='Data for testing.')
@click.option('--split', required=False, help='Volume separation of the test sample from data.')
@click.option('--model', required=False, help='The model for the test.')
@click.argument('command')
def main(command, data, test, split, model):
    if not data or not model:
        raise Exception('You must provide data and model')
    if command == 'train':
        if not os.path.isfile(data):
            raise Exception('Data file does not exist')
        train_model(path_to_data=data, path_to_model=model, split=split, path_to_test=test)

    elif command == 'predict':
        if not os.path.isfile(model):
            raise Exception('Model does not exist')
        predict(path_to_model=model, data=data)

    else:
        raise Exception("Invalid command")


if __name__ == '__main__':
    main()
