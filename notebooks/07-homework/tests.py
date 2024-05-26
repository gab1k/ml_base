import pandas as pd
import pytest
import os
from main import get_XY_traint_test, train_model, predict


def preprocess(data_path, model_path, df):
    if os.path.exists(data_path):
        os.remove(data_path)
    if os.path.exists(model_path):
        os.remove(model_path)
    df.to_csv(data_path, encoding='utf-8')


def end_testing(data_path, model_path):
    if os.path.exists(data_path):
        os.remove(data_path)
    if os.path.exists(model_path):
        os.remove(model_path)


def get_correct_df(is_train: bool = False):
    df = pd.DataFrame({'title': ['Ok', 'amazing', 'Terribly'],
                       'text': ['Not bad', 'I really liked the quality of service and the organization',
                                'The flight was canceled 12 times. I had to sit at the airport for more than a day']})
    if is_train:
        df['rating'] = ['4', '5', '1']
    return df


def create_model(data_path, model_path):
    preprocess(data_path, model_path, get_correct_df(is_train=True))
    train_model(data_path, model_path, None, None)
    assert os.path.exists(model_path) is True


def get_incorrect_df():
    return pd.DataFrame({'title': ['Ok', 'amazing', 'Terribly'],
                         'what?': ['i', 'd', 'k']})


def test_train_incorrect_data():
    data_path, model_path = "data.csv", "model.pkl"
    df = get_incorrect_df()
    preprocess(data_path=data_path, model_path=model_path, df=df)
    try:
        train_model(data_path, model_path, None, None)
        assert False
    except Exception as e:
        print(e)
        assert True
    finally:
        assert os.path.exists(model_path) is False
        end_testing(data_path=data_path, model_path=model_path)


def test_train_correct_data():
    data_path, model_path = "data.csv", "model.pkl"
    df = get_correct_df(is_train=True)
    preprocess(data_path=data_path, model_path=model_path, df=df)
    try:
        train_model(data_path, model_path, None, None)
        assert True
    except Exception as e:
        print(e)
        assert False
    finally:
        assert os.path.exists(model_path) is True
        end_testing(data_path=data_path, model_path=model_path)


def test_predict_with_csv(capsys):
    data_path, model_path = "data.csv", "model.pkl"
    create_model(data_path=data_path, model_path=model_path)

    predict(path_to_model=model_path, data=data_path)
    captured = capsys.readouterr()  # "4\n5\n1\n"
    results = captured.out.splitlines()
    for val in results:
        assert 1 <= int(val) <= 5
    assert len(results) == 3
    assert os.path.exists(data_path) is True
    end_testing(data_path=data_path, model_path=model_path)


def test_predict_with_text(capsys):
    data_path, model_path = "data.csv", "model.pkl"
    create_model(data_path=data_path, model_path=model_path)

    predict(path_to_model=model_path, data="Very bad")
    captured = capsys.readouterr()
    results = captured.out.splitlines()  # ['[4]']
    assert 1 <= int(results[0][1]) <= 5
    assert os.path.exists(data_path) is True
    end_testing(data_path=data_path, model_path=model_path)


def test_incorrect_get_XY_traint_test():
    data_path = "Incorrect data Один Два Три.csv"
    assert os.path.exists(data_path) is False
    try:
        get_XY_traint_test(path_to_data=data_path, split=None, path_to_test=None)
        assert False
    except Exception as e:
        print(e)
        assert True
    finally:
        assert os.path.exists(data_path) is False
        end_testing(data_path=data_path, model_path="")


def test_split_get_XY_traint_test():
    data_path, model_path = "data.csv", "wtf.wtf"
    df = pd.DataFrame({'title': ['t1', 't2', 't3', 't4'],
                       'text': ['txt1', 'txt2', 'txt3', 'txt4'],
                       'rating': [1, 2, 3, 4]})
    preprocess(data_path=data_path, model_path=model_path, df=df)
    x_train, x_test, y_train, y_test = get_XY_traint_test(path_to_data=data_path, split=0.2, path_to_test=None)
    assert len(x_train) == len(y_train) == 3
    assert len(x_test) == len(y_test) == 1
    assert y_train.iloc[0] == 4 and y_train.iloc[1] == 1 and y_train.iloc[2] == 3 and y_test.iloc[0] == 2
    end_testing(data_path=data_path, model_path=model_path)


if __name__ == '__main__':
    pytest.main()
