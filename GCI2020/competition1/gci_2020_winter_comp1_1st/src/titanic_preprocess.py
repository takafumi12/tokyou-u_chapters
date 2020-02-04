import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
import optuna
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestRegressor
from optuna.integration import lightgbm_tuner as lgb_tuner

# ---Adversarial Validation---
def preprocess_av(data, col_to_use=None):
    """Adversarial Validationの前処理用の関数
    大体のAUCが把握出来れば良いので、新しい特徴量を作成したりする必要はない。
    """
    def process_null_av(data):
        """欠損値処理用の関数
        最初は簡単に欠損値処理をする。
        数値変数のAgeとFareは中央値、カテゴリー変数Embarkedは最頻値で補完する。
        Cabinは欠損が多いので、欠損カテゴリーMを作る。
        """
        data.fillna({
                'Age': data['Age'].median(),
                'Cabin': 'M',
                'Embarked': data['Embarked'].mode(),
                'Fare': data['Fare'].median()
            },
            inplace=True
        )
        return data
    

    def feature_engineering_av(data):
        """特徴量エンジニアリング用の関数
        Cabinから細かい数字を落としてデッキ(ABCDEFGTM)に変換する。
        カテゴリー変数はone-hot encodingを使う。
        """
        data['Deck'] = data['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
        data.drop('Cabin', axis=1, inplace=True)
        data = pd.get_dummies(data)
        
        return data

    col_to_use = list(data.columns) if col_to_use is None else col_to_use
    data = data[col_to_use]
    data = process_null_av(data)
    data = feature_engineering_av(data)
          
    return data


def split_train_valid_test_data(data, target, col_to_use=None, test_size=0.25, valid_size=0.25):
    """全データを訓練/検証/テストデータに分ける関数
    全てのインスタンスに対してラベルが存在している必要がある。
    """
    col_to_use = list(data.columns) if col_to_use is None else col_to_use
    data = data[col_to_use]
    X = copy.deepcopy(data.drop(target, axis=1))
    y = copy.deepcopy(data[target])
    X_tv, X_test, y_tv, y_test = train_test_split(X, y,
                                                  test_size=test_size,
                                                  shuffle=True,
                                                  random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_tv, y_tv,
                                                          test_size=valid_size,
                                                          shuffle=True,
                                                          random_state=42)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def train_lgb_with_optuna(X_train, y_train, X_valid, y_valid,
                          lgb_params=None, num_boost_round=100,
                          early_stopping_rounds=10):
    """OptunaでLGBをチューニングする関数
    """
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
    } if lgb_params is None else lgb_params

    tuned_lgb = lgb_tuner.train(lgb_params, lgb_train,
                                     valid_sets=lgb_valid,
                                     num_boost_round=num_boost_round,
                                     early_stopping_rounds=early_stopping_rounds,
                                     verbose_eval=False,
                                     )
    return tuned_lgb


def adversarial_validation_titanic(data, col_to_use, print_score=True):
    """Adversarial Validationを行う関数
    """
    data = preprocess_av(data, col_to_use)
    X_train, y_train, X_valid, y_valid, X_test, y_test = \
        split_train_valid_test_data(data, target='IsTrain')
    tuned_lgb = train_lgb_with_optuna(X_train, y_train, X_valid, y_valid)
    y_pred = tuned_lgb.predict(X_test)
    score = roc_auc_score(y_test, y_pred)
    return tuned_lgb, y_pred, score


# ---kNN--
def preprocess_knn(data):
    """KNeighborsClassifierの訓練のためにデータを前処理する関数。
    """
    def fill_na_knn(data):
        # 年齢は敬称ごとの中央値で補完する
        data['Title'] = data['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
        mapping = {
            'Mlle': 'Miss',
            'Major': 'Mr',
            'Col': 'Mr',
            'Sir': 'Mr',
            'Don': 'Mr',
            'Mme': 'Miss',
            'Jonkheer': 'Mr',
            'Lady': 'Mrs',
            'Capt': 'Mr',
            'Countess': 'Mrs',
            'Ms': 'Miss',
            'Dona': 'Mrs'
        }
        data.replace({'Title': mapping}, inplace=True)
        titles = ['Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Rev']
        for title in titles:
            age_to_impute = data.groupby('Title')['Age'].median()[titles.index(title)]
            data.loc[(data['Age'].isnull()) & (data['Title'] == title), 'Age'] = age_to_impute

        # 運賃の値は平均値で補完する
        data['Fare'].fillna(data['Fare'].mean(), inplace=True)
        
        return data
        
    def feature_engineering_knn(data):
        # 家族サイズ
        data['FamilySize'] = data['Parch'] + data['SibSp'] + 1 # +1ない場合は？

        # 家族もしくは同じチケットで搭乗したグループの生存率を特徴量として追加
        # (Target Encodingなのでリークと過学習に注意する必要がある)
        # (参考: https://blog.amedama.jp/entry/target-mean-encoding-types)
        data['Surname'] = data['Name'].apply(lambda x: str.split(x, ",")[0])
        data['FamilySurvival'] = 0.5
        for grp, grp_df in data[['Survived','Name', 'Surname', 'Fare', 'Ticket', 'PassengerId',
                                   'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Surname', 'Fare']): 
            if (len(grp_df) != 1):
                for idx, row in grp_df.iterrows():
                    smax = grp_df.drop(idx)['Survived'].max()
                    smin = grp_df.drop(idx)['Survived'].min()
                    passenger_idx = row['PassengerId']
                    if (smax == 1.0):
                        data.loc[data['PassengerId'] == passenger_idx, 'FamilySurvival'] = 1
                    elif (smin==0.0):
                        data.loc[data['PassengerId'] == passenger_idx, 'FamilySurvival'] = 0
        for _, grp_df in data.groupby('Ticket'):
            if (len(grp_df) != 1):
                for idx, row in grp_df.iterrows():
                    if (row['FamilySurvival'] == 0) | (row['FamilySurvival']== 0.5):
                        smax = grp_df.drop(idx)['Survived'].max()
                        smin = grp_df.drop(idx)['Survived'].min()
                        passenger_id = row['PassengerId']
                        if (smax == 1.0):
                            data.loc[data['PassengerId'] == passenger_id, 'FamilySurvival'] = 1
                        elif (smin==0.0):
                            data.loc[data['PassengerId'] == passenger_id, 'FamilySurvival'] = 0

        # 運賃のビニング
        data['FareBin'] = pd.qcut(data['Fare'], 5, labels=False)

        # 年齢のビニング
        data['AgeBin'] = pd.qcut(data['Age'], 4, labels=False)

        # 性別の変換
        data['Sex'].replace({'male': 0,'female': 1}, inplace=True)

        return data
        
    data = fill_na_knn(data)
    data = feature_engineering_knn(data)
    return data


# ---Random Forest---
def preprocess_rf(data):
    """RandomForestClassifierの訓練のためにデータを前処理する関数。
    """
    def fill_na_rf(data):
        # Embarkedの補完
        data['Embarked'] = data['Embarked'].fillna('S')
        
        # Ageの補完
        age_df = data[['Age', 'Pclass', 'Sex', 'Parch', 'SibSp']]
        age_df = pd.get_dummies(age_df)
        known_age = age_df[age_df['Age'].notnull()].as_matrix()
        unknown_age = age_df[age_df['Age'].isnull()].as_matrix()
        y = known_age[:, 0]
        X = known_age[:, 1:]
        rfr = RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1)
        rfr.fit(X, y)
        predicted_ages = rfr.predict(unknown_age[:, 1::])
        data.loc[(data['Age'].isnull()), 'Age'] = predicted_ages
        
        # Fareの補完
        fare_median = data.loc[(data['Embarked'] == "S") & (data['Pclass'] == 3), 'Fare'].median()
        data['Fare'] = data['Fare'].fillna(fare_median)
        
        # Cabinの補完
        data['Cabin'] = data['Cabin'].fillna('M')
        
        return data
        
    def feature_engineering_rf(data):
        # CabinからDeck情報の抽出
        data['Deck'] = data['Cabin'].str.get(0)
        
        # 敬称の変換
        title_dict = {
            "Capt": "Officer",
            "Col": "Officer",
            "Major": "Officer",
            "Jonkheer": "Master",
            "Don": "Royalty",
            "Sir" : "Royalty",
            "Dr": "Officer",
            "Rev": "Officer",
            "the Countess": "Royalty",
            "Mme": "Mrs",
            "Mlle": "Miss",
            "Ms": "Mrs",
            "Mr" : "Mr",
            "Mrs" : "Mrs",
            "Miss" : "Miss",
            "Master" : "Master",
            "Lady" : "Royalty",
            "Dona" : "Royalty"    
        }
        data['Title'] = data['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
        data['Title'] = data['Title'].map(title_dict)

        # 名字での家族のグルーピング
        # 家族内の生存率の計算
        # 女子供が全員死亡した家族と、男が全員生存した家族の特徴量の置換
        # 以上を直接的に予測に利用したのがWoman_Child_Groupモデル
        data['Surname'] = data['Name'].map(lambda name:name.split(',')[0].strip())
        data['FamilyGroup'] = data['Surname'].map(data['Surname'].value_counts())
        woman_child_group=data.loc[(data['FamilyGroup']>=2) & ((data['Age']<=16) | (data['Sex']=='female'))]
        woman_child_group=woman_child_group.groupby('Surname')['Survived'].mean()
        dead_list=set(woman_child_group[woman_child_group.apply(lambda x:x==0)].index)
        man_group=data.loc[(data['FamilyGroup']>=2) & (data['Age']>16) & (data['Sex']=='male')]
        man_group=man_group.groupby('Surname')['Survived'].mean()
        survived_list=set(man_group[man_group.apply(lambda x:x==1)].index)
        data.loc[(data['Survived'].isnull()) & (data['Surname'].apply(lambda x:x in dead_list)),\
             ['Sex','Age','Title']] = ['male',28.0,'Mr']
        data.loc[(data['Survived'].isnull()) & (data['Surname'].apply(lambda x:x in survived_list)),\
             ['Sex','Age','Title']] = ['female',5.0,'Miss']
        
        # 家族の総数の計算とEDAを利用した3グループへのラベルエンコーディング
        # EDAから家族の数が{1, 5, 6, 7}と{2, 3, 4}では生存率に大きな差があるとわかる
        # よって、単純に家族の数を特徴量として用いるのではなく、生存率の大小でラベリングする
        # 家族数が8以上のデータは訓練データに含まれていないので、テストデータにあった場合は0とする
        data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
        data['FamilyLabel'] = 0
        data.loc[(data['FamilySize']>=2) & (data['FamilySize']<=4), 'FamilyLabel'] = 2
        data.loc[((data['FamilySize']>4) & (data['FamilySize']<=7)) | (data['FamilySize']==1), 'FamilyLabel'] = 1
        
        # 同じくTicketの枚数の計算とEDAを利用した3グループへのラベルエンコーディング
        data['TicketGroup'] = data.groupby('Ticket').transform('count')['PassengerId']
        data.loc[(data['TicketGroup']>=2) & (data['TicketGroup']<=4), 'TicketGroup'] = 2
        data.loc[((data['TicketGroup']>4) & (data['TicketGroup']<=8)) | (data['TicketGroup']==1), 'TicketGroup'] = 1
        data.loc[(data['TicketGroup']>8), 'TicketGroup'] = 0
        
        return data
        
    data = fill_na_rf(data)
    data = feature_engineering_rf(data)      
    return data


# ---Helpers---
def split_preprocessed_data(data, target, col_to_use=None, scaler=None):
    """trainデータとtestデータを結合して前処理した全データを分割する関数
    """
    col_to_use = list(data.columns) if col_to_use is None else col_to_use
    data = data[col_to_use]
    data = pd.get_dummies(data)
    X_train = data[data[target].notnull()].drop(target, axis=1)
    y_train = data[data[target].notnull()][target]  
    X_test = data[data[target].isnull()].drop(target, axis=1)
    
    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_test

        