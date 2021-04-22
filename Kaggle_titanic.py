import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')
sns.set(font_scale=2.5)
import missingno as msno

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

###############################################

df_train = pd.read_csv('train.csv')
# df_test  = pd.read_csv('test.csv')
# df_test.loc[df_test.Fare.isnull(), 'Fare'] = df_test['Fare'].mean()


# ## 결측치 확인 case1 (msg로 rate 확인)
# for col in df_train.columns:
#     msg = 'column : {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_train[col].isnull().sum() / df_train[col].shape[0]))
#     print(msg)

## 결측치 확인 case2 (missingno의 그래프를 활용한 결측치 확인1)
# msno.matrix(df = df_train.iloc[:, :], figsize = (8, 8), color = (0.8, 0.5, 0.2))

## 결측치 확인 case3 (missingno의 그래프를 활용한 결측치 확인2)
# msno.bar(df = df_train.iloc[:, :], figsize=(8, 8), color = (.8, .5, .2))

### 1.2 Target label 확인
# f, ax = plt.subplots(1, 2, figsize=(18, 8))
# df_train['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax = ax[0], shadow = True)
# ax[0].set_title('Pie plot - Survived')
# ax[0].set_ylabel('')
# sns.countplot(x = 'Survived', data = df_train, ax = ax[1])
# ax[1].set_title('Count plot - Survived')
# plt.show()

### 2. EDA

## 2.1 Pclass
# f, ax = plt.subplots(1, 2, figsize = (18, 8))
# df_train['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'], ax=ax[0])
# ax[0].set_title('Number of Passengers By Pclass', y = 1.02)
# ax[0].set_ylabel('Count')
# sns.countplot('Pclass', data = df_train, ax = ax[1], hue = 'Survived')
# ax[1].set_title('Pclass: Survived vs Dead', y = 1.02)
# ax[1].set_ylabel('')
# plt.show()

## 2.2 Sex
# f, ax = plt.subplots(1, 2, figsize = (18, 8))
# df_train[['Sex', 'Survived']].groupby(['Sex'], as_index = True).mean().plot.bar(ax = ax[0])
# ax[0].set_title('Survived vs Sex')
# sns.countplot(x = 'Sex', hue = 'Survived', data = df_train, ax = ax[1])
# plt.show()

# print(pd.crosstab(df_train['Sex'], df_train['Survived'], margins=True))

## 2.3 Both Sex and Pclass
# # sns.factorplot('Pclass', 'Survived', hue = 'Sex', data = df_train, size = 6, aspect = 1.5)
# sns.factorplot(x = 'Sex', y = 'Survived', col = 'Pclass',
#                data = df_train, aspect = 1)
# plt.show()

# ## 2.4 Age
# print(f"제일 나이 많은 탑승객 : {df_train['Age'].max()}")
# print("제일 어린 탑승객 : {:.1f} years".format(df_train['Age'].min()))
# print("탑승객 평균 나이 : {:.1f} years".format(df_train['Age'].mean()))

# ## 생존에 따른 연령 히스토그램 (커널 밀도 추정, Kernel Density Estimator)
# fig, ax = plt.subplots(1, 1, figsize=(9, 5))
# sns.kdeplot(df_train[df_train['Survived'] == 1]['Age'], ax = ax, shade=True)
# sns.kdeplot(df_train[df_train['Survived'] == 0]['Age'], ax = ax, shade=True)
# plt.legend(['Survived == 1', 'Survived == 0'])
# plt.show()

# ## 각 Pclass별 연령 분포 확인
# plt.figure(figsize=(8,6))
# df_train[df_train['Pclass'] == 1]['Age'].plot(kind = 'kde')
# df_train[df_train['Pclass'] == 2]['Age'].plot(kind = 'kde')
# df_train[df_train['Pclass'] == 3]['Age'].plot(kind = 'kde')
# plt.show()

# ## 나이대에 따른 생존률 변화
# commulate_survival_ratio = []
# for i in range(1, 81):
#     ratio = df_train[df_train['Age'] < i]['Survived'].sum() / len(df_train[df_train['Age'] < i]['Survived'])
#     commulate_survival_ratio.append(ratio)
# plt.figure(figsize = (7, 7))
# plt.plot(commulate_survival_ratio)
# plt.title('Survival rate change depending on range of Age', y = 1.02)
# plt.ylabel('Survival rate')
# plt.xlabel('Range of Age(0 ~ x)')
# plt.show()

# ### 2.5 Pclass, Sex, Age
# f, ax = plt.subplots(1, 2, figsize = (18,8))
# sns.violinplot('Pclass', 'Age', hue = 'Survived', data = df_train, scale = 'count', split = True, ax = ax[0])
# ax[0].set_title('Pclass and Age vs Survived')
# ax[0].set_yticks(range(0, 110, 10))
# sns.violinplot('Sex', 'Age', hue = 'Survived', data = df_train, scale = 'count', split = True, ax = ax[1])
# ax[1].set_title('Sex and Age vs Survived')
# ax[1].set_yticks(range(0, 110, 10))
# plt.show()

# ### 2.6 Emabarked
# df_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index = True).mean().sort_values(by = 'Survived', ascending = False).plot.bar()
# plt.show()

# ## 여러 Feature들로 split
# f, ax = plt.subplots(2, 2, figsize = (20, 15))
# sns.countplot('Embarked', data = df_train, ax = ax[0, 0])
# ax[0, 0].set_title('(1) No. Of Passengers Boraded')
# sns.countplot('Embarked', hue = 'Sex', data = df_train, ax = ax[0, 1])
# ax[0, 1].set_title('(2) Male-Female Split For Embarked')
# sns.countplot('Embarked', hue = 'Survived', data = df_train, ax = ax[1, 0])
# ax[1, 0].set_title('Embarked vs Survived')
# sns.countplot('Embarked', hue = 'Pclass', data = df_train, ax = ax[1, 1])
# ax[1, 1].set_title('Embarked vs Pclass')
# plt.show()

### 2.7 Family - SibSp(형제 자매) + Parch(부모, 자녀)
# 형제자매 + 부모자녀 = Family
df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1 # 자신 포함
# df_test['FamilySize'] = df_test['Sibsp'] + df_test['Parch'] + 1

# ## FamilySize와 생존 관계 확인
# f, ax = plt.subplots(1, 3, figsize = (40, 10))
# sns.countplot('FamilySize', data = df_train, ax = ax[0])
# ax[0].set_title('(1) No. Of Passengers Boarded', y = 1.02)
# sns.countplot('FamilySize', hue = 'Survived', data = df_train, ax = ax[1] )
# ax[1].set_title('(2) Survived countplot depending on FamilySize', y = 1.02)
# df_train[['FamilySize', 'Survived']].groupby('FamilySize').mean().sort_values('Survived', ascending = False).plot.bar(ax = ax[2])
# ax[2].set_title('(3) Survived rate depending on FamilySize', y = 1.02)
# plt.subplots_adjust(wspace = .2, hspace = .5)
# plt.show()

# ### 2.8 Fare
df_train['Fare'] = df_train['Fare'].map(lambda x : np.log(x) if x > 0 else 0)
# df_test['Fare'] = df_test['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
# fig, ax = plt.subplots(1, 1, figsize = (8, 8))
# g = sns.distplot(df_train['Fare'], color = 'b', label = f"Skewness : {df_train['Fare'].skew()}", ax = ax)
# g = g.legend(loc = 'best')
# plt.show()

### 2.9 Cabin
# 결측치가 많아(77.1%) 모델에 포함시키지 않음

# ### 2.10 Ticket
# df_train['Ticket'].value_counts()


##############################################
#########   3. Feature engineering  ##########
##############################################

# ### 3.1 Fill Null
# ### 3.1.1 Fill Null in Age using title
df_train['Initial']= df_train.Name.str.extract('([A-Za-z]+)\.')
# df_test['Initial']= df_train.Name.str.extract('([A-Za-z]+)\.')
#
df_train['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)

# df_test['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
#                         ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)

# df_train.groupby('Initial')['Survived'].mean().sort_values(ascending = False).plot.bar()
# plt.show()

### 각 Initial 별 Age 평균으로 결측치 치환
df_train.loc[(df_train.Age.isnull()) & (df_train.Initial == 'Master'), 'Age'] = 5
df_train.loc[(df_train.Age.isnull()) & (df_train.Initial == 'Miss'), 'Age'] = 22
df_train.loc[(df_train.Age.isnull()) & (df_train.Initial == 'Mr'), 'Age'] = 33
df_train.loc[(df_train.Age.isnull()) & (df_train.Initial == 'Mrs'), 'Age'] = 36
df_train.loc[(df_train.Age.isnull()) & (df_train.Initial == 'Other'), 'Age'] = 46

# df_test.loc[(df_train.Age.isnull()) & (df_train.Initial == 'Master'), 'Age'] = 5
# df_test.loc[(df_train.Age.isnull()) & (df_train.Initial == 'Miss'), 'Age'] = 22
# df_test.loc[(df_train.Age.isnull()) & (df_train.Initial == 'Mr'), 'Age'] = 33
# df_test.loc[(df_train.Age.isnull()) & (df_train.Initial == 'Mrs'), 'Age'] = 36
# df_test.loc[(df_train.Age.isnull()) & (df_train.Initial == 'Other'), 'Age'] = 46

### 3.1.2 Fill Null in Embarked
# Embarked는 S에서 가장 많은 탑승객이 있었으므로 Null을 S로 치환
df_train['Embarked'].fillna('S', inplace = True)


# ### 3.2 Change Age(continuous to categorical)
def category_age(x):
    if x < 10:
        return 0
    elif x < 20:
        return 1
    elif x < 30:
        return 2
    elif x < 40:
        return 3
    elif x < 50:
        return 4
    elif x < 60:
        return 5
    elif x < 70:
        return 6
    else:
        return 7
df_train['Age_cat'] = df_train['Age'].apply(category_age)
# df_test['Age_cat'] = df_test['Age'].apply(category_age)

### 3.2 Change Initial, Embarked and Sex (string to numerical)
# 각 카테고리들을 컴퓨터가 인식할 수 있도록 수치화 시켜줘야함(w. map method)
# Initial 카테고리 수치화
df_train['Initial'] = df_train['Initial'].map({'Master':0, 'Miss':1, 'Mr':2, 'Mrs':3, 'Other':4})
# df_test['Initial'] = df_test['Initial'].map({'Master':0, 'Miss':1, 'Mr':2, 'Mrs':3, 'Other':4})

# Embarked 카테고리 수치화
df_train['Embarked'] = df_train['Embarked'].map({'C':0, 'Q':1, 'S':2})
# df_test['EmBarked'] = df_test['EmBarked'].map({'C':0, 'Q':1, 'S':2})

# Sex 카테고리 수치화
df_train['Sex'] = df_train['Sex'].map({'female': 0, 'male':1})
# df_test['Sex'] = df_test['Sex'].map({'female': 0, 'male':1})

## Person Correlation of Features
# heatmap_data = df_train[['Survived', 'Pclass', 'Sex', 'Fare', 'Embarked', 'FamilySize','Initial', 'Age_cat']]
# colormap = plt.cm.RdBu
# sns.heatmap(heatmap_data.astype(float).corr(),
#             linewidths = .1, vmax = 1.0,
#             square = True, cmap = colormap, linecolor = 'white', annot=True, annot_kws = {"size":16})
# del heatmap_data
# plt.show()

