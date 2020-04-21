{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "# -*- coding: utf-8 -*-\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn import preprocessing\n",
    "%matplotlib inline\n",
    "import seaborn as sns\n",
    "\n",
    "# machine learning\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.svm import SVC, LinearSVC\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.linear_model import Perceptron\n",
    "from sklearn.linear_model import SGDClassifier\n",
    "from sklearn.tree import DecisionTreeClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  \\\n",
       "0            1         0       3   \n",
       "1            2         1       1   \n",
       "2            3         1       3   \n",
       "3            4         1       1   \n",
       "4            5         0       3   \n",
       "\n",
       "                                                Name     Sex   Age  SibSp  \\\n",
       "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
       "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
       "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       "4                           Allen, Mr. William Henry    male  35.0      0   \n",
       "\n",
       "   Parch            Ticket     Fare Cabin Embarked  \n",
       "0      0         A/5 21171   7.2500   NaN        S  \n",
       "1      0          PC 17599  71.2833   C85        C  \n",
       "2      0  STON/O2. 3101282   7.9250   NaN        S  \n",
       "3      0            113803  53.1000  C123        S  \n",
       "4      0            373450   8.0500   NaN        S  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Train = pd.read_csv('train.csv')\n",
    "Test = pd.read_csv('test.csv')\n",
    "AllData=[Train,Test]\n",
    "Train.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Data Overview"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 891 entries, 0 to 890\n",
      "Data columns (total 12 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   PassengerId  891 non-null    int64  \n",
      " 1   Survived     891 non-null    int64  \n",
      " 2   Pclass       891 non-null    int64  \n",
      " 3   Name         891 non-null    object \n",
      " 4   Sex          891 non-null    object \n",
      " 5   Age          714 non-null    float64\n",
      " 6   SibSp        891 non-null    int64  \n",
      " 7   Parch        891 non-null    int64  \n",
      " 8   Ticket       891 non-null    object \n",
      " 9   Fare         891 non-null    float64\n",
      " 10  Cabin        204 non-null    object \n",
      " 11  Embarked     889 non-null    object \n",
      "dtypes: float64(2), int64(5), object(5)\n",
      "memory usage: 83.7+ KB\n"
     ]
    }
   ],
   "source": [
    "Train.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Ticket has  holds no meaningful information so we should drop it.\n",
    "Cabin has a lot of Nan's which may be meaningfull (not everyone in the ship had a cabin which may contribute with survival. Age and sex may also be meaningfull (women and children first). The Age NaN's can be extrapolated (maybe)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "for dataset in AllData:\n",
    "    dataset.drop('Ticket', axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Correlation between features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x29232a9a588>"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x648 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "TrainDataC = Train.corr()\n",
    "#correlation heatmap, masking the upper part as its redundant\n",
    "mask = np.zeros_like(TrainDataC , dtype=np.bool)\n",
    "mask[np.triu_indices_from(mask)] = True\n",
    "plt.subplots(figsize=(12,9))\n",
    "sns.heatmap(TrainDataC, annot=True,square=True, mask=mask)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Pclass and fare correlate with survival. Age not so much surprisingly. Pclass correlates with fare, not surprinsingly. The high correlation between Parch and SibSp may be due to the families travelling together."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Age\n",
    "  \n",
    "### Directly looking into age (as an average per survival)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x29232e91588>"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.barplot('Survived', 'Age', data=Train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This is interesting as age (as an average) doesn't look very different.  Lets separate the age into groups:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "Age_Data = Train.loc[:,['PassengerId','Age', 'Survived']]\n",
    "Age_Data['Age'].fillna(-0.1, inplace=True)\n",
    "bins= [-100, 0, 10, 20, 30, 40, 50, 60, np.inf]\n",
    "labels = [0,1,2,3,4,5,6,7]\n",
    "#labels = ['Unknown','0-10','10-20','20-30','30-40','40-50','50-60','>60']\n",
    "Age_Data['AgeGroup'] = pd.cut(Age_Data['Age'], bins=bins, labels=labels, right=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x29232e4c6c8>"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.barplot('Survived', 'AgeGroup', data=Age_Data)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "If we replace Age by the median"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "Age_Data['Age'].replace(-0.1, np.nan, inplace=True)\n",
    "Age_Data['Age'].fillna((Age_Data['Age'].median()), inplace=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "bins= [0, 10, 20, 30, 40, 50, 60, np.inf]\n",
    "labels = [1,2,3,4,5,6,7]\n",
    "#labels = ['Unknown','0-10','10-20','20-30','30-40','40-50','50-60','>60']\n",
    "Age_Data['AgeGroup'] = pd.cut(Age_Data['Age'], bins=bins, labels=labels, right=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x29232d27a08>"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.barplot('Survived', 'AgeGroup', data=Age_Data)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This makes quite a lot of difference but younger seem to be less prone to die still. Lets apply this to our data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "for dataset in AllData:\n",
    "    dataset['AgeGroup'] = pd.cut(Age_Data['Age'], bins=bins, labels=labels, right=False)\n",
    "    dataset['Age'].fillna((Age_Data['Age'].median()), inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Gender"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x29232db7508>"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.barplot(x='Survived', y='Sex', data=Train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Women first indeed. Lets encode this categorical data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "for dataset in AllData:\n",
    "    dataset['Sex'].replace( ['male','female'], [0,1],inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  AgeGroup  Survived\n",
      "0        1  0.612903\n",
      "1        2  0.401961\n",
      "2        3  0.324937\n",
      "3        4  0.437126\n",
      "4        5  0.382022\n",
      "5        6  0.416667\n",
      "6        7  0.269231\n"
     ]
    }
   ],
   "source": [
    "print(Train[['AgeGroup', 'Survived']].groupby(['AgeGroup'], as_index = False).mean())\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# SibSp and Parch\n",
    "\n",
    "This can be concatenated in a new feature called family size (otherwise it may get confusing)  \n",
    "Family size = SibSp + Parch + 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "for dataset in AllData:\n",
    "    dataset['FamilySize']  = dataset['SibSp'] + dataset['Parch'] + 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x2923303b188>"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.barplot(x=Train.FamilySize, y=Train.Survived)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "4 is the magic number."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Fare"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x292330edf48>"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1080x360 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fx, axes = plt.subplots(1, 2, figsize=(15,5))\n",
    "sns.distplot(Train.Fare, ax=axes[0])\n",
    "sns.boxplot('Survived', 'Fare', data=Train, showfliers = False, ax=axes[1]) "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Fare is indeed correlated with survival. The distribution is very shifted to the left (more people payed less) and has a longtail distribution. Lets look more into this by dividing the data."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can categorize the data by dividing it by a number and returning an int. Also there is an NA in the test set which we should substitute by the median."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "Test['Fare'].replace(np.nan, Test['Fare'].mean(axis=0), inplace=True)\n",
    "for dataset in AllData:\n",
    "    dataset.Fare = (dataset.Fare /20).astype(np.int64) + 1 "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x2923321ce48>"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.distplot(Train.Fare)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x29233328f08>"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.distplot(Test.Fare)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This looks neater. Again, higher fare, higher survival changes"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Cabin"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Keeping with a simplistic approach, having a cabin number may be by itself correlated with higher social status and consequently higher survivability."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "for dataset in AllData:\n",
    "    dataset[\"HasCabin\"] = dataset[\"Cabin\"].notnull().astype('int')\n",
    "    \n",
    "sns.barplot(x=\"HasCabin\", y=\"Survived\", data=Train)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Success, having cabin tends to lead to survival. We have the data to reflect this then by creating the 'HasCabin' feature"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>AgeGroup</th>\n",
       "      <th>FamilySize</th>\n",
       "      <th>HasCabin</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>0</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>1</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>1</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>1</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>0</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  \\\n",
       "0            1         0       3   \n",
       "1            2         1       1   \n",
       "2            3         1       3   \n",
       "3            4         1       1   \n",
       "4            5         0       3   \n",
       "\n",
       "                                                Name  Sex   Age  SibSp  Parch  \\\n",
       "0                            Braund, Mr. Owen Harris    0  22.0      1      0   \n",
       "1  Cumings, Mrs. John Bradley (Florence Briggs Th...    1  38.0      1      0   \n",
       "2                             Heikkinen, Miss. Laina    1  26.0      0      0   \n",
       "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)    1  35.0      1      0   \n",
       "4                           Allen, Mr. William Henry    0  35.0      0      0   \n",
       "\n",
       "   Fare Cabin Embarked AgeGroup  FamilySize  HasCabin  \n",
       "0     1   NaN        S        3           2         0  \n",
       "1     4   C85        C        4           2         1  \n",
       "2     1   NaN        S        3           1         0  \n",
       "3     3  C123        S        4           2         1  \n",
       "4     1   NaN        S        4           1         0  "
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Train.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Port of embark"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count     889\n",
       "unique      3\n",
       "top         S\n",
       "freq      644\n",
       "Name: Embarked, dtype: object"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Train.Embarked.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2 0\n"
     ]
    }
   ],
   "source": [
    "print(Train.Embarked.isnull().sum(), Test.Embarked.isnull().sum())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Substituting nan's with the most representative - S"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "Train['Embarked'] = Train['Embarked'].fillna('S')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x292335e61c8>"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x216 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fx, axes = plt.subplots(1, 2, figsize=(12,3))\n",
    "\n",
    "sns.countplot(x=Train.Embarked, ax=axes[0])\n",
    "sns.barplot(x=Train.Embarked, y=Train.Survived, ax=axes[1])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Turning this into numerical.  \n",
    "1 -> S  \n",
    "2 -> C  \n",
    "3 -> Q  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "for dataset in AllData:\n",
    "    dataset['Embarked'] = dataset['Embarked'].map({'S': 1, 'C': 2, 'Q': 3}).astype(int)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Name"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can extrapolate a new feature - title - from the names."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "for dataset in AllData:\n",
    "    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+\\\\w\\\\.)+', expand=False)\n",
    "    dataset['Title'] = dataset.Title.str.replace('.','')\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x292336df448>"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "pd.crosstab(Train['Title'], Train['Survived']).plot(kind='bar')\n",
    "#print(Train[['Title', 'Survived']].groupby(['Title'], as_index = False).mean())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Losts of titles which may induce overfitting. Lets agregate the data:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "maping={'Master':1, 'Miss':2,'Mr':3,'Mrs':4, 'Other':5,'Royals':6}\n",
    "for dataset in AllData:\n",
    "    dataset['Title'] = dataset.Title.replace(['Lady','Countess','Sir'], 'Royals')\n",
    "    dataset['Title'] = dataset.Title.replace(['Capt','Col','Major','Dr', 'Jonkheer','Don','Rev'], 'Other')\n",
    "    dataset['Title'] = dataset.Title.replace('Mme','Mrs')\n",
    "    dataset['Title'] = dataset.Title.replace(['Mlle','Ms'],'Miss')\n",
    "    dataset['Title'] = dataset.Title.map(maping)\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x29233687548>"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "pd.crosstab(Train['Title'], Train['Survived']).plot(kind='bar')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We also mapped the categorical features into numeric\n",
    "  \n",
    "1 - Master  \n",
    "2 - Miss  \n",
    "3 - Mr  \n",
    "4 - Mrs  \n",
    "5 - Other  \n",
    "6 - Royals  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "Test['Title']=Test['Title'].fillna(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Data Preparation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>AgeGroup</th>\n",
       "      <th>FamilySize</th>\n",
       "      <th>HasCabin</th>\n",
       "      <th>Title</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>0</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>1</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>C85</td>\n",
       "      <td>2</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>1</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>1</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>C123</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>0</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  \\\n",
       "0            1         0       3   \n",
       "1            2         1       1   \n",
       "2            3         1       3   \n",
       "3            4         1       1   \n",
       "4            5         0       3   \n",
       "\n",
       "                                                Name  Sex   Age  SibSp  Parch  \\\n",
       "0                            Braund, Mr. Owen Harris    0  22.0      1      0   \n",
       "1  Cumings, Mrs. John Bradley (Florence Briggs Th...    1  38.0      1      0   \n",
       "2                             Heikkinen, Miss. Laina    1  26.0      0      0   \n",
       "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)    1  35.0      1      0   \n",
       "4                           Allen, Mr. William Henry    0  35.0      0      0   \n",
       "\n",
       "   Fare Cabin  Embarked AgeGroup  FamilySize  HasCabin  Title  \n",
       "0     1   NaN         1        3           2         0      3  \n",
       "1     4   C85         2        4           2         1      4  \n",
       "2     1   NaN         1        3           1         0      2  \n",
       "3     3  C123         1        4           2         1      4  \n",
       "4     1   NaN         1        4           1         0      3  "
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>AgeGroup</th>\n",
       "      <th>FamilySize</th>\n",
       "      <th>HasCabin</th>\n",
       "      <th>Title</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>892</td>\n",
       "      <td>3</td>\n",
       "      <td>Kelly, Mr. James</td>\n",
       "      <td>0</td>\n",
       "      <td>34.5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>NaN</td>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>893</td>\n",
       "      <td>3</td>\n",
       "      <td>Wilkes, Mrs. James (Ellen Needs)</td>\n",
       "      <td>1</td>\n",
       "      <td>47.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>4.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>894</td>\n",
       "      <td>2</td>\n",
       "      <td>Myles, Mr. Thomas Francis</td>\n",
       "      <td>0</td>\n",
       "      <td>62.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>NaN</td>\n",
       "      <td>3</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>895</td>\n",
       "      <td>3</td>\n",
       "      <td>Wirz, Mr. Albert</td>\n",
       "      <td>0</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>896</td>\n",
       "      <td>3</td>\n",
       "      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>\n",
       "      <td>1</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>4.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Pclass                                          Name  Sex  \\\n",
       "0          892       3                              Kelly, Mr. James    0   \n",
       "1          893       3              Wilkes, Mrs. James (Ellen Needs)    1   \n",
       "2          894       2                     Myles, Mr. Thomas Francis    0   \n",
       "3          895       3                              Wirz, Mr. Albert    0   \n",
       "4          896       3  Hirvonen, Mrs. Alexander (Helga E Lindqvist)    1   \n",
       "\n",
       "    Age  SibSp  Parch  Fare Cabin  Embarked AgeGroup  FamilySize  HasCabin  \\\n",
       "0  34.5      0      0     1   NaN         3        3           1         0   \n",
       "1  47.0      1      0     1   NaN         1        4           2         0   \n",
       "2  62.0      0      0     1   NaN         3        3           1         0   \n",
       "3  27.0      0      0     1   NaN         1        4           1         0   \n",
       "4  22.0      1      1     1   NaN         1        4           3         0   \n",
       "\n",
       "   Title  \n",
       "0    3.0  \n",
       "1    4.0  \n",
       "2    3.0  \n",
       "3    3.0  \n",
       "4    4.0  "
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Test.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train = Train[['Pclass', 'Sex', 'Fare', 'Embarked', 'AgeGroup', 'FamilySize', 'HasCabin', 'Title']]\n",
    "X_test = Test[['Pclass', 'Sex', 'Fare', 'Embarked', 'AgeGroup', 'FamilySize', 'HasCabin', 'Title']]\n",
    "Y_train = Train['Survived']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(891, 8) (418, 8) (891,)\n"
     ]
    }
   ],
   "source": [
    "print(X_train.shape, X_test.shape, Y_train.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Model comparisons"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8002244668911336"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Logistic reg\n",
    "classifier = LogisticRegression()\n",
    "classifier.fit(X_train, Y_train)\n",
    "Y_pred = classifier.predict(X_test)\n",
    "classifier.score(X_train, Y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Checking feature coefficient:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Feature</th>\n",
       "      <th>Corr</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Fare</td>\n",
       "      <td>2.640000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>Title</td>\n",
       "      <td>0.621919</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>AgeGroup</td>\n",
       "      <td>0.190970</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Embarked</td>\n",
       "      <td>0.025116</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>HasCabin</td>\n",
       "      <td>-0.203844</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>FamilySize</td>\n",
       "      <td>-0.292588</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Sex</td>\n",
       "      <td>-0.877705</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      Feature      Corr\n",
       "1        Fare  2.640000\n",
       "6       Title  0.621919\n",
       "3    AgeGroup  0.190970\n",
       "2    Embarked  0.025116\n",
       "5    HasCabin -0.203844\n",
       "4  FamilySize -0.292588\n",
       "0         Sex -0.877705"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "coef = pd.DataFrame(X_train.columns.delete(0))\n",
    "coef.columns = ['Feature']\n",
    "coef['Corr'] = pd.Series(classifier.coef_[0])\n",
    "coef.sort_values(by='Corr', ascending=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "    Fare is very important for some reason. We may have overfitting"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8338945005611672"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#SVC\n",
    "\n",
    "svc=SVC()\n",
    "svc.fit(X_train,Y_train)\n",
    "Y_pred = svc.predict(X_test)\n",
    "svc.score(X_train, Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8518518518518519"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#KNN\n",
    "\n",
    "knn = KNeighborsClassifier(n_neighbors=5)\n",
    "knn.fit(X_train,Y_train)\n",
    "Y_pred = knn.predict(X_test)\n",
    "knn.score(X_train, Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7890011223344556"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Naive Bayes\n",
    "\n",
    "NB = GaussianNB()\n",
    "NB.fit(X_train, Y_train)\n",
    "Y_pred = NB.predict(X_test)\n",
    "NB.score(X_train, Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7003367003367004"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#gradient descent\n",
    "\n",
    "sgd = SGDClassifier()\n",
    "sgd.fit(X_train,Y_train)\n",
    "Y_pred = NB.predict(X_test)\n",
    "sgd.score(X_train, Y_train)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9068462401795735"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Decision Tree\n",
    "\n",
    "DT = DecisionTreeClassifier()\n",
    "DT.fit(X_train,Y_train)\n",
    "Y_pred = DT.predict(X_test)\n",
    "DT.score(X_train, Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9068462401795735"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Rand Forest\n",
    "\n",
    "RF = RandomForestClassifier()\n",
    "RF.fit(X_train, Y_train)\n",
    "Y_pred = RF.predict(X_test)\n",
    "RF.score(X_train, Y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "   While DT and RF scored the same, we will select RF because they overfit less to the training set (which is quite small)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Hyperparameter Tuning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    "#param_grid = { \"criterion\" : [\"gini\", \"entropy\"], \"min_samples_leaf\" : [1, 5, 10, 25, 50, 70], \"min_samples_split\" : [2, 4, 10, 12, 16, 18, 25, 35], \"n_estimators\": [100, 400, 700, 1000, 1500]}\n",
    "#from sklearn.model_selection import GridSearchCV, cross_val_score\n",
    "\n",
    "#rf = RandomForestClassifier(n_estimators=100, max_features='auto', oob_score=True, random_state=1, n_jobs=-1)\n",
    "\n",
    "#clf = GridSearchCV(estimator=rf, param_grid=param_grid, n_jobs=-1)\n",
    "\n",
    "#clf.fit(X_train, Y_train)\n",
    "\n",
    "#clf.best_params_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "oob score: 83.39 %\n"
     ]
    }
   ],
   "source": [
    "#post tuning\n",
    "random_forest = RandomForestClassifier(criterion = \"entropy\", \n",
    "                                       min_samples_leaf = 1, \n",
    "                                       min_samples_split = 25,   \n",
    "                                       n_estimators=700, \n",
    "                                       max_features='auto', \n",
    "                                       oob_score=True, \n",
    "                                       random_state=1, \n",
    "                                       n_jobs=-1)\n",
    "\n",
    "random_forest.fit(X_train, Y_train)\n",
    "Y_prediction = random_forest.predict(X_test)\n",
    "\n",
    "score=random_forest.score(X_train, Y_train)\n",
    "print(\"oob score:\", round(random_forest.oob_score_, 4)*100, \"%\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[505,  44],\n",
       "       [105, 237]], dtype=int64)"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.model_selection import cross_val_predict\n",
    "from sklearn.metrics import confusion_matrix\n",
    "predictions = cross_val_predict(random_forest, X_train, Y_train, cv=20)\n",
    "confusion_matrix(Y_train, predictions)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot learning curve\n",
    "def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,\n",
    "                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):\n",
    "    plt.figure()\n",
    "    plt.title(title)\n",
    "    if ylim is not None:\n",
    "        plt.ylim(*ylim)\n",
    "    plt.xlabel(\"Training examples\")\n",
    "    plt.ylabel(\"Score\")\n",
    "    train_sizes, train_scores, test_scores = learning_curve(\n",
    "        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)\n",
    "    train_scores_mean = np.mean(train_scores, axis=1)\n",
    "    train_scores_std = np.std(train_scores, axis=1)\n",
    "    test_scores_mean = np.mean(test_scores, axis=1)\n",
    "    test_scores_std = np.std(test_scores, axis=1)\n",
    "    plt.grid()\n",
    "\n",
    "    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,\n",
    "                     train_scores_mean + train_scores_std, alpha=0.1,\n",
    "                     color=\"r\")\n",
    "    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,\n",
    "                     test_scores_mean + test_scores_std, alpha=0.1, color=\"g\")\n",
    "    plt.plot(train_sizes, train_scores_mean, 'o-', color=\"r\",\n",
    "             label=\"Training score\")\n",
    "    plt.plot(train_sizes, test_scores_mean, 'o-', color=\"g\",\n",
    "             label=\"Validation score\")\n",
    "\n",
    "    plt.legend(loc=\"best\")\n",
    "    return plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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xx13sH37dmbMmMGMGTMAGD58OHv37gXgF7/4RXqq7tRU4Rs3buTkk0/m6quvZvTo0cycObPZfVL8+c9/ZsyYMYwfP54zzjgDcJz8t7/9bcaOHcu4cePS038/9dRTTJw4kbFjx/L5z38+/S6GDx/O/PnzmTZtGg8//DDvvfce55xzDpMnT+YjH/kI77zzTqffewobB2EYRk58/fGv8/rO1qf7/ufWfxJJNJ+5tT5Wzxf+9gXuebVpuu9EIoHfne57wsAJ3HpO65MA9uvXj6lTp/L4449zwQUXsHjxYi677DJEhHA4zMMPP0yvXr3Yu3cvH/rQhzj//PNbTIqX4o477qCkpIQ333yTN998s9l03QsWLKBv374kEgnOOuss3nzzTa699lp++9vfsmLFCvr3798srVdffZX77ruPl156CVXl1FNP5cwzz6SiooJ169axaNEi7rnnHv7jP/6Dhx56iE9/+tPNrp8/fz7Lli1jyJAh6Sqru+++m/fff5/XXnuNQCDA/v37aWxs5KqrruKpp55i1KhRfPazn+WOO+7g61//OuCsMf38889TU1PDhRdeyJ133snIkSN56aWX+PKXv8zTTz/d6rvNBStBGIaRF7LFob3wXMmsZsqsXlJVvv/97zNu3Dg+9rGPsW3bNnbt2tVqOs8++2zaUY8bN45x48alzz3wwANMmjSJiRMnsnr1as+J+DJ5/vnnueiiiygtLaWsrIzZs2fz3HPPATBixAgmTJgAtD6l+Ic//GGuuuoq7rnnnnT10JNPPsk111yTHp3dt29f1q5dy4gRIxg1ahQAV155Jc8++2w6ncsuuwxwRnm/8MILXHrppUyYMIEvfelL7Nixo81nyAUrQRiGkRNt5fQBht86nE3VLaf7Pq73cay8amX6uKaD031feOGFfPOb3+Rf//oXDQ0N6Zz/woUL2bNnD6+++irBYJDhw4d7TvGdiVfp4v333+fnP/85r7zyChUVFVx11VXtpqNtTNWdmiocnOnCvaqY7rzzTl566SUeffRRJkyYwOuvv46qdmhKcGiaFjyZTNKnTx/PFe4OBStBGIaRFxactYCSYPPpvkuCJSw469Cm+y4rK2P69Ol8/vOfb9Y4XV1dzYABAwgGg6xYsYJNXmtRZHDGGWew0F3+9N///jdvvvkm4EwVXlpaSu/evdm1axePPfZY+pry8nJq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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "from sklearn.model_selection import cross_val_score\n",
    "scores = cross_val_score(random_forest, X_train, Y_train, cv=10, scoring = \"accuracy\")\n",
    "# Plot learning curves\n",
    "from sklearn.model_selection import learning_curve\n",
    "from sklearn.model_selection import validation_curve\n",
    "\n",
    "title = \"Learning Curves (Random Forest)\"\n",
    "cv = 10\n",
    "plot_learning_curve(random_forest, title, X_train, Y_train, ylim=(0.7, 1.01), cv=cv, n_jobs=1);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [],
   "source": [
    "submission = pd.DataFrame({\n",
    "        \"PassengerId\": Test['PassengerId'],\n",
    "        \"Survived\": Y_prediction\n",
    "    })\n",
    "submission.to_csv('submission_NEW.csv', index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
