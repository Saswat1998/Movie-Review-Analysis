{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('TextFiles/moviereviews.tsv', sep='\\t')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
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
       "      <th>label</th>\n",
       "      <th>review</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>neg</td>\n",
       "      <td>how do films like mouse hunt get into theatres...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>neg</td>\n",
       "      <td>some talented actresses are blessed with a dem...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>pos</td>\n",
       "      <td>this has been an extraordinary year for austra...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>pos</td>\n",
       "      <td>according to hollywood movies made in last few...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>neg</td>\n",
       "      <td>my first press screening of 1998 and already i...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  label                                             review\n",
       "0   neg  how do films like mouse hunt get into theatres...\n",
       "1   neg  some talented actresses are blessed with a dem...\n",
       "2   pos  this has been an extraordinary year for austra...\n",
       "3   pos  according to hollywood movies made in last few...\n",
       "4   neg  my first press screening of 1998 and already i..."
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "label      0\n",
       "review    35\n",
       "dtype: int64"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.dropna(inplace=True)"
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
       "label     0\n",
       "review    0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "blanks = []"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "for i,lb,rv in  df.itertuples():\n",
    "    if rv.isspace():\n",
    "        blanks.append(i)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[57,\n",
       " 71,\n",
       " 147,\n",
       " 151,\n",
       " 283,\n",
       " 307,\n",
       " 313,\n",
       " 323,\n",
       " 343,\n",
       " 351,\n",
       " 427,\n",
       " 501,\n",
       " 633,\n",
       " 675,\n",
       " 815,\n",
       " 851,\n",
       " 977,\n",
       " 1079,\n",
       " 1299,\n",
       " 1455,\n",
       " 1493,\n",
       " 1525,\n",
       " 1531,\n",
       " 1763,\n",
       " 1851,\n",
       " 1905,\n",
       " 1993]"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "blanks"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\saswat\\Anaconda3\\lib\\site-packages\\ipykernel_launcher.py:1: FutureWarning: supplying multiple axes to axis is deprecated and will be removed in a future version.\n",
      "  \"\"\"Entry point for launching an IPython kernel.\n"
     ]
    },
    {
     "ename": "ValueError",
     "evalue": "No axis named 57 for object type <class 'type'>",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-10-c781c7073217>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mdf\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdropna\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mblanks\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0minplace\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mTrue\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\pandas\\core\\frame.py\u001b[0m in \u001b[0;36mdropna\u001b[1;34m(self, axis, how, thresh, subset, inplace)\u001b[0m\n\u001b[0;32m   4568\u001b[0m             \u001b[1;32mfor\u001b[0m \u001b[0max\u001b[0m \u001b[1;32min\u001b[0m \u001b[0maxis\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   4569\u001b[0m                 result = result.dropna(how=how, thresh=thresh, subset=subset,\n\u001b[1;32m-> 4570\u001b[1;33m                                        axis=ax)\n\u001b[0m\u001b[0;32m   4571\u001b[0m         \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   4572\u001b[0m             \u001b[0maxis\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_get_axis_number\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0maxis\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\pandas\\core\\frame.py\u001b[0m in \u001b[0;36mdropna\u001b[1;34m(self, axis, how, thresh, subset, inplace)\u001b[0m\n\u001b[0;32m   4570\u001b[0m                                        axis=ax)\n\u001b[0;32m   4571\u001b[0m         \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 4572\u001b[1;33m             \u001b[0maxis\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_get_axis_number\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0maxis\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   4573\u001b[0m             \u001b[0magg_axis\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;36m1\u001b[0m \u001b[1;33m-\u001b[0m \u001b[0maxis\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   4574\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\pandas\\core\\generic.py\u001b[0m in \u001b[0;36m_get_axis_number\u001b[1;34m(cls, axis)\u001b[0m\n\u001b[0;32m    359\u001b[0m                 \u001b[1;32mpass\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    360\u001b[0m         raise ValueError('No axis named {0} for object type {1}'\n\u001b[1;32m--> 361\u001b[1;33m                          .format(axis, type(cls)))\n\u001b[0m\u001b[0;32m    362\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    363\u001b[0m     \u001b[1;33m@\u001b[0m\u001b[0mclassmethod\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mValueError\u001b[0m: No axis named 57 for object type <class 'type'>"
     ]
    }
   ],
   "source": [
    "df.dropna(blanks, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.drop(blanks,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1938"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df['review']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "y = df['label']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.pipeline import Pipeline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.feature_extraction.text import TfidfVectorizer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.svm import LinearSVC"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "text_clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',LinearSVC())])"
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
       "Pipeline(memory=None,\n",
       "     steps=[('tfidf', TfidfVectorizer(analyzer='word', binary=False, decode_error='strict',\n",
       "        dtype=<class 'numpy.float64'>, encoding='utf-8', input='content',\n",
       "        lowercase=True, max_df=1.0, max_features=None, min_df=1,\n",
       "        ngram_range=(1, 1), norm='l2', preprocessor=None, smooth_idf=True,...ax_iter=1000,\n",
       "     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,\n",
       "     verbose=0))])"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "text_clf.fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions = text_clf.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import confusion_matrix, classification_report, accuracy_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[235  47]\n",
      " [ 41 259]]\n"
     ]
    }
   ],
   "source": [
    "print(confusion_matrix(y_test,predictions))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "         neg       0.85      0.83      0.84       282\n",
      "         pos       0.85      0.86      0.85       300\n",
      "\n",
      "   micro avg       0.85      0.85      0.85       582\n",
      "   macro avg       0.85      0.85      0.85       582\n",
      "weighted avg       0.85      0.85      0.85       582\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(y_test,predictions))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.8487972508591065\n"
     ]
    }
   ],
   "source": [
    "print(accuracy_score(y_test,predictions))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['neg']\n"
     ]
    }
   ],
   "source": [
    "print(text_clf.predict([\"That movie extremely gross and not suitable for younger audience\"]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['pos']\n"
     ]
    }
   ],
   "source": [
    "print(text_clf.predict([\"that was one hell of a rollercoaster! Awesome directions and brilliant VFX.\"]))"
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
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
