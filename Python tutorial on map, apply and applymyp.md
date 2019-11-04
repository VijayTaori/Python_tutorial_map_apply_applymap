

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
# Let's work with Kaggel's Titanic_train dataset 
# This is a dataset for training a machine learning algoritham
titanic_train = pd.read_csv('titanic_train.csv')
```


```python
# Check out the head
titanic_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Sex_0_1</th>
      <th>Sex_01_applySeriesMethod</th>
      <th>Sex_01_applySeriesMethod_addPrint</th>
      <th>Sex_applyDF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### map


```python
# map is a series method
# Let's say instead of 'Sex' column with catogaries 'male' and 'female' you would like 0 and 1 
# There are lot of different ways to achieve that and 'map' is one of them
titanic_train['Sex_0_1'] = titanic_train['Sex'].map({'female':1, 'male':0})
titanic_train[['Sex','Sex_0_1']].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sex</th>
      <th>Sex_0_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>male</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>female</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>female</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>female</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>male</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### apply (series)


```python
# Imagine you would like to pass a function which performs certain logic over the series or just one columnof dataframe
# first define the funcion
def Sex_01_applySeriesMethod(oranges):
    if oranges == 'male':
        return 0
    elif oranges == 'female':
        return 1


# Now let's put this function to make a new column(or series)
titanic_train['Sex_01_applySeriesMethod'] = titanic_train['Sex'].apply(Sex_01_applySeriesMethod)
# Let's check out the head
titanic_train[['Sex','Sex_0_1','Sex_01_applySeriesMethod']].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sex</th>
      <th>Sex_0_1</th>
      <th>Sex_01_applySeriesMethod</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>male</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>female</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>female</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>female</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>male</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# How does this work?
# The function name is long but it reflects what to look for in the output
def Sex_01_applySeriesMethod_addPrint(oranges):
    print('stage1')
    if oranges == 'male':
        print('Fizz')
        return 0
    print('stage2')
    if oranges == 'female':
        print('Buzz')
        return 1
        
    


titanic_train['Sex_01_applySeriesMethod_addPrint'] = titanic_train['Sex'].apply(Sex_01_applySeriesMethod_addPrint)
# Let's check out the head
titanic_train[['Sex','Sex_0_1','Sex_01_applySeriesMethod', 'Sex_01_applySeriesMethod_addPrint']].head()
# So this cell explains how it works, you do have move around the print statements to figure it out
```

    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    stage2
    Buzz
    stage1
    stage2
    Buzz
    stage1
    Fizz
    stage1
    Fizz
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sex</th>
      <th>Sex_0_1</th>
      <th>Sex_01_applySeriesMethod</th>
      <th>Sex_01_applySeriesMethod_addPrint</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>female</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>female</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>female</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### apply (DataFrame)


```python
# Let's say we would like to change two columns of this dataframe
# 1. Changing 'Sex' to 0 and 1
# 2. Changing 'Embarked' to 10, 20 and 30
# 
def Sex_01_applyDFMethod(oranges):
    '''Since we will be working with DF, specifically two columns
    we will have to decide before hand which one is the firt column 
    and this will be the 0th or 1th element of 'oranges' '''
    if oranges[0] == 'male':
        if oranges[1]=='C':
            return (0,10)#oranges[0] = 0 and oranges[1] = 10)
        elif oranges[1]=='Q':
            return (0,20)#oranges[0] = 0 and oranges[1] = 20)
        elif oranges[1]=='S':
            return (0,30)#oranges[0] = 0 and oranges[1] = 30)
        else:
            return (0,'unknown embarked')#oranges[0] = 0 and oranges[1] = 'unknown embarked')
    elif oranges[0] == 'female':
        if oranges[1]=='C':
            return (1,10)#oranges[0] = 1 and oranges[1] = 10)
        elif oranges[1]=='Q':
            return (1,20)#oranges[0] = 1 and oranges[1] = 20)
        elif oranges[1]=='S':
            return (1,30)#oranges[0] = 1 and oranges[1] = 30)
        else:
            return (1,'unknown embarked')# oranges[0] = 1 and oranges[1] = 'unknown embarked')
    else:
        if oranges[1]=='C':
            return ('unknown sex',10)#oranges[0] = 'unknown sex' and oranges[1] = 10)
        elif oranges[1]=='Q':
            return ('unknown sex',20)#oranges[0] = 'unknown sex' and oranges[1] = 20)
        elif oranges[1]=='S':
            return ('unknown sex',30)#oranges[0] = 'unknown sex' and oranges[1] = 30)
        else:
            return ('unknown sex','unknown embarked')#oranges[0] = 'unknown sex' and oranges[1] = 'unknown embarked')
        
        
df_sex_embarked = titanic_train[['Sex','Embarked']].apply(Sex_01_applyDFMethod, axis=1)
print(type(df_sex_embarked))
print(type(df_sex_embarked[0]))
print((df_sex_embarked))

titanic_train['Sex_applyDF'] = df_sex_embarked.apply(lambda x:x[0])
titanic_train['Embarked_applyDF'] = df_sex_embarked.apply(lambda x:x[1])


titanic_train[['Sex_applyDF','Embarked_applyDF']].head()
```

    <class 'pandas.core.series.Series'>
    <class 'tuple'>
    0      (0, 30)
    1      (1, 10)
    2      (1, 30)
    3      (1, 30)
    4      (0, 30)
    5      (0, 20)
    6      (0, 30)
    7      (0, 30)
    8      (1, 30)
    9      (1, 10)
    10     (1, 30)
    11     (1, 30)
    12     (0, 30)
    13     (0, 30)
    14     (1, 30)
    15     (1, 30)
    16     (0, 20)
    17     (0, 30)
    18     (1, 30)
    19     (1, 10)
    20     (0, 30)
    21     (0, 30)
    22     (1, 20)
    23     (0, 30)
    24     (1, 30)
    25     (1, 30)
    26     (0, 10)
    27     (0, 30)
    28     (1, 20)
    29     (0, 30)
            ...   
    861    (0, 30)
    862    (1, 30)
    863    (1, 30)
    864    (0, 30)
    865    (1, 30)
    866    (1, 10)
    867    (0, 30)
    868    (0, 30)
    869    (0, 30)
    870    (0, 30)
    871    (1, 30)
    872    (0, 30)
    873    (0, 30)
    874    (1, 10)
    875    (1, 10)
    876    (0, 30)
    877    (0, 30)
    878    (0, 30)
    879    (1, 10)
    880    (1, 30)
    881    (0, 30)
    882    (1, 30)
    883    (0, 30)
    884    (0, 30)
    885    (1, 20)
    886    (0, 30)
    887    (1, 30)
    888    (1, 30)
    889    (0, 10)
    890    (0, 20)
    Length: 891, dtype: object
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sex_applyDF</th>
      <th>Embarked_applyDF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>30</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
</div>



### apply DF another example


```python
# Find if there are any null values in Age column
print(titanic_train['Age'].isnull().sum()) # Adding up all the True(s)(1) and False(s)(0) from the below command
titanic_train['Age'].isnull()
# So it turns out it 'Age' has 177 missing values
# Can we fill these up by using average value from each Pclass?
```

    177
    




    0      False
    1      False
    2      False
    3      False
    4      False
    5       True
    6      False
    7      False
    8      False
    9      False
    10     False
    11     False
    12     False
    13     False
    14     False
    15     False
    16     False
    17      True
    18     False
    19      True
    20     False
    21     False
    22     False
    23     False
    24     False
    25     False
    26      True
    27     False
    28      True
    29      True
           ...  
    861    False
    862    False
    863     True
    864    False
    865    False
    866    False
    867    False
    868     True
    869    False
    870    False
    871    False
    872    False
    873    False
    874    False
    875    False
    876    False
    877    False
    878     True
    879    False
    880    False
    881    False
    882    False
    883    False
    884    False
    885    False
    886    False
    887    False
    888     True
    889    False
    890    False
    Name: Age, Length: 891, dtype: bool




```python
print(type(pd.isnull))
print(type(np.isnan))
```

    <class 'function'>
    <class 'numpy.ufunc'>
    


```python
# Let's define a function
def Ave_age_per_Pclass(oranges):
    if np.isnan(oranges[0]):
        if oranges[1]==1:
            return'1st class missing Age'
        elif oranges[1]==2:
            return '2nd class missing Age'
        elif oranges[1]==3:
            return '3rd class missing Age'
    else:
        return oranges[0]
  
    
    
titanic_train['Age_per_Pclass'] = titanic_train[['Age', 'Pclass']].apply(Ave_age_per_Pclass, axis=1)
# The axis can be zero if you would like to apply the function on axis zero
titanic_train['Age_per_Pclass']  
```




    0                         22
    1                         38
    2                         26
    3                         35
    4                         35
    5      3rd class missing Age
    6                         54
    7                          2
    8                         27
    9                         14
    10                         4
    11                        58
    12                        20
    13                        39
    14                        14
    15                        55
    16                         2
    17     2nd class missing Age
    18                        31
    19     3rd class missing Age
    20                        35
    21                        34
    22                        15
    23                        28
    24                         8
    25                        38
    26     3rd class missing Age
    27                        19
    28     3rd class missing Age
    29     3rd class missing Age
                   ...          
    861                       21
    862                       48
    863    3rd class missing Age
    864                       24
    865                       42
    866                       27
    867                       31
    868    3rd class missing Age
    869                        4
    870                       26
    871                       47
    872                       33
    873                       47
    874                       28
    875                       15
    876                       20
    877                       19
    878    3rd class missing Age
    879                       56
    880                       25
    881                       33
    882                       22
    883                       28
    884                       25
    885                       39
    886                       27
    887                       19
    888    3rd class missing Age
    889                       26
    890                       32
    Name: Age_per_Pclass, Length: 891, dtype: object



# applymap


```python
# Lets check the info method
titanic_train.info()
# there are some columns in int and float type
# Let's say we want to convert these all the int types and float types in float types
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 18 columns):
    PassengerId                          891 non-null int64
    Survived                             891 non-null int64
    Pclass                               891 non-null int64
    Name                                 891 non-null object
    Sex                                  891 non-null object
    Age                                  714 non-null float64
    SibSp                                891 non-null int64
    Parch                                891 non-null int64
    Ticket                               891 non-null object
    Fare                                 891 non-null float64
    Cabin                                204 non-null object
    Embarked                             889 non-null object
    Sex_0_1                              891 non-null int64
    Sex_01_applySeriesMethod             891 non-null int64
    Sex_01_applySeriesMethod_addPrint    891 non-null int64
    Sex_applyDF                          891 non-null int64
    Embarked_applyDF                     891 non-null object
    Age_per_Pclass                       891 non-null object
    dtypes: float64(2), int64(9), object(7)
    memory usage: 125.4+ KB
    


```python
# there are some columns in int and float type
# Let's say we want to convert these all the int types and float types in float types
# The best way of getting these column names are from describe method
titanic_train.describe().columns
titanic_train_floats = titanic_train[titanic_train.describe().columns].applymap(float)
titanic_train_floats.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 11 columns):
    PassengerId                          891 non-null float64
    Survived                             891 non-null float64
    Pclass                               891 non-null float64
    Age                                  714 non-null float64
    SibSp                                891 non-null float64
    Parch                                891 non-null float64
    Fare                                 891 non-null float64
    Sex_0_1                              891 non-null float64
    Sex_01_applySeriesMethod             891 non-null float64
    Sex_01_applySeriesMethod_addPrint    891 non-null float64
    Sex_applyDF                          891 non-null float64
    dtypes: float64(11)
    memory usage: 76.6 KB
    
