
# Initialization and Optimization

For this lab on initialization and optimization, let's look at a slightly different type of neural network. This time, we will not perform a classification task as we've done before.  Instead, we'll look at a regression problem.

We can just as well use deep learning networks for regression as for a classification problem. However, note that getting regression to work with neural networks is a harder problem because the output is unbounded ($\hat y$ can technically range from $-\infty$ to $+\infty$, and the models are especially prone to **_exploding gradients_**. This issue makes a regression exercise the perfect learning case!

Run the cell below to import everything we'll need for this lab.


```python
import numpy as np
np.random.seed(0)
import pandas as pd
from keras.models import Sequential
from keras import initializers
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from keras import optimizers
from sklearn.model_selection import train_test_split
```

    Using TensorFlow backend.


## 1. Loading the data

The data we'll be working with is data related to facebook posts published during the year of 2014 on the Facebook's page of a renowned cosmetics brand.  It includes 7 features known prior to post publication, and 12 features for evaluating the post impact. What we want to do is make a predictor for the number of "likes" for a post, taking into account the 7 features prior to posting.

First, let's import the data set and delete any rows with missing data.  

The dataset is contained with the file `dataset_Facebook.csv`. In the cell below, use pandas to read in the data from this file. Because of the way the data is structure, make sure you also set the `sep` parameter to `";"`, and the `header` parameter to `0`. 

Then, use the DataFrame's built-in `.dropna()` function to remove any rows with missing values. 


```python
# load dataset
data = pd.read_csv("dataset_Facebook.csv", sep = ";", header=0)
data = data.dropna()
```

Now, let's check the shape of our data to ensure that everything looks correct. 


```python
np.shape(data) #Expected Output: (495, 19)
```




    (495, 19)



And finally, let's inspect the `.head()` of the DataFrame to get a feel for what our dataset looks like. 


```python
data.head()
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
      <th>Page total likes</th>
      <th>Type</th>
      <th>Category</th>
      <th>Post Month</th>
      <th>Post Weekday</th>
      <th>Post Hour</th>
      <th>Paid</th>
      <th>Lifetime Post Total Reach</th>
      <th>Lifetime Post Total Impressions</th>
      <th>Lifetime Engaged Users</th>
      <th>Lifetime Post Consumers</th>
      <th>Lifetime Post Consumptions</th>
      <th>Lifetime Post Impressions by people who have liked your Page</th>
      <th>Lifetime Post reach by people who like your Page</th>
      <th>Lifetime People who have liked your Page and engaged with your post</th>
      <th>comment</th>
      <th>like</th>
      <th>share</th>
      <th>Total Interactions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>139441</td>
      <td>Photo</td>
      <td>2</td>
      <td>12</td>
      <td>4</td>
      <td>3</td>
      <td>0.0</td>
      <td>2752</td>
      <td>5091</td>
      <td>178</td>
      <td>109</td>
      <td>159</td>
      <td>3078</td>
      <td>1640</td>
      <td>119</td>
      <td>4</td>
      <td>79.0</td>
      <td>17.0</td>
      <td>100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>139441</td>
      <td>Status</td>
      <td>2</td>
      <td>12</td>
      <td>3</td>
      <td>10</td>
      <td>0.0</td>
      <td>10460</td>
      <td>19057</td>
      <td>1457</td>
      <td>1361</td>
      <td>1674</td>
      <td>11710</td>
      <td>6112</td>
      <td>1108</td>
      <td>5</td>
      <td>130.0</td>
      <td>29.0</td>
      <td>164</td>
    </tr>
    <tr>
      <th>2</th>
      <td>139441</td>
      <td>Photo</td>
      <td>3</td>
      <td>12</td>
      <td>3</td>
      <td>3</td>
      <td>0.0</td>
      <td>2413</td>
      <td>4373</td>
      <td>177</td>
      <td>113</td>
      <td>154</td>
      <td>2812</td>
      <td>1503</td>
      <td>132</td>
      <td>0</td>
      <td>66.0</td>
      <td>14.0</td>
      <td>80</td>
    </tr>
    <tr>
      <th>3</th>
      <td>139441</td>
      <td>Photo</td>
      <td>2</td>
      <td>12</td>
      <td>2</td>
      <td>10</td>
      <td>1.0</td>
      <td>50128</td>
      <td>87991</td>
      <td>2211</td>
      <td>790</td>
      <td>1119</td>
      <td>61027</td>
      <td>32048</td>
      <td>1386</td>
      <td>58</td>
      <td>1572.0</td>
      <td>147.0</td>
      <td>1777</td>
    </tr>
    <tr>
      <th>4</th>
      <td>139441</td>
      <td>Photo</td>
      <td>2</td>
      <td>12</td>
      <td>2</td>
      <td>3</td>
      <td>0.0</td>
      <td>7244</td>
      <td>13594</td>
      <td>671</td>
      <td>410</td>
      <td>580</td>
      <td>6228</td>
      <td>3200</td>
      <td>396</td>
      <td>19</td>
      <td>325.0</td>
      <td>49.0</td>
      <td>393</td>
    </tr>
  </tbody>
</table>
</div>



## 2. Initialization

### 2.1 Normalize the input data

A big part of Deep Learning is cleaning the data and getting into a shape usable by a neural network.  Let's get some additional practice with this.


Take a look at our input data. We'll use the 7 first columns as our predictors. We'll do the following two things:
- Normalize the continuous variables --> you can do this using `np.mean()` and `np.std()`
- make dummy variables of the categorical variables (you can do this by using `pd.get_dummies`)

We only count "Category" and "Type" as categorical variables. Note that you can argue that "Post month", "Post Weekday" and "Post Hour" can also be considered categories, but we'll just treat them as being continuous for now.

In the cell below, convert the data as needed by normalizing or converting to dummy variables, and then concatenate it all back into a single DataFrame once you've finished.  


```python
X0 = data["Page total likes"]
X1 = data["Type"]
X2 = data["Category"]
X3 = data["Post Month"]
X4 = data["Post Weekday"]
X5 = data["Post Hour"]
X6 = data["Paid"]

## standardize/categorize
X0= (X0-np.mean(X0))/(np.std(X0))
dummy_X1= pd.get_dummies(X1)
dummy_X2= pd.get_dummies(X2)
X3= (X3-np.mean(X3))/(np.std(X3))
X4= (X4-np.mean(X4))/(np.std(X4))
X5= (X5-np.mean(X5))/(np.std(X5))

# Add them all back into a single DataFrame
X = pd.concat([X0, dummy_X1, dummy_X2, X3, X4, X5, X6], axis=1)

# Store our labels in a separate variable
Y = data["like"]
```


```python
#Note: you get the same result for standardization if you use StandardScaler from sklearn.preprocessing

#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X0 = sc.fit_transform(X0)
```

Our data is fairly small. Let's just split the data up in a training set and a validation set!

In the cell below:

* Split the data into training and testing sets by passing `X` and `Y` into `train_test_split`.  Set a `test_size` of `0.2`.


```python
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)
```

Let's check the shape to make sure everything worked correctly.


```python
X_val.shape # Expected Output: (99, 12)
```




    (99, 12)




```python
X_train.shape # Expected Output: (396, 12)
```




    (396, 12)



## Building a Neural Network for Regression

Now, we'll build a neural network to predict the number of likes we think a post will receive.  

In the cell below, create a model with the following specifications:

* 1 Hidden Layer with 8 neurons.  In this layer, also set `input_dim` to `12`, and `activation` to `"relu"`.
* An output layer with 1 neuron.  For this neuron, set the activation to `linear`.  


```python
model = Sequential()
model.add(Dense(8, input_dim=12, activation='relu'))
model.add(Dense(1, activation = 'linear'))
```

Now, we need to compile the model, with the following hyperparameters:

* `optimizer='sgd'`
* `loss='mse'`
* `metrics=['mse']`

Note that since our model is training for a regression task, not a classification task, we'll need to use a loss metric that corresponds with regression tasks--Mean Squared Error. 


```python
model.compile(optimizer= "sgd" ,loss='mse',metrics=['mse'])
```

Finally, let's train the model.  Call `model.fit()`. In addition to to the training data and labels, also set:

* `batch_size=32`
* `epochs=100`
* `verbose=1`
* `validation_data=(X_val, y_val)`


```python
hist = model.fit(X_train, Y_train, batch_size=32, 
                 epochs=100, validation_data = (X_val, Y_val), verbose=1)
```

    Train on 396 samples, validate on 99 samples
    Epoch 1/100
    396/396 [==============================] - 0s 141us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 2/100
    396/396 [==============================] - 0s 47us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 3/100
    396/396 [==============================] - 0s 59us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 4/100
    396/396 [==============================] - 0s 61us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 5/100
    396/396 [==============================] - 0s 59us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 6/100
    396/396 [==============================] - 0s 59us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 7/100
    396/396 [==============================] - 0s 59us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 8/100
    396/396 [==============================] - 0s 56us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 9/100
    396/396 [==============================] - 0s 56us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 10/100
    396/396 [==============================] - 0s 61us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 11/100
    396/396 [==============================] - 0s 53us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 12/100
    396/396 [==============================] - 0s 68us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 13/100
    396/396 [==============================] - 0s 58us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 14/100
    396/396 [==============================] - 0s 65us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 15/100
    396/396 [==============================] - 0s 65us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 16/100
    396/396 [==============================] - 0s 60us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 17/100
    396/396 [==============================] - 0s 62us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 18/100
    396/396 [==============================] - 0s 63us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 19/100
    396/396 [==============================] - 0s 55us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 20/100
    396/396 [==============================] - 0s 57us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 21/100
    396/396 [==============================] - 0s 59us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 22/100
    396/396 [==============================] - 0s 57us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 23/100
    396/396 [==============================] - 0s 63us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 24/100
    396/396 [==============================] - 0s 59us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 25/100
    396/396 [==============================] - 0s 63us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 26/100
    396/396 [==============================] - 0s 57us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 27/100
    396/396 [==============================] - 0s 63us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 28/100
    396/396 [==============================] - 0s 47us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 29/100
    396/396 [==============================] - 0s 63us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 30/100
    396/396 [==============================] - 0s 49us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 31/100
    396/396 [==============================] - 0s 64us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 32/100
    396/396 [==============================] - 0s 45us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 33/100
    396/396 [==============================] - 0s 64us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 34/100
    396/396 [==============================] - 0s 48us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 35/100
    396/396 [==============================] - 0s 50us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 36/100
    396/396 [==============================] - 0s 65us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 37/100
    396/396 [==============================] - 0s 57us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 38/100
    396/396 [==============================] - 0s 57us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 39/100
    396/396 [==============================] - 0s 55us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 40/100
    396/396 [==============================] - 0s 54us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 41/100
    396/396 [==============================] - 0s 64us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 42/100
    396/396 [==============================] - 0s 53us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 43/100
    396/396 [==============================] - 0s 50us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 44/100
    396/396 [==============================] - 0s 70us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 45/100
    396/396 [==============================] - 0s 58us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 46/100
    396/396 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 47/100
    396/396 [==============================] - 0s 47us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 48/100
    396/396 [==============================] - 0s 47us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 49/100
    396/396 [==============================] - 0s 47us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 50/100
    396/396 [==============================] - 0s 50us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 51/100
    396/396 [==============================] - 0s 48us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 52/100
    396/396 [==============================] - 0s 47us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 53/100
    396/396 [==============================] - 0s 47us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 54/100
    396/396 [==============================] - 0s 45us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 55/100
    396/396 [==============================] - 0s 45us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 56/100
    396/396 [==============================] - 0s 48us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 57/100
    396/396 [==============================] - 0s 44us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 58/100
    396/396 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 59/100
    396/396 [==============================] - 0s 55us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 60/100
    396/396 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 61/100
    396/396 [==============================] - 0s 53us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 62/100
    396/396 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 63/100
    396/396 [==============================] - 0s 48us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 64/100
    396/396 [==============================] - 0s 49us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 65/100
    396/396 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 66/100
    396/396 [==============================] - 0s 53us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 67/100
    396/396 [==============================] - 0s 47us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 68/100
    396/396 [==============================] - 0s 54us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 69/100
    396/396 [==============================] - 0s 50us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 70/100
    396/396 [==============================] - 0s 49us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 71/100
    396/396 [==============================] - 0s 53us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 72/100
    396/396 [==============================] - 0s 47us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 73/100
    396/396 [==============================] - 0s 48us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 74/100
    396/396 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 75/100
    396/396 [==============================] - 0s 47us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 76/100
    396/396 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 77/100
    396/396 [==============================] - 0s 47us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 78/100
    396/396 [==============================] - 0s 53us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 79/100
    396/396 [==============================] - 0s 49us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 80/100
    396/396 [==============================] - 0s 52us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 81/100
    396/396 [==============================] - 0s 42us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 82/100
    396/396 [==============================] - 0s 50us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 83/100
    396/396 [==============================] - 0s 50us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 84/100
    396/396 [==============================] - 0s 47us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 85/100
    396/396 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 86/100
    396/396 [==============================] - 0s 44us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 87/100
    396/396 [==============================] - 0s 51us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 88/100
    396/396 [==============================] - 0s 46us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 89/100
    396/396 [==============================] - 0s 47us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 90/100
    396/396 [==============================] - 0s 46us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 91/100
    396/396 [==============================] - 0s 47us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 92/100
    396/396 [==============================] - 0s 54us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 93/100
    396/396 [==============================] - 0s 47us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 94/100
    396/396 [==============================] - 0s 48us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 95/100
    396/396 [==============================] - 0s 54us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 96/100
    396/396 [==============================] - 0s 47us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 97/100
    396/396 [==============================] - 0s 53us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 98/100
    396/396 [==============================] - 0s 47us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 99/100
    396/396 [==============================] - 0s 47us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan
    Epoch 100/100
    396/396 [==============================] - 0s 53us/step - loss: nan - mean_squared_error: nan - val_loss: nan - val_mean_squared_error: nan


Did you see what happend? all the values for training and validation loss are "nan". There could be several reasons for that, but as we already mentioned there is likely a vanishing or exploding gradient problem.  This means that the values got so large or so small that they no longer fit in memory.   R

Recall that we normalized out inputs. But how about the outputs? Let's have a look.


```python
Y_train.head()
```




    212     36.0
    107    193.0
    411     75.0
    71     449.0
    473    136.0
    Name: like, dtype: float64



Yes, indeed. We didn't normalize them and we should, as they take pretty high values. Let
s rerun the model but make sure that the output is normalized as well!

### 2.2 Normalizing the output

In the cell below, we've included all the normalization code that we wrote up top, but this time, we've added a line to normalize the data in `Y`, as well. This should help alot!

Run the cell below to normalize our data and our labels.


```python
X0 = data["Page total likes"]
X1 = data["Type"]
X2 = data["Category"]
X3 = data["Post Month"]
X4 = data["Post Weekday"]
X5 = data["Post Hour"]
X6 = data["Paid"]

## standardize/categorize
X0= (X0-np.mean(X0))/(np.std(X0))
dummy_X1= pd.get_dummies(X1)
dummy_X2= pd.get_dummies(X2)
X3= (X3-np.mean(X3))/(np.std(X3))
X4= (X4-np.mean(X4))/(np.std(X4))
X5= (X5-np.mean(X5))/(np.std(X5))

X = pd.concat([X0, dummy_X1, dummy_X2, X3, X4, X5, X6], axis=1)

Y = (data["like"]-np.mean(data["like"]))/(np.std(data["like"]))
```

Now, let's split our data into appropriate training and testing sets again.  Split the data, just like we did before.  Use the same `test_size` as we did last time, too. 


```python
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)
```

Now, let's reinitialize our model and build it from scratch again.  

**_NOTE:_**  If we don't reinitialize our model, our training would start with the weight values we ended with during the last training session.  In order to start fresh, we need to declare a new `Sequential()` object.  

Build the model with the exact same architecture and hyperparameters as we did above.


```python
model = Sequential()
model.add(Dense(8, input_dim=12, activation='relu'))
model.add(Dense(1, activation = 'linear'))
```

Now, compile the model with the same parameters we used before.


```python
model.compile(optimizer= "sgd" ,loss='mse',metrics=['mse'])
```

And finally, fit the model using the same parameters we did before. 


```python
hist = model.fit(X_train, Y_train, batch_size=32, 
                 epochs=100, validation_data = (X_val, Y_val), verbose = 1)
```

    Train on 396 samples, validate on 99 samples
    Epoch 1/100
    396/396 [==============================] - 0s 157us/step - loss: 1.7564 - mean_squared_error: 1.7564 - val_loss: 0.3350 - val_mean_squared_error: 0.3350
    Epoch 2/100
    396/396 [==============================] - 0s 44us/step - loss: 1.3417 - mean_squared_error: 1.3417 - val_loss: 0.3003 - val_mean_squared_error: 0.3003
    Epoch 3/100
    396/396 [==============================] - 0s 62us/step - loss: 1.2700 - mean_squared_error: 1.2700 - val_loss: 0.3007 - val_mean_squared_error: 0.3007
    Epoch 4/100
    396/396 [==============================] - 0s 68us/step - loss: 1.2396 - mean_squared_error: 1.2396 - val_loss: 0.2924 - val_mean_squared_error: 0.2924
    Epoch 5/100
    396/396 [==============================] - 0s 57us/step - loss: 1.2187 - mean_squared_error: 1.2187 - val_loss: 0.2871 - val_mean_squared_error: 0.2871
    Epoch 6/100
    396/396 [==============================] - 0s 69us/step - loss: 1.2036 - mean_squared_error: 1.2036 - val_loss: 0.2840 - val_mean_squared_error: 0.2840
    Epoch 7/100
    396/396 [==============================] - 0s 62us/step - loss: 1.1944 - mean_squared_error: 1.1944 - val_loss: 0.2734 - val_mean_squared_error: 0.2734
    Epoch 8/100
    396/396 [==============================] - 0s 61us/step - loss: 1.1836 - mean_squared_error: 1.1836 - val_loss: 0.2706 - val_mean_squared_error: 0.2706
    Epoch 9/100
    396/396 [==============================] - 0s 66us/step - loss: 1.1727 - mean_squared_error: 1.1727 - val_loss: 0.2823 - val_mean_squared_error: 0.2823
    Epoch 10/100
    396/396 [==============================] - 0s 62us/step - loss: 1.1647 - mean_squared_error: 1.1647 - val_loss: 0.2893 - val_mean_squared_error: 0.2893
    Epoch 11/100
    396/396 [==============================] - 0s 65us/step - loss: 1.1561 - mean_squared_error: 1.1561 - val_loss: 0.2701 - val_mean_squared_error: 0.2701
    Epoch 12/100
    396/396 [==============================] - 0s 66us/step - loss: 1.1517 - mean_squared_error: 1.1517 - val_loss: 0.2773 - val_mean_squared_error: 0.2773
    Epoch 13/100
    396/396 [==============================] - 0s 60us/step - loss: 1.1462 - mean_squared_error: 1.1462 - val_loss: 0.2630 - val_mean_squared_error: 0.2630
    Epoch 14/100
    396/396 [==============================] - 0s 63us/step - loss: 1.1422 - mean_squared_error: 1.1422 - val_loss: 0.2688 - val_mean_squared_error: 0.2688
    Epoch 15/100
    396/396 [==============================] - 0s 56us/step - loss: 1.1383 - mean_squared_error: 1.1383 - val_loss: 0.2531 - val_mean_squared_error: 0.2531
    Epoch 16/100
    396/396 [==============================] - 0s 66us/step - loss: 1.1368 - mean_squared_error: 1.1368 - val_loss: 0.2582 - val_mean_squared_error: 0.2582
    Epoch 17/100
    396/396 [==============================] - 0s 57us/step - loss: 1.1345 - mean_squared_error: 1.1345 - val_loss: 0.2619 - val_mean_squared_error: 0.2619
    Epoch 18/100
    396/396 [==============================] - 0s 65us/step - loss: 1.1330 - mean_squared_error: 1.1330 - val_loss: 0.2666 - val_mean_squared_error: 0.2666
    Epoch 19/100
    396/396 [==============================] - 0s 60us/step - loss: 1.1296 - mean_squared_error: 1.1296 - val_loss: 0.2803 - val_mean_squared_error: 0.2803
    Epoch 20/100
    396/396 [==============================] - 0s 60us/step - loss: 1.1279 - mean_squared_error: 1.1279 - val_loss: 0.2630 - val_mean_squared_error: 0.2630
    Epoch 21/100
    396/396 [==============================] - 0s 65us/step - loss: 1.1266 - mean_squared_error: 1.1266 - val_loss: 0.2556 - val_mean_squared_error: 0.2556
    Epoch 22/100
    396/396 [==============================] - 0s 58us/step - loss: 1.1236 - mean_squared_error: 1.1236 - val_loss: 0.2602 - val_mean_squared_error: 0.2602
    Epoch 23/100
    396/396 [==============================] - 0s 62us/step - loss: 1.1199 - mean_squared_error: 1.1199 - val_loss: 0.2481 - val_mean_squared_error: 0.2481
    Epoch 24/100
    396/396 [==============================] - 0s 54us/step - loss: 1.1208 - mean_squared_error: 1.1208 - val_loss: 0.2550 - val_mean_squared_error: 0.2550
    Epoch 25/100
    396/396 [==============================] - 0s 64us/step - loss: 1.1187 - mean_squared_error: 1.1187 - val_loss: 0.2657 - val_mean_squared_error: 0.2657
    Epoch 26/100
    396/396 [==============================] - 0s 52us/step - loss: 1.1155 - mean_squared_error: 1.1155 - val_loss: 0.2653 - val_mean_squared_error: 0.2653
    Epoch 27/100
    396/396 [==============================] - 0s 59us/step - loss: 1.1133 - mean_squared_error: 1.1133 - val_loss: 0.2827 - val_mean_squared_error: 0.2827
    Epoch 28/100
    396/396 [==============================] - 0s 55us/step - loss: 1.1109 - mean_squared_error: 1.1109 - val_loss: 0.2574 - val_mean_squared_error: 0.2574
    Epoch 29/100
    396/396 [==============================] - 0s 57us/step - loss: 1.1115 - mean_squared_error: 1.1115 - val_loss: 0.2590 - val_mean_squared_error: 0.2590
    Epoch 30/100
    396/396 [==============================] - 0s 57us/step - loss: 1.1099 - mean_squared_error: 1.1099 - val_loss: 0.2621 - val_mean_squared_error: 0.2621
    Epoch 31/100
    396/396 [==============================] - 0s 53us/step - loss: 1.1092 - mean_squared_error: 1.1092 - val_loss: 0.2676 - val_mean_squared_error: 0.2676
    Epoch 32/100
    396/396 [==============================] - 0s 62us/step - loss: 1.1082 - mean_squared_error: 1.1082 - val_loss: 0.2687 - val_mean_squared_error: 0.2687
    Epoch 33/100
    396/396 [==============================] - 0s 54us/step - loss: 1.1074 - mean_squared_error: 1.1074 - val_loss: 0.2609 - val_mean_squared_error: 0.2609
    Epoch 34/100
    396/396 [==============================] - 0s 61us/step - loss: 1.1064 - mean_squared_error: 1.1064 - val_loss: 0.2577 - val_mean_squared_error: 0.2577
    Epoch 35/100
    396/396 [==============================] - 0s 51us/step - loss: 1.1081 - mean_squared_error: 1.1081 - val_loss: 0.2548 - val_mean_squared_error: 0.2548
    Epoch 36/100
    396/396 [==============================] - 0s 60us/step - loss: 1.1048 - mean_squared_error: 1.1048 - val_loss: 0.2609 - val_mean_squared_error: 0.2609
    Epoch 37/100
    396/396 [==============================] - 0s 52us/step - loss: 1.1018 - mean_squared_error: 1.1018 - val_loss: 0.2697 - val_mean_squared_error: 0.2697
    Epoch 38/100
    396/396 [==============================] - 0s 55us/step - loss: 1.1022 - mean_squared_error: 1.1022 - val_loss: 0.2647 - val_mean_squared_error: 0.2647
    Epoch 39/100
    396/396 [==============================] - 0s 61us/step - loss: 1.1010 - mean_squared_error: 1.1010 - val_loss: 0.2692 - val_mean_squared_error: 0.2692
    Epoch 40/100
    396/396 [==============================] - 0s 52us/step - loss: 1.0985 - mean_squared_error: 1.0985 - val_loss: 0.2885 - val_mean_squared_error: 0.2885
    Epoch 41/100
    396/396 [==============================] - 0s 57us/step - loss: 1.0990 - mean_squared_error: 1.0990 - val_loss: 0.2637 - val_mean_squared_error: 0.2637
    Epoch 42/100
    396/396 [==============================] - 0s 69us/step - loss: 1.0977 - mean_squared_error: 1.0977 - val_loss: 0.2552 - val_mean_squared_error: 0.2552
    Epoch 43/100
    396/396 [==============================] - 0s 60us/step - loss: 1.0979 - mean_squared_error: 1.0979 - val_loss: 0.2570 - val_mean_squared_error: 0.2570
    Epoch 44/100
    396/396 [==============================] - 0s 55us/step - loss: 1.0963 - mean_squared_error: 1.0963 - val_loss: 0.2592 - val_mean_squared_error: 0.2592
    Epoch 45/100
    396/396 [==============================] - 0s 58us/step - loss: 1.0930 - mean_squared_error: 1.0930 - val_loss: 0.2602 - val_mean_squared_error: 0.2602
    Epoch 46/100
    396/396 [==============================] - 0s 50us/step - loss: 1.0924 - mean_squared_error: 1.0924 - val_loss: 0.2623 - val_mean_squared_error: 0.2623
    Epoch 47/100
    396/396 [==============================] - 0s 47us/step - loss: 1.0919 - mean_squared_error: 1.0919 - val_loss: 0.2625 - val_mean_squared_error: 0.2625
    Epoch 48/100
    396/396 [==============================] - 0s 46us/step - loss: 1.0915 - mean_squared_error: 1.0915 - val_loss: 0.2561 - val_mean_squared_error: 0.2561
    Epoch 49/100
    396/396 [==============================] - 0s 47us/step - loss: 1.0919 - mean_squared_error: 1.0919 - val_loss: 0.2517 - val_mean_squared_error: 0.2517
    Epoch 50/100
    396/396 [==============================] - 0s 46us/step - loss: 1.0905 - mean_squared_error: 1.0905 - val_loss: 0.2573 - val_mean_squared_error: 0.2573
    Epoch 51/100
    396/396 [==============================] - 0s 50us/step - loss: 1.0907 - mean_squared_error: 1.0907 - val_loss: 0.2604 - val_mean_squared_error: 0.2604
    Epoch 52/100
    396/396 [==============================] - 0s 45us/step - loss: 1.0878 - mean_squared_error: 1.0878 - val_loss: 0.2538 - val_mean_squared_error: 0.2538
    Epoch 53/100
    396/396 [==============================] - 0s 47us/step - loss: 1.0877 - mean_squared_error: 1.0877 - val_loss: 0.2496 - val_mean_squared_error: 0.2496
    Epoch 54/100
    396/396 [==============================] - 0s 45us/step - loss: 1.0882 - mean_squared_error: 1.0882 - val_loss: 0.2553 - val_mean_squared_error: 0.2553
    Epoch 55/100
    396/396 [==============================] - 0s 53us/step - loss: 1.0862 - mean_squared_error: 1.0862 - val_loss: 0.2617 - val_mean_squared_error: 0.2617
    Epoch 56/100
    396/396 [==============================] - 0s 53us/step - loss: 1.0858 - mean_squared_error: 1.0858 - val_loss: 0.2833 - val_mean_squared_error: 0.2833
    Epoch 57/100
    396/396 [==============================] - 0s 54us/step - loss: 1.0832 - mean_squared_error: 1.0832 - val_loss: 0.2800 - val_mean_squared_error: 0.2800
    Epoch 58/100
    396/396 [==============================] - 0s 58us/step - loss: 1.0825 - mean_squared_error: 1.0825 - val_loss: 0.2727 - val_mean_squared_error: 0.2727
    Epoch 59/100
    396/396 [==============================] - 0s 55us/step - loss: 1.0826 - mean_squared_error: 1.0826 - val_loss: 0.2894 - val_mean_squared_error: 0.2894
    Epoch 60/100
    396/396 [==============================] - 0s 54us/step - loss: 1.0834 - mean_squared_error: 1.0834 - val_loss: 0.2752 - val_mean_squared_error: 0.2752
    Epoch 61/100
    396/396 [==============================] - 0s 50us/step - loss: 1.0810 - mean_squared_error: 1.0810 - val_loss: 0.2650 - val_mean_squared_error: 0.2650
    Epoch 62/100
    396/396 [==============================] - 0s 55us/step - loss: 1.0818 - mean_squared_error: 1.0818 - val_loss: 0.2640 - val_mean_squared_error: 0.2640
    Epoch 63/100
    396/396 [==============================] - 0s 51us/step - loss: 1.0801 - mean_squared_error: 1.0801 - val_loss: 0.2709 - val_mean_squared_error: 0.2709
    Epoch 64/100
    396/396 [==============================] - 0s 52us/step - loss: 1.0800 - mean_squared_error: 1.0800 - val_loss: 0.2661 - val_mean_squared_error: 0.2661
    Epoch 65/100
    396/396 [==============================] - 0s 56us/step - loss: 1.0781 - mean_squared_error: 1.0781 - val_loss: 0.2737 - val_mean_squared_error: 0.2737
    Epoch 66/100
    396/396 [==============================] - 0s 50us/step - loss: 1.0793 - mean_squared_error: 1.0793 - val_loss: 0.2629 - val_mean_squared_error: 0.2629
    Epoch 67/100
    396/396 [==============================] - 0s 51us/step - loss: 1.0771 - mean_squared_error: 1.0771 - val_loss: 0.2593 - val_mean_squared_error: 0.2593
    Epoch 68/100
    396/396 [==============================] - 0s 49us/step - loss: 1.0777 - mean_squared_error: 1.0777 - val_loss: 0.2864 - val_mean_squared_error: 0.2864
    Epoch 69/100
    396/396 [==============================] - 0s 48us/step - loss: 1.0773 - mean_squared_error: 1.0773 - val_loss: 0.2755 - val_mean_squared_error: 0.2755
    Epoch 70/100
    396/396 [==============================] - 0s 47us/step - loss: 1.0745 - mean_squared_error: 1.0745 - val_loss: 0.2791 - val_mean_squared_error: 0.2791
    Epoch 71/100
    396/396 [==============================] - 0s 52us/step - loss: 1.0751 - mean_squared_error: 1.0751 - val_loss: 0.2670 - val_mean_squared_error: 0.2670
    Epoch 72/100
    396/396 [==============================] - 0s 50us/step - loss: 1.0738 - mean_squared_error: 1.0738 - val_loss: 0.2730 - val_mean_squared_error: 0.2730
    Epoch 73/100
    396/396 [==============================] - 0s 50us/step - loss: 1.0719 - mean_squared_error: 1.0719 - val_loss: 0.2752 - val_mean_squared_error: 0.2752
    Epoch 74/100
    396/396 [==============================] - 0s 50us/step - loss: 1.0731 - mean_squared_error: 1.0731 - val_loss: 0.3143 - val_mean_squared_error: 0.3143
    Epoch 75/100
    396/396 [==============================] - 0s 50us/step - loss: 1.0758 - mean_squared_error: 1.0758 - val_loss: 0.2964 - val_mean_squared_error: 0.2964
    Epoch 76/100
    396/396 [==============================] - 0s 51us/step - loss: 1.0750 - mean_squared_error: 1.0750 - val_loss: 0.2819 - val_mean_squared_error: 0.2819
    Epoch 77/100
    396/396 [==============================] - 0s 51us/step - loss: 1.0722 - mean_squared_error: 1.0722 - val_loss: 0.2824 - val_mean_squared_error: 0.2824
    Epoch 78/100
    396/396 [==============================] - 0s 46us/step - loss: 1.0730 - mean_squared_error: 1.0730 - val_loss: 0.2712 - val_mean_squared_error: 0.2712
    Epoch 79/100
    396/396 [==============================] - 0s 50us/step - loss: 1.0705 - mean_squared_error: 1.0705 - val_loss: 0.2746 - val_mean_squared_error: 0.2746
    Epoch 80/100
    396/396 [==============================] - 0s 48us/step - loss: 1.0711 - mean_squared_error: 1.0711 - val_loss: 0.2962 - val_mean_squared_error: 0.2962
    Epoch 81/100
    396/396 [==============================] - 0s 45us/step - loss: 1.0700 - mean_squared_error: 1.0700 - val_loss: 0.2845 - val_mean_squared_error: 0.2845
    Epoch 82/100
    396/396 [==============================] - 0s 49us/step - loss: 1.0688 - mean_squared_error: 1.0688 - val_loss: 0.2767 - val_mean_squared_error: 0.2767
    Epoch 83/100
    396/396 [==============================] - 0s 49us/step - loss: 1.0686 - mean_squared_error: 1.0686 - val_loss: 0.2711 - val_mean_squared_error: 0.2711
    Epoch 84/100
    396/396 [==============================] - 0s 46us/step - loss: 1.0675 - mean_squared_error: 1.0675 - val_loss: 0.2658 - val_mean_squared_error: 0.2658
    Epoch 85/100
    396/396 [==============================] - 0s 49us/step - loss: 1.0685 - mean_squared_error: 1.0685 - val_loss: 0.2624 - val_mean_squared_error: 0.2624
    Epoch 86/100
    396/396 [==============================] - 0s 46us/step - loss: 1.0656 - mean_squared_error: 1.0656 - val_loss: 0.2696 - val_mean_squared_error: 0.2696
    Epoch 87/100
    396/396 [==============================] - 0s 45us/step - loss: 1.0650 - mean_squared_error: 1.0650 - val_loss: 0.2678 - val_mean_squared_error: 0.2678
    Epoch 88/100
    396/396 [==============================] - 0s 47us/step - loss: 1.0648 - mean_squared_error: 1.0648 - val_loss: 0.3027 - val_mean_squared_error: 0.3027
    Epoch 89/100
    396/396 [==============================] - 0s 44us/step - loss: 1.0670 - mean_squared_error: 1.0670 - val_loss: 0.2796 - val_mean_squared_error: 0.2796
    Epoch 90/100
    396/396 [==============================] - 0s 51us/step - loss: 1.0645 - mean_squared_error: 1.0645 - val_loss: 0.2724 - val_mean_squared_error: 0.2724
    Epoch 91/100
    396/396 [==============================] - 0s 47us/step - loss: 1.0653 - mean_squared_error: 1.0653 - val_loss: 0.2857 - val_mean_squared_error: 0.2857
    Epoch 92/100
    396/396 [==============================] - 0s 48us/step - loss: 1.0633 - mean_squared_error: 1.0633 - val_loss: 0.2741 - val_mean_squared_error: 0.2741
    Epoch 93/100
    396/396 [==============================] - 0s 46us/step - loss: 1.0637 - mean_squared_error: 1.0637 - val_loss: 0.2824 - val_mean_squared_error: 0.2824
    Epoch 94/100
    396/396 [==============================] - 0s 48us/step - loss: 1.0615 - mean_squared_error: 1.0615 - val_loss: 0.2891 - val_mean_squared_error: 0.2891
    Epoch 95/100
    396/396 [==============================] - 0s 53us/step - loss: 1.0601 - mean_squared_error: 1.0601 - val_loss: 0.2763 - val_mean_squared_error: 0.2763
    Epoch 96/100
    396/396 [==============================] - 0s 44us/step - loss: 1.0614 - mean_squared_error: 1.0614 - val_loss: 0.3228 - val_mean_squared_error: 0.3228
    Epoch 97/100
    396/396 [==============================] - 0s 51us/step - loss: 1.0644 - mean_squared_error: 1.0644 - val_loss: 0.2816 - val_mean_squared_error: 0.2816
    Epoch 98/100
    396/396 [==============================] - 0s 45us/step - loss: 1.0601 - mean_squared_error: 1.0601 - val_loss: 0.2788 - val_mean_squared_error: 0.2788
    Epoch 99/100
    396/396 [==============================] - 0s 42us/step - loss: 1.0580 - mean_squared_error: 1.0580 - val_loss: 0.2817 - val_mean_squared_error: 0.2817
    Epoch 100/100
    396/396 [==============================] - 0s 45us/step - loss: 1.0581 - mean_squared_error: 1.0581 - val_loss: 0.2836 - val_mean_squared_error: 0.2836


The model did much, much better this time around!

Run the cell below to get the model's predictions for both the training and validation sets. 


```python
pred_train = model.predict(X_train).reshape(-1)
pred_val = model.predict(X_val).reshape(-1)
```

Let's look at the first 10 predictions from `pred_train`. Display those in the cell below.


```python
pred_train[:10]
```




    array([ 0.13764419, -0.48095602, -0.30586046,  0.54349709, -0.08094749,
           -0.04072087,  0.09825368,  0.10513391, -0.43215472, -0.06913403], dtype=float32)



Let's manually calculate the Mean Squared Error in the cell below.  

As a refresher, here's the formula for calculating Mean Squared Error:

<img src='mse_formula.gif'>

Use `pred_train` and `Y_train` to calculate our training MSE in the cell below.  

**_HINT:_** Use numpy to make short work of this!


```python
MSE_train = np.mean((pred_train-Y_train)**2)
MSE_train 
```




    1.0533202252145233



Now, calculate the MSE for our validation set in the cell below.


```python
MSE_val = np.mean((pred_val-Y_val)**2)
MSE_val 
```




    0.28357290537863905



### 2.3 Use weight initializers

Another way to increase the performance of our models is to initialize our weights in clever ways.  We'll explore some of those options below.  

#### 2.3.1  He initialization

Let's try and use a weight initializer.  We'll start with the **_He normalizer_**, which initializes the weight vector to have an average 0 and a variance of 2/n, with $n$ the number of features feeding into a layer.

In the cell below:

* Recreate the Neural Network that we created above.  This time, in the hidden layer, set the `kernel_initializer` to `"he_normal"`.
* Compile and fit the model with the same hyperparameters as we used before.  


```python
model = Sequential()
model.add(Dense(8, input_dim=12, kernel_initializer= "he_normal",
                activation='relu'))
model.add(Dense(1, activation = 'linear'))

model.compile(optimizer= "sgd" ,loss='mse',metrics=['mse'])
hist = model.fit(X_train, Y_train, batch_size=32, 
                 epochs=100, validation_data = (X_val, Y_val),verbose=0)
```

Great!

Run the cells below to get training and validation predictions are recalculate our MSE for each. 


```python
pred_train = model.predict(X_train).reshape(-1)
pred_val = model.predict(X_val).reshape(-1)

MSE_train = np.mean((pred_train-Y_train)**2)
MSE_val = np.mean((pred_val-Y_val)**2)
```


```python
print(MSE_train) 
print(MSE_val) 
```

    1.0937170669391523
    0.2951410757665797


The initializer does not really help us to decrease the MSE. We know that initializers can be particularly helpful in deeper networks, and our network isn't very deep. What if we use the `Lecun` initializer with a `tanh` activation?

#### 2.3.2  Lecun initialization

In the cell below, recreate the network again.  This time, set hidden layer's activation to `'tanh'`, and the `kernel_initializer` to `'lecun_normal'`.

Then, fit and compile the model as did before.  


```python
model = Sequential()
model.add(Dense(8, input_dim=12, 
                kernel_initializer= "lecun_normal", activation='tanh'))
model.add(Dense(1, activation = 'linear'))

model.compile(optimizer= "sgd" ,loss='mse',metrics=['mse'])
hist = model.fit(X_train, Y_train, batch_size=32, 
                 epochs=100, validation_data = (X_val, Y_val), verbose=0)
```

Now, run the cells below to get the predictions and calculate the MSE for training and validation again.  


```python
pred_train = model.predict(X_train).reshape(-1)
pred_val = model.predict(X_val).reshape(-1)

MSE_train = np.mean((pred_train-Y_train)**2)
MSE_val = np.mean((pred_val-Y_val)**2)
```


```python
print(MSE_train) 
print(MSE_val) 
```

    1.1090986602307809
    0.261894898816509


## 3. Optimization

Another option we have is to play with the optimizers we choose for gradient descent during our back propagation step.  So far, we've only made use of basic `'sgd'`, or **_Stochastic Gradient Descent_**.  However, there are more advanced optimizers available to use will often converge to better minima, usually in a quicker fashion. 

In this lab, we'll try the two most popular methods: **_RMSprop_** and **_adam_**.

### 3.1 RMSprop

In the cell below, recreate the original network that we built in this lab--no kernel intialization parameter, and the activation set to `'relu'`. 

This time, when you compile the model, set the `optimizer` parameter to `"rmsprop"`.  No changes to the `fit()` call are needed--keep those parameters the same.  


```python
model = Sequential()
model.add(Dense(8, input_dim=12, activation='relu'))
model.add(Dense(1, activation = 'linear'))

model.compile(optimizer= "rmsprop" ,loss='mse',metrics=['mse'])
hist = model.fit(X_train, Y_train, batch_size=32, 
                 epochs=100, validation_data = (X_val, Y_val), verbose = 0)
```

Now, run the cell below to get predictions and compute the MSE again.


```python
pred_train = model.predict(X_train).reshape(-1)
pred_val = model.predict(X_val).reshape(-1)

MSE_train = np.mean((pred_train-Y_train)**2)
MSE_val = np.mean((pred_val-Y_val)**2)
```


```python
print(MSE_train) 
print(MSE_val) 
```

    1.0401446175465527
    0.29571405622862257


### 3.2 Adam

Another popular optimizer is **_adam_**, which stands for `Adaptive Moment Estimation`. This is an optimzer that was created and open-sourced by a team at OpenAI, and is generally seen as the go-to choice for optimizers today. Adam combines the RMSprop algorithm with the concept of momentum, and is generally very effective at getting converging quickly.  

In the cell below, create the same network that we did above, but this time, set the optimizer parameter to `'adam'`.  Leave all other parameters the same. 


```python
model = Sequential()
model.add(Dense(8, input_dim=12, activation='relu'))
model.add(Dense(1, activation = 'linear'))

model.compile(optimizer= "Adam" ,loss='mse',metrics=['mse'])
hist = model.fit(X_train, Y_train, batch_size=32, 
                 epochs=100, validation_data = (X_val, Y_val), verbose = 0)
```


```python
pred_train = model.predict(X_train).reshape(-1)
pred_val = model.predict(X_val).reshape(-1)

MSE_train = np.mean((pred_train-Y_train)**2)
MSE_val = np.mean((pred_val-Y_val)**2)
```


```python
print(MSE_train) 
print(MSE_val) 
```

    1.0780545941567379
    0.2531044843507429


### 3.3 Learning rate decay with momentum


The final item that we'll get practice with in this lab is implementing a **_Learning Rate Decay_** strategy, along with **_Momentum_**.  We'll accomplish this by creating a `SGD` object and setting learning rate, decay, and momentum parameters at initialization.  In this way, we can then pass in the `SGD` object we've initialized to our specificataions during the compile step, rather than just a string representing an off-the-shelf `'SGD'`  optimizer.  

In the cell below:

* Create a `SGD` optimizer, which can be found in the `optimizers` module.  
    * Set the `lr` parameter to  `0.03`.
    * Set the `decay` parameter to `0.0001`
    * Set the `momentum` parameter to `0.9`.
* Recreate the same network we used during the previous example.  
* Set the optimizer parameter during the compile step to the `sgd` object we created. 
* Fit the model with the same hyperparameters as we used before.  


```python
sgd = optimizers.SGD(lr=0.03, decay=0.0001, momentum=0.9)
model = Sequential()
model.add(Dense(8, input_dim=12, activation='relu'))
model.add(Dense(1, activation = 'linear'))

model.compile(optimizer= sgd ,loss='mse',metrics=['mse'])
hist = model.fit(X_train, Y_train, batch_size=32, 
                 epochs=100, validation_data = (X_val, Y_val), verbose = 0)
```

Finally, run the cell below to calcluate the MSE for our final version of this model and see how a learning rate decay strategy affected the model. 


```python
pred_train = model.predict(X_train).reshape(-1)
pred_val = model.predict(X_val).reshape(-1)

MSE_train = np.mean((pred_train-Y_train)**2)
MSE_val = np.mean((pred_val-Y_val)**2)
```


```python
print(MSE_train) 
print(MSE_val) 
```

    0.9672024539323882
    0.3320629254936059


## Further reading

https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/

https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/

https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/

https://stackoverflow.com/questions/37232782/nan-loss-when-training-regression-network
