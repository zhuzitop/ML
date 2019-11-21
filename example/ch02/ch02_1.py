import os
import tarfile
from six.moves import urllib
%pylab
import pandas as pd

HOUSING_PATH = ""

#加载数据
def load_housing_data(housing_path=HOUSING_PATH):
	csv_path = os.path.join(housing_path, "housing.csv")
	return pd.read_csv(csv_path)
	
housing = load_housing_data()
housing.head()

#显示数据，直方图显示所有数据
#%matplotlib inline 
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()

#创建测试集
import numpy as np
def split_train_test(data, test_ratio):
	shuffled_indices = np.random.permutation(len(data))
	test_set_size = int(len(data)*test_ratio)
	test_indices = shuffled_indices[:test_set_size]
	train_indices = shuffled_indices[test_set_size:]
	return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), "train + ", len(test_set), "test")
#Scikit-learn 提供的函数，用来分成子集，random_state=42设置随机生成器种子
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size = 0.2, random_state=42)
print(len(train_set), "train + ", len(test_set), "test")

#hash标识符的测试集，固定测试集
import hashlib
def test_set_check(identifier, test_ratio, hash):
	return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio
def split_train_test_by_id(data, test_ratio, id_column, hash = hashlib.md5):
	ids = data[id_column]
	in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
	return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = housing.reset_index()
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
print(len(train_set), "train + ", len(test_set), "test")

housing["income_cat"] = np.ceil(housing["median_income"]/1.5)
housing["income_cat"].where(housing["income_cat"]<5,5.0,inplace=True)

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
	strat_train_set = housing.loc[train_index]
	strat_test_set = housing.loc[test_index]
	
housing["income_cat"].value_counts() / len(housing)

#创建一个副本
housing = strat_train_set.copy()
#地理数据可视化
housing.plot(kind="scatter",x="longitude",y="latitude")
housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.4)

housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.4,
	s=housing["population"]/100, label="population",
	c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()

#寻找相关性，皮尔逊相关系数，每对属性之间的标准相关系数
corr_matrix=housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

#Pandas的scatter_matrix函数，检测属性之间的相关性
from pandas.plotting import scatter_matrix
#import pandas as pd
attributes = ["median_house_value","median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize(12,8))

housing.plot(kind="scatter", x="median_income",y="median_house_value",alpha=0.1)

#尝试各种属性的组合
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


#数据清洗
#
#
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

housing.dropna(subset=["total_bedrooms"])
housing.drop("total_bedrooms",axis=1)
median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median)

#Scikit-Learn提供了一个非常容易上手的教程来处理缺失值：imputer
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy = "median")

housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)

imputer.statistics_
housing_num.median().values

X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)

#Scikit-Learn为这类任务提供了一个转换器LabelEncoder，将文本标签转化为数字
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)
housing_cat_encoded 

#使用classes_属性查看这个编码器的映射
print(encoder.classes_)

#OneHotEncoder编码器，可以将整数分类值转换为独热向量
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
housing_cat_1hot

housing_cat_1hot.toarray()

#自定义转换器
from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix,bedrooms_ix, population_ix, household_ix = 3,4,5,6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
	def __init__(self, add_bedrooms_per_room = True):
		self.add_bedrooms_per_room = add_bedrooms_per_room
	def fit(self, X, y=None):
		return self
	def transform(self, X, y=None):
		rooms_per_household = X[:,rooms_ix] / X[:,household_ix]
		population_per_household = X[:,population_ix] / X[:,household_ix]
		if self.add_bedrooms_per_room:
			bedrooms_per_room = X[:,bedrooms_ix] / X[:,rooms_ix]
			return np.c_[X,rooms_per_household,population_per_household, bedrooms_per_room]
		else:
			return np.c_[X,rooms_per_household,population_per_household]
attr_adder=CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


#转换流水线   Pipeline
#处理数值的流水线
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer

num_pipeline = Pipeline([
		('imputer',Imputer(strategy="median")),
		('attribs_adder',CombinedAttributesAdder()),
		('std_scaler',StandardScaler()),
	])
housing_num_tr = num_pipeline.fit_transform(housing_num)

#为分类值的自定义转换器
from sklearn.base import BaseEstimator, TransformerMixin
class DataFrameSelector(BaseEstimator,TransformerMixin):
	def __init__(self,attribute_names):
		self.attribute_names = attribute_names
	def fit(self, X, y = None):
		return self
	def transform(self, X):
		return X[self.attribute_names].values
		
#为了解决fit_transform三个参数的错误
from sklearn.base import TransformerMixin
class MyLabelBinarizer(TransformerMixin):
	def __init__(self,*args, **kwargs):
		self.encoder = LabelBinarizer(*args, **kwargs)
	def fit(self, x,y=0):
		self.encoder.fit(x)
		return self
	def transform(self, x, y=0):
		return self.encoder.transform(x)

#分类值上应用LabelBinarizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelBinarizer
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
		('selector',DataFrameSelector(num_attribs)),
		('imputer',Imputer(strategy="median")),
		('attribs_adder',CombinedAttributesAdder()),
		('std_scaler',StandardScaler()),
	])
cat_pipeline = Pipeline([
		('selector',DataFrameSelector(cat_attribs)),
		('label_binarizer',MyLabelBinarizer()),
	])
full_pipeline = FeatureUnion(transformer_list=[
		("num_pipeline",num_pipeline),
		("cat_pipeline",cat_pipeline),
	])

housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared
housing_prepared.shape