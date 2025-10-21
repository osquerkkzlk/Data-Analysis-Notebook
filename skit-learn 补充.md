# skit-learn 补充



## 1、混淆矩阵基础

在二分类任务中（例如判断数字是否为 5，y_train_5 是 True 或 False），混淆矩阵用来评估模型的预测结果。它包含以下四个元素：

1. TP（True Positive，真正例）

   - 实际为正类（True，即数字 5），模型也预测为正类（True）。
   - 示例：一个样本的真实标签是 5，模型预测也是 5。

2. TN（True Negative，真负例）

   - 实际为负类（False，即非 5），模型也预测为负类（False）。
   - 示例：一个样本的真实标签是 3，模型预测不是 5。

3. FP（False Positive，假正例）

   - 实际为负类（False，即非 5），模型错误预测为正类（True）。
   - 示例：一个样本的真实标签是 7，模型预测是 5。

4. FN（False Negative，假负例）

   - 实际为正类（True，即 5），模型错误预测为负类（False）。
   - 示例：一个样本的真实标签是 5，模型预测不是 5。

## 2、常见参数

1. 学习模型参数（fit）。
2. 预测新数据（predict）。
3. 评估性能（score 或 cross_val_score）。

## 3、常见知识

#### 1、进行数据集的切分

```
from sklearn.model_selection import train_test_split
```

#### 2、进行标签二值化

以一对多的方式对标签进行二值化。在学习时，这只包括为每个类学习一个回归器或二元分类器。在此过程中，需要将多类标签转换为二进制标签（属于或不属于该类）。`LabelBinarizer` 使用 transform 方法使此过程变得简单。在预测时，人们分配相应模型为其提供最大置信度的类

```python
from sklearn.preprocessing import LabelBinarizer
import numpy as np
lb.fit(np.array([[0, 1, 1], [1, 0, 0]]))
lb.classes_
lb.transform([0, 1, 2, 1])
array([[1, 0, 0],
       [0, 1, 0],
       [0, 0, 1],
       [0, 1, 0]])
```

==补充：one-hot编码==

One-hot 编码是一种将分类变量转换为机器学习算法可用的数值形式的技术。它的基本原理是将每个类别表示为一个只包含一个1，其余位置都是0的二进制向量。

例如，如果我们有一个"颜色"特征，包含"红"、"绿"、"蓝"三种可能的值，使用one-hot编码后会变成：

- 红色：[1, 0, 0]
- 绿色：[0, 1, 0]
- 蓝色：[0, 0, 1]

这种编码的主要优点包括：

1. 避免了赋予类别数值顺序关系（例如红=1，绿=2，蓝=3可能暗示了不存在的顺序关系）
2. 适用于大多数机器学习算法
3. 保持了分类数据的独立性

不过，这种方法也有一些缺点：

- 会增加特征的维度，尤其是当类别很多时
- 可能导致稀疏矩阵
- 存在"虚拟变量陷阱"的问题，即如果包含所有编码特征，会导致多重共线性

在实际应用中，Python的scikit-learn库和pandas提供了简便的函数来实现one-hot编码：

```python
# 使用pandas
import pandas as pd
df = pd.DataFrame({'颜色': ['红', '绿', '蓝', '红']})
pd.get_dummies(df)

# 使用scikit-learn
from sklearn.preprocessing import OneHotEncoder
import numpy as np
encoder = OneHotEncoder()
colors = np.array(['红', '绿', '蓝', '红']).reshape(-1, 1)
encoder.fit_transform(colors).toarray()
```



## 4、KNN（K-Nearest Neighbors，K 近邻）

- **算法类型**：监督学习（classification/regression）

- **用途**：分类或回归，比如判断某个样本属于哪一类

- **核心思想**：

  > 新样本 → 找到最近的 K 个已知标签的样本 → 投票或加权平均 → 得到预测值。

- **典型场景**：

  - 判断一个用户是“高收入”还是“低收入”
  - 根据历史销量预测商品销量

- **关键点**：需要有**带标签的数据**。

**简单例子**

```
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
```



## 5、.model_selection.train_test_split

```python
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data_numeric, target, random_state=42, test_size=0.25
)
```

自动划分训练集和测试集



## 6、LogisticRegression （逻辑回归）

用于进行分类任务，尤其是二分类

```
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
```

## 7、DummyClassifier（基准分类器）

`DummyClassifier` 是 **Scikit-learn** 提供的一个“基准分类器”，主要用于 **基线（baseline）对比**，帮助我们判断一个真正的模型是否“真的有价值”

```python
# 创建 DummyClassifier
dummy = DummyClassifier(strategy="most_frequent")  # 总是预测出现最多的类别
dummy.fit(X_train, y_train)

# 预测
y_pred = dummy.predict(X_test)
print("预测结果:", y_pred)
print("准确率:", accuracy_score(y_test, y_pred))

```



## 8、StandardScaler（特征缩放）

用于把特征转换为 **均值为 0、标准差为 1** 的分布，这个过程叫 **标准化（Standardization）**。

```python
scaler = StandardScaler()

# 2. 计算均值、标准差并转换
data_scaled = scaler.fit_transform(data)

print("均值:", scaler.mean_)       # 每列的均值
print("标准差:", scaler.scale_)   # 每列的标准差
print("标准化结果:\n", data_scaled)
```

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 拆分训练和测试
X_train, X_test, y_train, y_test = train_test_split(data, [0,1,1,0], random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 在训练集上 fit + transform
X_test_scaled  = scaler.transform(X_test)       # 在测试集上仅 transform

```

要用训练集拟合下的标准差和均值对测试集进行标准化。



## 9、make_pipeline

自动生成管道，进行连续处理。并且内部自动分配名字，`scikit-learn` 会自动根据类名生成，如果类名重复，会自动加 `_1`, `_2` 之类的后缀。

```py
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(
    StandardScaler(), 
    LogisticRegression()
)
model.fit(data_train, target_train)
predicted_target = model.predict(data_test)
score = model.score(data_test, target_test)

```

模型会自动对测试数据进行标准化。



## 10、cross_validate（模型评估）

仅用于模型评估，最终会返回一个字典

```python
cv_result = cross_validate(model, data_numeric, target, cv=5)# 5折

```

- （i）在每个折叠的训练数据上训练模型的时间， `fit_time`

- （ii） 使用模型对每个折叠的测试数据进行预测的时间， `score_time`

- （iii） 每个折叠的测试数据的默认分数，`test_score`。

- 可以通过在 `cross_validate` 中传递选项 `return_estimator=True` 来检索每个拆分/折叠的这些拟合模型。

  ```python
  cv_results = cross_validate(regressor, data, target, return_estimator=True)
  
  ```

- 默认情况下，`cross_validate` **只计算验证集（test fold）的分数**；`return_train_score=True` 是 **`cross_validate`** 函数特有的一个参数，用来控制是否同时返回 **训练集上的分数**。

## 11、make_column_selector（根据数据类型选择列）

```
from sklearn.compose import make_column_selector as selector

categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(data)
categorical_columns
```



## 12、make_column_transformer

可以对不同的列做不同的处理，最后合并为一个大矩阵。

```python
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import OneHotEncoder, StandardScaler

categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
numerical_preprocessor = StandardScaler()
preprocessor = make_column_transformer(
    (OneHotEncoder(handle_unknown="ignore"), categorical_columns),
    (StandardScaler(), numerical_columns)
)

```



```python
preprocessor = make_column_transformer(
    (categorical_preprocessor, categorical_columns),
    remainder="passthrough",
)

```

正常情况下，未被指定的列会被删除，而remainder则可以保留下来



## 13、StratifiedKFold（分层Kfold）

用于**分类任务**的交叉验证分割器，**每个折里各类别的比例尽量与整体一致**。这样评估更**稳定**、对**类别不平衡**更友好，避免某些折里几乎没正样本/负样本导致指标失真。



## 14、ValidationCurveDisplay（验证损失曲线显示）

```python
from sklearn.model_selection import ValidationCurveDisplay,validation_curve

train_scores,test_scores=validation_curve(
    pipeline_model,
    data,target,
    param_name="kneighborsclassifier__n_neighbors",
    param_range=param_range,
    cv=ShuffleSplit(n_splits=5),
    scoring="balanced_accuracy"
)
disp = ValidationCurveDisplay(
    param_range=param_range,
    train_scores=train_scores,
    test_scores=test_scores,
    param_name="kneighborsclassifier__n_neighbors",
    score_name="Balanced Accuracy",
)
disp.plot()
```

param_name表示将要修改的参数名称，param_range表示修改的参数的值，negate_score表示对值取反。

validation_curve也可以用于找到某个参数的最佳值，因此可以进行粗略搜索，可以与 GridSearchCV做配合。

## 15、GridSearchCV（精确搜索）

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "classifier__learning_rate": (0.01, 0.1, 1, 10),  # 4 possible values
    "classifier__max_leaf_nodes": (3, 10, 30),  # 3 possible values
}  # 12 unique combinations
model_grid_search = GridSearchCV(model, param_grid=param_grid, n_jobs=2, cv=2)
model_grid_search.fit(data_train, target_train)
model_grid_search.best_params_  # 最佳参数组合
```

