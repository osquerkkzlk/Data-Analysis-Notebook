# kaggle大师征程

​																			【日期】2025-09-24

## 1、知识补充

### 1、loc访问

pd数据也可以通过布尔掩码，得到符合条件的行。



### 2、==pd数据的遍历==

pd数据遍历，比如for循环遍历，默认是使用其列名来遍历的，所以下面的效果是一样的，

```python
for col in pd_data:
for col in pd_data.columns
```



### 3、==视图和副本==

| 写法                               | 返回的是什么              | 可否直接就地修改 |
| ---------------------------------- | ------------------------- | ---------------- |
| `df[col]`                          | **通常是视图**（Copy）    | ❌ 不一定生效     |
| `df.loc[row_indexer, col_indexer]` | **通常是视图**（View）    | ✅ 可以直接修改   |
| `df.loc[row_indexer][col_indexer]` | **链式索引 → 常返回副本** | ❌ 很可能不生效   |
| `df.iloc[...]`                     | 类似 `.loc`               | ✅ 可以直接修改   |

要注意的是，pd数据行切片或列切片，基本上都是返回副本。但是呢，如果你直接对某列操作则会返回视图。





### 4、水平条形图（显示 MI ）

```python
def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores)
```



### 5、map（）

只有series数据可以用map方法，DataFrame数据没有map方法，此时只能用apply，如 data.apply(lambda x:x.map(...))。apply的作用机理是分别传入数据的每一个列(请注意，是整列数据而不是列名)作为x，然后应用函数变换。



### 6、mode陷阱

data.mode（）会返回一个dataframe格式的数据，并且可能有多个众数，如果你也要用众数填充缺失值的化，你可以直接.iloc[0]也就是返回第一个众数即可。



### 7、正确处理训练数据和测试数据

```python
from sklearn.impute import KNNImputer

def deal_nan(data, data_labels, standard_data, standard_labels):
    data = data.copy()
    # 数值列使用整体均值填充（避免标签依赖）
    data[numeric_columns] = data[numeric_columns].fillna(standard_data[numeric_columns].mean())
    # 非数值列使用整体众数填充
    for col in object_columns:
        mode_value = standard_data[col].mode().iloc[0] if not standard_data[col].mode().empty else 'Yes'
        data[col] = data[col].fillna(mode_value)
    # 映射
    data[object_columns] = data[object_columns].apply(lambda x: x.map({"Yes": 0, "No": 1}))
    # 可选：使用KNN插值（无标签依赖）
    imputer = KNNImputer(n_neighbors=5)
    data[numeric_columns] = imputer.fit_transform(data[numeric_columns])
    return data, data_labels.map({"Extrovert": 0, "Introvert": 1})
```

```python
def deal_nan_test(data, train_data):
    data = data.copy()
    data[numeric_columns] = data[numeric_columns].fillna(train_data[numeric_columns].mean())
    for col in object_columns:
        mode_value = train_data[col].mode().iloc[0] if not train_data[col].mode().empty else 'Yes'
        data[col] = data[col].fillna(mode_value)
    data[object_columns] = data[object_columns].apply(lambda col: col.map({"Yes": 0, "No": 1}))
    return data
```



### 8、drop_duplicates（）

data.drop_duplicates() 会删除完全相同的行



### 9、rename

```python
data.rename(columns={'Personality': 'match_p'})
```

重命名列



### 10、聚类技术（作为预处理）

聚类技术，比如k均值技术往往作为特征提取的手段之一，你可以把簇标签、簇到中心距离作为心得特征加入到数据中，减少噪声。



### 11、round（）

pd数据，data.round(2)表示保留两位小数。



### 12、tolist()

data.tolist()会把series数据直接转换为列表，需要注意的是他只能转换一维数据。



### 13、图标绘制标题

suptitle（）实际上给整个画布写标题，而title（）是给单个子图写标题。



### 14、迭代插值

插值时，训练数据和测试数据要分开插值，一般用训练数据拟合，用测试数据变换。同时，插补数据需要用到数据的全部特征，（当然标签可以丢弃，这无关紧要），但是数值特征和分类特征需要分别插值，因为目标不同。插值的原理在于先用均值策略或者其他的策略填充缺失值，然后用迭代器去捕捉数据的不同特征之间的关系，最后填补缺失值。



### 15、display

jupyter环境中直接使用，可用于美观地显示内容，相当于单元格运行。因为你直接用print会打印表格，用display则会展示原本的样子。

```
display(data.head())
```



### 16、duplicated（）

应用该方法会返回一个布尔类型的Series数据，如果第一次出现标记为False，此后均标记为True，如果此时应用  sum  方法，则会求出该数据集重复数据数

```
data.duplicated().sum()
```



### 17、缺失值显示

对于0/1数据来讲，平均值就是数据1所占的比例。

```python
print("\nMissing values  in TEST data:\n",test_df.isna().mean().apply(lambda x: f"{x:.2%}"))
```



### 18、









## 2、项目心得

### 1、**预测外向还是内向**   竞赛

我自己做了一个特征工程，但是似乎作用没有那么大。我将向大师学习,下面都是他的思路，我将如实记录下来。（大师一）

* 首先，导入数据。

* 探索性数据数据分析（EDA）. 你需要检查数据集中是否存在异常点或离群点。接着，检查数据集是否平衡，最后处理缺失值和分析数据集各个特征的意义。大师打印了训练数据、测试数据、外部附加数据各列的缺失值及其所占的比例。

  ```python
  cols = train.columns
  vals = train.isna().sum()
  per = train.isna().sum()*100/train.shape[0]
  print("\033[1mPercentage of Missing Values\033[0m".center(50))
  print(pd.DataFrame({"Count": vals, 'Percentage': per.round(2)}))
  ```

  在缺失值插值上面，它使用了  IterativeImputer（迭代插值），用其他特征预测缺失值，迭代多轮更新预测，直至收敛。这种方法考虑了其他特征的相关性，会生成合理的填充值，缺点是计算量大，需要数据特征之间存在一定的相关性。（是MICE的具体实现）。

* 之后，用多种模型训练即可。



之后，我下载了另一个笔记本，目前觉得还行（大师二），下面是具体思路。

* 首先进行了数据检查。检查训练集和测试集的数据范围和缺失值、类别数，包括 info() 、describe() 、nunique（）手段。之后检查了二者的数据分布，主要是看是否出现数据漂移（测试集和训练集分布不一致，使得模型在测试集上预测困难）的情况











