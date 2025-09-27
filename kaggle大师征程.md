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



### 18、发现特征信息

**corr ** 会计算皮尔逊相关系数，值的大小在  -1  和  1  之间。1 表示正相关，-1 表示负相关，0表示无线性关系。局限性在于他只能捕捉线性关系，对分类变量或者非线性关系不敏感。

**mutual_info_score** 会计算互信息，用于衡量一个变量对另一个变量提供了多少信息，值大于等于0（没有上限）。它可以捕捉非线性关系，也能处理分类特征，缺点是值得大小没有直观解释性。



### 19、PCA（主成分分析）

pca是一种降维方法，可以压缩特征维度，也可以可视化高维数据，具体见



### 20、heatmap(热力图)

```python
sns.heatmap(train_df[['Time_spent_Alone', 'Social_event_attendance','Going_outside', 'Friends_circle_size','Post_frequency']].corr(),
            annot = True, cmap='coolwarm')
```

data.corr()会计算各个特征之间的皮尔逊系数矩阵。热力图绘图函数中参数annot表示是否显示数值，cmap表示是映射形式。

热力图会反应两个特征之间的线性依赖关系，值在-1和1之间，1表示完全正相关，-1表示完全负相关，0表示不相关。热力图可以快速识别强相关的特征对，对于高度相关的特征，可以考虑去掉其中一个，减小多重共线性，提高泛化能力，对于相关的特征，可以作为独立信息保留。如果相关性很高可能会影响线性模型，反之适合用非线性模型，如随机森林等模型。

高度相关的特征传递的是相似的信息，模型无法确定哪个特征真正有效，为了拟合训练数据，模型会给相关特征分配较大的系数，使得训练数据中微小噪声会导致系数显著变化，从而影响预测。

但要注意的是，相关性只能检测线性关系，非线性关系不会被发现。并且计算相关性会默认忽略缺失值，大量的缺失值会导致结果失真。 样本太小，结果也不准确。如果特征很多，可以先做特征筛选，只保留和目标高相关的特征，再画热力图。

最好只比较数值特征，对于分类特征不要做，否则皮尔逊系数会认为类别值存在顺序。



### 21、相似特征

两个或以上的特征在统计上高度相关，提供了重复信息，对模型贡献不大，反而可能增加噪声，用PCA降维。



### 22、交叉表 （crosstab）

```python
pd.crosstab(
    [train_df['Personality'], train_df["Stage_fear"]],# 这是行索引
    train_df["Drained_after_socializing"]  # 这是列索引
)

```

它会统计不同组合下的数据频数，并以交叉表的形式返回。可用于处理极化数据的缺失值

```python
# 先用交叉表发现规律，之后的策略是尽可能让两列特征保持相同值，若都为缺失值就填充"missing",要我看，如果'missing'较少，直接去掉就好
help_train_df = train_df[['Stage_fear','Drained_after_socializing','Personality']].copy()
help_train_df['Stage_fear'] = help_train_df['Stage_fear'].mask(help_train_df['Stage_fear'].isna() & help_train_df['Drained_after_socializing']
                                            .notna(), help_train_df['Drained_after_socializing'])
help_train_df['Drained_after_socializing'] = help_train_df['Drained_after_socializing'].mask(help_train_df['Drained_after_socializing']
                                            .isna() & help_train_df['Stage_fear'].notna(), help_train_df['Stage_fear'])
help_train_df[cat_cols1]=help_train_df[cat_cols1].fillna('Missing').astype(str)

display(pd.crosstab([help_train_df['Personality'],help_train_df["Stage_fear"]],help_train_df['Drained_after_socializing']))
```



### 23、mask()方法

该方法的逻辑在于，如果condition为真，就填充value，通常condition可以用&、|逻辑组合，比如上一个知识点的代码逻辑

```python
data.mask(condition,value)
```



### 24、MI (互信息)

当某特征有多个较大的互信息时，可以考虑去掉一些，减小信息冗余。



### 25、drop

```python
pd_data.drop("id")
```

pd数据，默认axis=0，表示删除这一行，上面代码就错了，后面应该加上axis=1。



### 26、皮尔逊相关系数

皮尔逊相关系数用于衡量两个变量之间的线性相关程度，值在-1和1之间
$$
r_{xy} = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}
           {\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2}
            \sqrt{\sum_{i=1}^n (y_i - \bar{y})^2}}
$$


### 27、astype()

`.astype()` 只能对 **整个 Series/数组** 用，不能对单个元素用。



### 28、NAN

在 Python 里，`NaN`（缺失值）有个特点：**它不等于任何东西，连它自己都不等于**。要判断一个数是否是缺失值，要使用pd.isna(),pd.notna()



### 29、transform()的用处

transform可以保持变换前后的形状，会广播回原形状，一般用于分组填充缺失值。而直接  .方法（如.median()）则会缩短长度，只会返回每个分组的中位数。



### 30、rename()

```python
data.rename(columns={"a":"b"})
```



### 31、fillna()

```python
all_data.fillna({"Stage_fear":"Missing","Drained_after_socializing":"Missing","match_p":"Missing"},inplace=True)
```

fillna()的这种写法可以直接替换不同列中的缺失值。



### 32、pd.get_dummies()

```python
all_data=pd.get_dummies(all_data,columns=["Stage_fear","Drained_after_socializing","match_p"],
                       prefix=["Stage","Drained","match"])
```

prefix用于控制生成列的名字前缀，比如原本是Stage_fear_yes，现在会变成Stage_yes。



### 33、bool

在sklearn中布尔值会被视为  0/1  数值特征。



### 34、RFECV（递归特征消除+交叉验证）

他是特征选择的一种方法，作用是通过反复训练模型，自动找出对模型性能最优的一组特征，并丢弃不重要的特征，同时用交叉验证判断验证结果。

数据集的特征并不是都有用的，有些特征高度相关而冗余，有些特征完全无关，这些特征会增加计算量，增加过拟合风险，或降低泛化能力。

```python
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

estimator = RandomForestClassifier(random_state=42)

selector = RFECV(
    estimator=estimator,        # 基模型
    step=1,                     # 每次删除多少特征
    cv=StratifiedKFold(5),     # 交叉验证
    scoring='accuracy',        # 评估指标
    n_jobs=-1                   # 多线程加速
)

selector.fit(X, y)

print("最佳特征数:", selector.n_features_) # 最优特征数
print("最佳特征:", X.columns[selector.support_]) # support_是最优特征掩码

```



### 35、互信息和递归特征 ==对比==

互信息用于衡量每个特征和目标之间的**独立信息**（信息增益），但缺点是单变量分析，每次只看一个信息，忽略多种信息的混合作用。RFECV可以找出 **最优特征子集**，使得模型的性能最好。这是一种多变量分析方法，特征的重要性是结合多种特征一起评估的，会直接优化模型性能。

特征具有互补性，有可能某特征的信息增益较小，但是与其他特征组合时会显著提高信息增益。打个比方，最终考试有两种指标：时间和平时分数，平时分数的信息增益大，时间的信息增益小，但是不能说时间不重要，把二者结合起来看更优。



### 36、**分组填充思想**



```python
all_data=fill_missing_by_quantile_group(all_data,"Friends_circle_size","Post_frequency")
all_data["Post_frequency"].fillna(all_data["Post_frequency"].median(),inplace=True)
```















## 2、项目心得

### 1、**预测外向还是内向**   竞赛

#### 1、（大师一）

我自己做了一个特征工程，但是似乎作用没有那么大。我将向大师学习,下面都是他的思路，我将如实记录下来。

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

#### 2、（大师二）

之后，我下载了另一个笔记本，目前觉得还行（大师二），下面是具体思路。

* 首先进行了数据检查。检查训练集和测试集的数据范围和缺失值、类别数，包括 info() 、describe() 、nunique（）手段。之后检查了二者的数据分布，主要是看是否出现数据漂移（测试集和训练集分布不一致，使得模型在测试集上预测困难）的情况。
* 之后，分别绘制了热力图（用于查看各个特征之间的线性相关性）和计数条形图（sns.countplot会自动统计各个类别数，并用条形图显示）同时打印了标签各个类别的比例。并且，作者也绘制了堆叠图，用于显示各个列中标签的数量

```python
# 漂移检查
for col in test_df.columns:
    if col != 'id' and test_df[col].dtype in[np.int64,np.float64]:
        sns.kdeplot(test_df[col], label='test', fill=True)
        sns.kdeplot(train_df[col], label='train', fill=True)
        plt.legend()
        plt.show()
        
# 堆叠图 train_df.groupby([col,'Personality']).size().unstack().plot(kind='bar', stacked=True, title=col)
```

* 卡尔检验

```python
columns = ['Time_spent_Alone_MISS','Stage_fear_MISS', 'Social_event_attendance_MISS', 'Going_outside_MISS','Drained_after_socializing_MISS', 
           'Friends_circle_size_MISS','Post_frequency_MISS']
for col in columns:
    train_df.groupby([col,'Personality']).size().unstack().plot(kind='bar', stacked=True, title=col)
    result = pd.crosstab(train_df[col],train_df['Personality'], normalize='index')*100
    chi2, p, _, _ = chi2_contingency(result)
    # display(result)
    print(f"Chi2 = {chi2:.3f}, p-value = {p:.4f} Column: {col} ")
```

* 在填充缺失值方面，作者使用了分箱填充缺失值的策略。对指定列的缺失值进行了等频分箱+分组中位数填充处理，相当于用相似区间的中位数替代缺失值，提高填充的准确性。作者先用分箱方法填充缺失值，之后用全局均值填充可能因源列有缺失值未被填充的缺失值。

  那么什么时候用中位数填充，什么时候用平均值填充呢。当特征数据存在明显的分段特性（不同的区间的值明显有差异时，如存款，城市gdp，年龄相关的特征等）时，采用分箱可以有效处理偏态分布或有极端值的特征，这时采用各个区间的中位数填充较为合适，并且分箱后特征类别变少，模型更稳定。

  当缺失值完全随机，与其他变量无关时，用全局均值也没问题，当数据规模足够大时，均值可以很好地代表总体水平，或者当特征和目标关系很弱（分箱意义不大时），用均值反而更简洁。



#### 3、学习成果

成果见[笔记本链接（本地）](F:\anaconda_projects\kaggle\Introverts_and_Extroverts\very_import Fight_again.ipynb)（kaggle 筛选模型最好一个高分数，一个次一点的分数）





