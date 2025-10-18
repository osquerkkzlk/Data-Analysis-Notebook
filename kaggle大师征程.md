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

只有series数据可以用map方法，DataFrame数据没有map方法，此时只能用apply，如 data.apply(lambda x:x.map(...))需要注意的是，map方法是主元素操作，比如data.map(frac)表示对该列中的每一个元素操作，frac函数接收到的参数实际上是单个元素值。

series数据和dataframe数据都有apply方法，一个是取出每个元素操作，一个是取出每个列操作。apply的作用机理是分别传入数据的每一个列(请注意，是整列数据而不是列名)作为x，然后应用函数变换。



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

重命名列，columns参数不要忘记加。



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

重命名列，columns参数不要忘记加。

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



但是有一些需要你注意的地方。首先，并不是说你朝数据中添加的特征越多，结果就至少不变差，更可能变好，这是错误的想法。实际上，RFECV的工作机制使得它并不是独立评估每个特征，而是在特征组合下评估性能，因此，垃圾特征不仅会“占用资源”，还可能误导模型的结构和重要性排序。

另一方面，RFECV 常常会因为前期选出“垃圾特征”（单个来看不重要，但是组合起来可能会干扰）而判断错误，导致后面的迭代路径发生偏离，也就是“路径依赖”。另一方面，有些特征不是垃圾，可能因为和一些特征高度相关，使得迭代器会选取它（分裂增益高），进而原本的特征可能被替代或者掩埋，从而改变了特征组合的相对重要性，影响了递归路径。

所以，在递归特征之前，可以先对特征进行初步筛选，比如互信息。









### 35、互信息和递归特征 ==对比==

互信息用于衡量每个特征和目标之间的**独立信息**（信息增益），但缺点是单变量分析，每次只看一个信息，忽略多种信息的混合作用。RFECV可以找出 **最优特征子集**，使得模型的性能最好。这是一种多变量分析方法，特征的重要性是结合多种特征一起评估的，会直接优化模型性能。

特征具有互补性，有可能某特征的信息增益较小，但是与其他特征组合时会显著提高信息增益。打个比方，最终考试有两种指标：时间和平时分数，平时分数的信息增益大，时间的信息增益小，但是不能说时间不重要，把二者结合起来看更优。



### 36、**分组填充思想**



```python
all_data=fill_missing_by_quantile_group(all_data,"Friends_circle_size","Post_frequency")
all_data["Post_frequency"].fillna(all_data["Post_frequency"].median(),inplace=True)
```



### 37、corrwith()

```python
data[features].corrwith(F1)
```

代码表示计算取出  data  中的特征列与特征  F1  之间的皮尔逊相关系数，只能计算线性关系的强弱。



### 38、目标编码

平均编码（bin计数），利用某特征信息，对数据进行分组聚合，对每组分别应用编码。

```python
autos["make_encoded"] = autos.groupby("make")["price"].transform("mean")

```

但是缺点是当某类别数较少时，平均值会不稳定，引入噪声。因此我们引入了平滑参数m，公式如下。其中m表示平滑参数，nc表示当前类别数，mean（c）表示该类别的平均目标值，global mean表示全局平均值。m越大编码值越偏向于全局平均值，更保守。m越小，结果越接近类别平均值，越容易受到噪声影响。
$$
\text{encoding}(c) = \frac{n_c}{n_c + m} \cdot \text{mean}(c) + \frac{m}{n_c + m} \cdot \text{global mean}
$$
下面是目标编码的使用场景高基数特征，指某个分类特征有很多的类别，比如邮政编码，用户id等，此时如果使用独热编码，数据特征会变得庞大，如果使用标签编码，会产生顺序关系。如果  id  只是唯一标识符就直接去掉，如果id有意义（用于ID，设备ID，时间ID等），那么我们选择做特征工程而不直接使用。另一个场景是，根据经验可以知道强相关的特征，如汽车品牌和价格。

```python
encoder = MEstimateEncoder(cols=["Neighborhood"],m=1.)  # 告诉编码器要对该列进行编码
encoder.fit(X_encode,y_encode)

# Encode the training split
X_train = encoder.transform(X_pretrain, y_train)

```



### 39、index_col

```python
df = pd.read_csv("data.csv", index_col=0)

```

读取csv文件时，有参数 index_col，该参数会指定选取哪一列当作默认索引，这样就不会创建额外的索引。



### 40、DataFrame 本质

DataFrame 本质上是一个二维表格，有行和列。在创建该表格时，pandas需要知道有多少行和每列的数据是什么。如果传入标量，pandas不知道应该应该创建多少行，因为标量只是单个值，不是列表等可迭代对象，此时我们应该传入index参数以明确告诉它应该创建几行，比如index=[0]表示只有一行。



### 41、pd数据访问列名

使用 "." 进行访问数据，如 data.columns_name 时，列名必须符合python变量的命名规则（不能有空格、标点、数字开头），而且如果列名和 DataFrame的方法名冲突时该方法就会失效。

使用data[column_name]的方法中规中矩，值得信赖。



### 42、set_index()

data.set_index(name)   会把数据中的某一列视为新的索引，并返回一个i虚拟的DataFrame数据，不过也有inplace参数来控制是否就地修改。



### 43、访问数据

pandas 访问数据时，可以使用loc和iloc来访问。其中，loc[]可以接收布尔类型值，表示选取布尔值为真的行。



### 44、isin()

会返回一个布尔值，表示数据是否在列表中,如data["ID"].isin(["124"])



### 45、isna()和notna()

则会返回数据是否为缺失值的布尔矩阵，你应该能看懂。



### 46、apply

该方法会对数据中的每一列（默认）或者每一行数据调用frac方法。data.apply(frac，axis=0)，当axis=0表示对每一列处理，axis=1表示对每一行处理。不论是map、还是apply，他们都不会修改原始的表格数据，反而会返回一个新的表格数据。



### 47、squeeze()

data.squeeze()可以把一行或者一列的dataframe数据展开成series数据。



### 48、idxmax()

pd的series数据有方法idxmax()可以返回最大值对应的标签索引，可以直接用loc来访问对应行。



### 49、链式比较

这是python特有的特性，例如 9<x<3。



### 50、series

该类型数据也是有列属性的，毕竟所谓dataframe就是由一个个series堆叠起来的。如果说我们想对某一列做相同的变换，就用 axis=0,如果我们想筛选有特殊条件的行，就用 axis=1



### 51、agg（）

agg（）方法允许你在dataframe数据上运行多种函数，一般还是用在分组聚合上面。

```python
reviews.groupby(['country']).price.agg([len, min, max])

```



### 52、reset_index()

data.reset_index()会把数据恢复到原来的索引，一般用于分组数据处理完后恢复到原形状。



### 53、groupby()

data.groupby(col)对数据进行分组聚合之后，会返回一个按照索引col排列的dataframe数据（不是按照索引在数据集中出现的顺序，而是按照索引的大小）。



### 54、sort_index（）

该方法会按照索引进行排序，可以解决  groupby（）出现的问题。

```python
countries_reviewed.sort_index()
```

```python
# 如果两个数据有相同索引，并且都表示同一事物，那么join！

powerlifting_combined = powerlifting_meets.set_index("MeetID").join(powerlifting_competitors.set_index("MeetID"))

```



### 55、sort_values（）

该方法既依据一个列可以对数据进行排序，也可以依据多个列进行排序，此时优先级是从前到后。==并且主要参数by是dataframe的参数，series数据没有该参数。==

```python
countries_reviewed.sort_values(by=['country', 'len'])

```



### 56、size（）与count（）

data.groupby(target_col)之后，如果使用  count（）方法，他会对dataframe中的每列都做计数，即统计非缺失值的数量。如果使用 size（）方法，则会单纯的统计每一组有多行，返回一个  series  数据。



### 57、dtype和dtypes属性

你可以使用  dtype  属性获取特定列的值的类型。而dtypes属性则会获取所有列的属性。

```python
data.target.dtype
data.dtypes
```



### 58、缺失值

NAN 即  not a number ，其实更多的是指 not value ，表示值缺失，不仅可以表示数值的缺失，也可以表示非数值的缺失。

替换缺失值，最简单的方式就是 data.fillna("Unknown") 。此外，还可以通过replace（）方法便捷地替换缺失值,即  data.replace（A，B）,用B替代A。



### 59、astype（）

该方法可以快速转变数据类型，需要注意的是他不是就地转换。



### 60、value_counts（）

data.value_counts（）本身就有参数 ascending用于控制升序还是降序。



### 61、rename（）

```python
reviews.rename(columns={'points': 'score'})
```

该函数用于将列名更改为我们满意的名字。

```python
reviews.rename(index={0: 'firstEntry', 1: 'secondEntry'})
```

此外，你也可以通过  index  参数对行名进行重命名。



### 62、rename_axis（）

```python
reviews.rename_axis("wines", axis='rows').rename_axis("fields", axis='columns')
```

该方法会重命名索引的标签名字



### 63、拼接数据

第一种方法是 concat（）方法，他可以对数据进行横向拼接和纵向拼接。

```python
pd.concat((data1,data2),axis)
```

第二种是 join（）方法，更加灵活。join（）大默认方法是左拼接，以左侧的索引为基准，参数how可以指定拼接的方式。参数  lsuffix  表示原数据如果与新数据列名重复，就在列名后加上该字符串表示区分， rshuffix表示新数据如果与原数据列名重复，就在列名后加上该字符串表示区分。

| how       | 含义                                                         |
| --------- | ------------------------------------------------------------ |
| `"left"`  | 左连接（默认），保留左 DataFrame 的所有索引，右 DataFrame 的数据可能补 NaN |
| `"right"` | 右连接，保留右 DataFrame 的所有索引                          |
| `"outer"` | 外连接，保留两边所有索引，缺失部分补 NaN                     |
| `"inner"` | 内连接，只保留两边都匹配的索引                               |

```python
left.join(right, lsuffix='_CAN', rsuffix='_UK')
```



### 64、np.product()

np.product(data.shape)会计算元素的乘积。



### 65、处理缺失值

首先是观察数据，并分析数据缺失的原因：此值丢失是因为它没有被记录还是因为它不存在。如果一些值因为不存在而丢失，那么尝试猜测它可能是没有意义的。如果某个值因为没有记录而丢失，那么可以尝试根据该列和行中的其他值来猜测他可能是什么，这就是插补技术。

data.dropna（axis=1）表示删除有缺失值的列，==该axis参数是反直觉的==，其他的axis参数则是正常的。

data.fillna()直接填充缺失值

data.fillna(method='bfill', axis=0).fillna(0) 表示用同一列后面一个非缺失值填补，如果该列权威缺失值就用0来填充。



### 66、缩放和规范化

**缩放**的核心思想是改变数值范围，但不改变分布的形状，它会把所有特征压缩到【0，1】区间或者把特征标准化成均值为0，方差为1。

**规范化**改变数据分布，使其形状更接近正态分布。



### 67、解析日期

但最常见的是 `%d` 表示日，`%m` 表示月，%`y` 表示两位数年份，%`Y` 表示四位数年份。data.to_datetime(data.time  ,  format="%d/%m/%y")。当单列中有多种日期时，可以考虑自动推断日期，代码如下：data.to_datetime(data.time,infer_datetime_format=True)。

日期数据数据有日期访问器 dt，使用如下：data.time.dt.day（访问天）



### 68、pd.read_csv（）

```python
fifa_data = pd.read_csv(fifa_filepath, index_col="Date", parse_dates=True)

```

改代码会把日期列直接作为索引，并通过设置  parse_dates  参数自动解析时间序列。



### 69、seaborn 画图（提高）

#### 1、 折线图  sns.lineplot()

有两种写法，一种是直接传入数据，此时默认把索引当作横轴，如果是一维数据则会生成从0开始的索引。另一种是传入 x 和  y（x指定横轴，y指定纵轴，data=df表示原始表格数据）。

#### 2、条形图  sns.barplot（）

```python
sns.barplot(x=flight_data.index, y=flight_data['NK'])
# 一个技巧就是，你可以通过指定x和y来绘制水平条形图和竖直条形图
```

#### 3、热力图  sns.heatmap（）

```python
sns.heatmap(data=flight_data, annot=True)  # 参数annot表示显示数值 
```

#### 4、散点图  sns.scatterplot（）和  sns.regplot（） 和  sns.lmplot（）和  sns.swarmplot（）

```python
# 干净的散点图
sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'])

# 带回归线的散点图，用于显示数据趋势
sns.regplot(x=insurance_data['bmi'], y=insurance_data['charges'])

# 带颜色编码的散点图
sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'], hue=insurance_data['smoker'])

# 带回归线的颜色编码散点图(每个列别都会拟合一条回归线)
sns.lmplot(x="bmi", y="charges", hue="smoker", data=insurance_data)

#分类散点图
sns.swarmplot(x=insurance_data['smoker'],
              y=insurance_data['charges'])
```

#### 5、分布图  sns.histplot（）和 sns.kdeplot（）

```python
# 直方图
sns.histplot(iris_data['Petal Length (cm)'])

# 核密度估计(平滑直方图) 只能对连续数值使用。
sns.kdeplot()

# 颜色编码绘图
sns.histplot(data=iris_data, x='Petal Length (cm)', hue='Species')

sns.kdeplot(data=iris_data, x='Petal Length (cm)', hue='Species', shade=True)

```



### 70、相关系数

计算相关系数的前提是数据连续，假设是数据特征间存在线性关系



### 71、make_column_transform（）

该函数可以对特定列应用特定的变换，但是数据的列名顺序与应用的顺序相关，==因此要特别注意==。



### 72、OnehotEncoder

`OneHotEncoder` 不会自动判断“这个列是不是分类变量”。它只要看到是数值，就直接当作**类别取值**。



### 73、分箱操作（pd.cut和pd.qcut）

首先会自动排序，然后划分区间，这是隐式完成的。

pd.qcut() 自动分箱技术，使得每个区间的样本数大致相同。pd.qcut(data,q,labels)

pd.cut()手动指定分箱区间，可以手动指定划分区间，pd.qcut(data,bins,q,labels)



### 74、Stacking  （堆叠集成）

把多个基础模型（即 estimators）的预测结果作为一个特征输入到一个“元模型”中，元模型来学习如何最优地融合这些预测。有属性 estimator 和estimator_，final_estimator和final_estimator _，其中没有下划线则表示 在定义阶段传入的原始模型或原始模型列表，而有下划线则表示训练后的模型对象或模型对象列表。

若基础模型为复杂非线性模型，元模型采用线性回归或者岭回归；若基础模型较简单，则元模型可用非线性模型（如随机森林或GBDT）。



### 75、MLP模型

sklearn提供了MLPRegressor的接口，一般只需要配置hidden_layer_sizes=（a,b,c,..）也就是每一层的神经元数。



### 76、category  类型

在pandas中，category 是一种专门为**离散特征**设计的数据类型，它不是简单的字符串，而是用整数编码加映射表来存储的。把特征类型转换为  catrgory  后，xgboost等模型就可以自动识别并处理，不需要再手动做编码处理，减小预处理复杂度。同时，pandas内部进行了优化，使得该类型内存消耗小（只需要保存映射表和行索引标号），加速训练过程。category  的映射是无序的，虽然它会把离散特征映射为0、1、2...但是并不会引入顺序意义，pandas也会阻止索引进行大小比较。



### 77、np张量

张量相加是按位相加的，这一点和列表是不同的。



### 78、pd.Categorical（）

该方法可以对指定列进行类别编码，并允许指定编码顺序，其中  data['risk']是一个类别对象，还不是一个值，所以我们通过访问类别属性的接口  cat.codes  得到编码值。参数  ordered （默认是False，也就是类别无序，只是标识不同类别）设置为True  的作用是表明类别是有序的

```python
import pandas as pd

data = pd.DataFrame({
    'risk': ['高', '低', '中', '高', '低']
})

# 明确类别顺序
data['risk'] = pd.Categorical(
    data['risk'],
    categories=['低', '中', '高'],
    ordered=True
)

# 转换为对应的整数编码
data['risk_code'] = data['risk'].cat.codes
print(data)
```



### 79、基于外部统计表的目标均值映射

基于外部统计表的目标均值映射，即通过外部知识进行统计构造特征。代码如下

```python
def external_mean_encoding(base_df, ref_df, target_col, cols):
    base_df = base_df.copy()
    for col in cols:
        mean_map = ref_df.groupby(col)[target_col].mean()
        base_df[f"org_{col}"] = base_df[col].map(mean_map)
        
    return base_df
```



### 80、合成数据集的误区

合成数据集的方法中的一些特征，在经过深度学习处理之后可能不再起作用，这时我们依旧按照生成数据的方法取分析，反而可能会误导模型，所以我们应该仔细分析哪些是有用的哪些变得无用，通过皮尔逊相关系数、互信息或者xgboost中的重要性排名来考虑。



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





### 2、道路风险预测  竞赛

1、在分析数据特征时，直接用全部的训练数据，没必要划分后用训练集。

