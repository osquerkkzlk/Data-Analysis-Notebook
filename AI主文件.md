# 1、numpy包

------

## 1. **导入 NumPy**

```python
import numpy as np
```

------

## 2. **创建数组**

### (1) 从列表创建数组

```python
arr = np.array([1, 2, 3, 4, 5])
print(arr)  # [1 2 3 4 5]
```

### (2) 创建多维数组

```python
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print(arr2d)
```

### (3) 生成特殊数组

```python
np.zeros((2, 3))    # 创建 2×3 全零数组
np.ones((3, 3))     # 创建 3×3 全一数组
np.full((2, 2), 7)  # 创建 2×2 值为 7 的数组
np.eye(4)           # 创建 4×4 单位矩阵
np.empty((3, 3))    # 创建未初始化的 3×3 数组（值不确定）
```

### (4) 生成等差数列

```python
np.arange(0, 10, 2)  # [0 2 4 6 8]
np.linspace(0, 1, 5)  # 生成 5 个等间距的点 [0.   0.25 0.5  0.75 1.  ]
```

### (5) 生成随机数组

```python
np.random.rand(2, 3)        # 生成 2×3 的 0-1 之间均匀分布的随机数
np.random.randn(3, 3)       # 生成 3×3 的标准正态分布随机数
np.random.randint(1, 10, (2, 2))  # 生成 1 到 10 之间的随机整数数组
```

------

## 3. **数组属性**

```python
arr.shape  # 获取数组形状，如 (3, 4)
arr.size   # 获取数组元素总数
arr.dtype  # 获取数组数据类型
arr.ndim   # 获取数组维度
```

------

## 4. **数组运算**

### (1) 数学运算

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(a + b)  # [5 7 9]
print(a - b)  # [-3 -3 -3]
print(a * b)  # [ 4 10 18]
print(a / b)  # [0.25 0.4  0.5 ]
print(np.exp(a))  # e^a
print(np.sqrt(a))  # 开平方
print(np.log(a))  # 取对数
```

### (2) 聚合函数

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(np.sum(arr))  # 求和
print(np.mean(arr))  # 求均值
print(np.max(arr))  # 最大值
print(np.min(arr))  # 最小值
print(np.std(arr))  # 标准差
print(np.var(arr))  # 方差
print(np.(arr))  # 最大值索引
print(np.argmin(arr))  # 最小值索引
print(arr.mean()) #计算平均值
print(arr.median()) #计算中位数
print(arr.cumsum()) #计算累加
print(arr.diff()) #计算累差
print(arr.nonzero()) #找出非零的数字
print(arr.sort()) #逐行进行排序，每一行之间不会干扰
print(arr.T) 或 者print(arr.transpose()) #对矩阵进行转置
print(arr.clip(a,b)) #让矩阵中小于a的数字都变成a，所有大于b的数字都变成b
```

### (3) 按轴计算

```python
np.sum(arr, axis=0)  # 按列求和
np.sum(arr, axis=1)  # 按行求和
```

------

## 5. **数组索引和切片**

### (1) 一维数组索引

```python
arr = np.array([10, 20, 30, 40])
print(arr[2])  # 30
print(arr[-1])  # 40
```

### (2) 多维数组索引

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr[1, 2])  # 6，当然也可以写成arr[1][2]✔️
print(arr[0, :])  # 取第一行
print(arr[:, 1])  # 取第二列
print(arr[0:2, 1:3])  # 取左上角 2×2 矩阵
```

### (3) 条件索引

```python
arr = np.array([10, 20, 30, 40, 50])
print(arr[arr > 25])  # [30 40 50]
```

------

## 6. **数组变形和拼接**

### (1) 变形

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.reshape(3, 2))  # 变成 3×2,reshape方法可以改变矩阵的空间构型,当然也可以实现矩阵的转置
print(arr.flatten())  # 展平成 1D  数组
```

### (2) 拼接

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])

print(np.vstack((a, b)))  # 垂直拼接
print(np.hstack((a, b.T)))  # 水平拼接（b 需要转置）
```

### (3) 拆分

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(np.split(arr, 3, axis=1))  # 按列拆分
print(np.split(arr, 2, axis=0))  # 按行拆分
#进行不等量的分割
print(np.array_split(arr,2,axis=1))
print(np.vsplit(arr,2)) #纵向分割，也就是上下分割
print(np.hsplit(arr,3)) #横向分割，也就是左右分割
```

------

## 7. **广播机制**

`NumPy `允许形状不同的数组进行运算，例如：

```python
a = np.array([[1], [2], [3]])
b = np.array([10, 20, 30])
print(a + b)  
# [[11 21 31]
#  [12 22 32]
#  [13 23 33]]
```

NumPy 会自动扩展较小的数组，使其形状匹配较大的数组。

------

## 8. **线性代数**

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(np.dot(A, B))  # 矩阵乘法
print(A @ B)  # 矩阵乘法的简写
print(np.linalg.inv(A))  # 逆矩阵
print(np.linalg.det(A))  # 矩阵行列式
print(np.linalg.eig(A))  # 特征值和特征向量
```

------

## 9. **随机数与随机数种子**

```python
np.random.seed(42)  # 保证随机数可复现
```

------

`np.random.rand` 是 `NumPy` 库中的一个函数，用于生成指定形状的随机数组，数组中的元素是从 `[0, 1)` 区间内均匀分布的随机数（即 0 到 1 之间，不包含 1）。它是 `NumPy `随机模块（`numpy.random`）的一部分，常用于生成模拟数据或初始化随机值。

### 语法
```python
np.random.rand(d0, d1, ..., dn)
```
- **`d0, d1, ..., dn`**：指定生成数组的形状（维度）。可以传入任意数量的整数参数，表示多维数组的大小。
- **返回值**：一个形状为 `(d0, d1, ..., dn)` 的 NumPy 数组，元素为 `[0, 1)` 之间的随机浮点数。

### 示例
1. **生成单个随机数（标量）**：
   如果不传参数，`np.random.rand()` 会报错。

   ==要生成单个随机数，可以用 `np.random.random()` 或明确指定形状。==

   ```python
   import numpy as np
   print(np.random.rand(1)[0])  # 生成一个数，例如 0.5488135039273248
   ```

2. **生成一维数组**：

   ```python
   import numpy as np
   print(np.random.rand(3))  # 生成 3 个随机数
   ```
   输出示例：
   ```
   [0.5488135  0.71518937 0.60276338]
   ```

3. **生成二维数组**：
   ```python
   import numpy as np
   print(np.random.rand(2, 3))  # 生成 2x3 的随机数组
   ```
   输出示例：
   ```
   [[0.54488318 0.4236548  0.64589411]
    [0.43758721 0.891773   0.96366276]]
   ```

### 特点
- **均匀分布**：生成的随机数服从 `[0, 1)` 的均匀分布。
- **每次运行不同**：除非设置随机种子（`np.random.seed()`），否则每次生成的数都不同。
- **与 `np.random.random` 的区别**：
  - `np.random.rand` 使用形状参数直接指定维度（如 `rand(2, 3)`）。
  - `np.random.random` 需要传入一个元组（如 `random((2, 3))`）。

### 设置随机种子
为了让结果可重复，可以使用 `np.random.seed()`：
```python
import numpy as np
np.random.seed(42)  # 设置种子为 42
print(np.random.rand(3))
```
输出（固定结果）：
```
[0.37454012 0.95071431 0.73199394]
```



## 10.矩阵循环

### 1、利用嵌套for循环

示例：直接用 `for` 循环遍历二维数组

```python
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

# 直接对矩阵进行嵌套的 for 循环
for row in arr:
    for element in row:
        print(element, end=' ')  # 输出：1 2 3 4 5 6
```

解释：

- `for row in arr:` 遍历数组的每一行。
- `for element in row:` 遍历当前行的每一个元素。

这种方式适用于任何维度的数组，只不过对于更高维度的数组，循环会更加复杂。如果是三维或更高维度的数组，你可能需要增加更多的 `for` 循环层次。

例如，对于三维数组：

```python
arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

for matrix in arr:
    for row in matrix:
        for element in row:
            print(element, end=' ')  # 输出：1 2 3 4 5 6 7 8
```

直接使用 `for` 循环访问 NumPy 数组的元素是完全可以的，但对于更复杂的操作，`flat` 或 `flatten` 等方法可能会更简洁。

### 2、利用flat属性

`arr.flat()` 不是正确的调用方式，`flat` 是 `NumPy` 中数组的一个属性，而不是一个方法。正确的语法应该是：

```python
print(arr.flat)
```

`arr.flat` 返回一个迭代器，可以遍历数组的所有元素，类似于展开的视图。

如果你想以一维数组的形式打印所有元素，可以直接使用 `arr.flat`。但如果你想将它转为一个一维数组并输出，可以使用 `arr.flatten()` 方法。示例如下：

#### 示例 1：使用 `arr.flat`（迭代器）

```python
import numpy as np
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.flat)  # 输出的是一个 flat 对象，你可以迭代它
for item in arr.flat:
    print(item, end=' ')  # 输出：1 2 3 4 5 6
```

#### 示例 2：使用 `arr.flatten()`（返回一维数组）

```python
print(arr.flatten())  # 输出：[1 2 3 4 5 6]
```

#### 区别：

- `arr.flat` 是一个迭代器，你可以遍历数组的元素。
- `arr.flatten()` 返回一个新的一维数组，包含数组所有元素。

如果你只是想打印出数组的元素，`flatten()` 更直接。

## 11.矩阵转置

在 `NumPy` 中，`arr.T` 是用来获取数组的转置的。对于一维数组，转置的效果是没有变化的，因为一维数组的转置结果仍然是原数组。

```python
import numpy as np
arr = np.array([1, 2, 3, 4])

print(arr)  # 输出原始一维数组
print(arr.T)  # 输出一维数组的转置（对一维数组没有影响）
```

输出：

```plaintext
[1 2 3 4]
[1 2 3 4]
```

解释：

- 对于一维数组（如 `[1, 2, 3, 4]`），它的转置操作不会改变它。也就是说，`arr.T` 和 `arr` 是相同的。
- **`arr.T` 只对二维及更高维度的数组有效**，例如矩阵（二维数组）才会真正发生转置，即交换行和列。

==要想对一维矩阵进行转置，也就是让行矩阵变成列矩阵，可进行`reshape`方法==



示例：二维数组转置

```python
arr2 = np.array([[1, 2], [3, 4]])

print(arr2)  # 原始矩阵
print(arr2.T)  # 转置后的矩阵
```

**输出：**

```plaintext
[[1 2]
 [3 4]]

[[1 3]
 [2 4]]
```

在二维数组中，`T` 交换了矩阵的行和列。

总结：

- 对于 **一维数组**，`arr.T` 没有效果，数组保持不变。
- 对于 **二维数组**（矩阵），`arr.T` 会交换行和列，即执行转置操作。

## 12、深浅拷贝的问题

==在`numpy`==中，切片不是一个深拷贝，反而是一个浅拷贝哦

在矩阵（或数组）操作中，理解 `copy` 的概念非常重要，尤其是在像 `NumPy` 这样的库中，矩阵的 **拷贝（copy）** 行为直接影响到数据的共享、修改和内存管理。

#### 1. **矩阵的浅拷贝与深拷贝**

**浅拷贝（View）**：

- **浅拷贝**（view）表示的是创建一个新的对象，但这个对象与原始矩阵共享同一块内存空间。因此，修改新矩阵的内容会影响原始矩阵，反之亦然。
- `NumPy` 中的切片、索引、转置等操作，通常返回的是浅拷贝（视图）。

**深拷贝（Copy）**：

- **深拷贝**（copy）表示创建一个新的对象，并且该对象拥有独立的内存空间，修改新矩阵的内容不会影响原始矩阵，反之亦然。
- `NumPy` 中通过 `.copy()` 方法可以显式地创建深拷贝。

#### 2. **浅拷贝（View）示例**

```python
import numpy as np

# 创建一个矩阵
arr = np.array([[1, 2, 3], [4, 5, 6]])

# 切片操作返回的是一个浅拷贝（视图）
arr_view = arr[0:1, :]

print("原始矩阵：\n", arr)
print("视图矩阵：\n", arr_view)

# 修改视图矩阵
arr_view[0, 0] = 99

# 查看修改后的矩阵
print("修改后的原始矩阵：\n", arr)
print("修改后的视图矩阵：\n", arr_view)
```

**输出：**

```plaintext
原始矩阵：
 [[1 2 3]
 [4 5 6]]
视图矩阵：
 [[1 2 3]]
修改后的原始矩阵：
 [[99  2  3]
 [ 4  5  6]]
修改后的视图矩阵：
 [[99  2  3]]
```

#### 解释：

- `arr_view = arr[0:1, :]` 创建了原始矩阵 `arr` 的一个切片，返回的是一个 **浅拷贝**（视图）。
- 修改视图 `arr_view` 中的元素，会影响到原始矩阵 `arr`，因为它们共享同一块内存。

#### 3. **深拷贝（Copy）示例**

如果你不希望矩阵的修改影响到原始矩阵，可以使用 `.copy()` 创建一个深拷贝：

```python
import numpy as np

# 创建一个矩阵
arr = np.array([[1, 2, 3], [4, 5, 6]])

# 使用 .copy() 创建一个深拷贝
arr_copy = arr.copy()

print("原始矩阵：\n", arr)
print("深拷贝矩阵：\n", arr_copy)

# 修改深拷贝矩阵
arr_copy[0, 0] = 99

# 查看修改后的矩阵
print("修改后的原始矩阵：\n", arr)
print("修改后的深拷贝矩阵：\n", arr_copy)
```

**输出：**

```plaintext
原始矩阵：
 [[1 2 3]
 [4 5 6]]
深拷贝矩阵：
 [[1 2 3]
 [4 5 6]]
修改后的原始矩阵：
 [[1 2 3]
 [4 5 6]]
修改后的深拷贝矩阵：
 [[99  2  3]
 [ 4  5  6]]
```

#### 解释：

- `arr_copy = arr.copy()` 创建了一个 **深拷贝**，这意味着 `arr_copy` 和 `arr` 是两个独立的矩阵，它们各自拥有自己的内存空间。
- 修改 `arr_copy` 不会影响原始矩阵 `arr`。

#### 4. **何时使用 `.copy()`？**

在以下情况下，使用 `.copy()` 是非常必要的：

- 当你不希望原始矩阵被修改时（例如，你需要保留原矩阵作为参考），使用 `.copy()` 来创建独立的副本。
- 当你进行矩阵变换（如切片、索引等）并需要修改变换后的矩阵时，使用 `.copy()` 来避免意外地修改原始矩阵。

#### 5. **矩阵切片与 `.copy()` 的区别**

- **切片（浅拷贝）**：使用切片操作时，返回的通常是原矩阵的视图，它与原矩阵共享内存，修改视图会影响原矩阵。
- **`.copy()`（深拷贝）**：使用 `.copy()` 显式创建新的矩阵对象，并复制数据，修改新矩阵不会影响原矩阵。

#### 6. **NumPy 中的 `flags` 属性**

你可以通过 `flags` 属性检查一个 `NumPy` 数组是否是视图（浅拷贝）。如果是视图，`OWNDATA` 标志为 `False`，表示数据是共享的。

示例：

```python
arr = np.array([[1, 2], [3, 4]])
arr_view = arr[0:1, :]

print(arr.flags.owndata)  # True，表示 arr 是独立的内存
print(arr_view.flags.owndata)  # False，表示 arr_view 是 arr 的视图
```

#### 7. **广播与视图**

在进行矩阵广播操作时，`NumPy` 可能会返回视图（浅拷贝）。例如，当你执行标量与矩阵相乘时，矩阵的数据不会被复制，而是会在原有数据上操作，这样能提高效率。

示例：

```python
arr = np.array([[1, 2], [3, 4]])
arr2 = arr * 2  # 广播操作，返回的是视图而非副本
print(arr2)
```

#### 8.小结：

- **浅拷贝（View）**：切片、转置等操作返回的通常是视图，修改视图会影响原矩阵。
- **深拷贝（Copy）**：通过 `.copy()` 方法可以创建独立的副本，修改副本不会影响原矩阵。
- 使用 `.copy()` 可以确保矩阵数据不受外部修改的影响，尤其在进行复杂计算或保存数据时很有用。



## 13、矩阵的常见误区

你的疑问很合理！让我澄清一下 `np.random.rand(1,4)` 生成的数组维度问题，以及它为什么不是一维数组。

#### `np.random.rand(1,4)` 的结果
`np.random.rand(1,4)` 生成的是一个 **二维数组**，其形状（shape）为 `(1, 4)`。这里的 `1` 表示第一维的大小（行数），`4` 表示第二维的大小（列数）。虽然它只有一个“行”，但它仍然是一个二维结构，而不是一维数组。

示例

```python
import numpy as np
array = np.random.rand(1, 4)
print(array)
print(array.shape)
```

输出示例：
```
[[0.37454012 0.95071431 0.73199394 0.59865848]]
(1, 4)
```
- **内容**：一个包含 4 个随机数的“行”。
- **形状**：`(1, 4)`，表示 1 行 4 列。
- **维度**：`array.ndim` 会返回 `2`，确认它是二维的。

#### 为什么不是一维的？
在 `NumPy `中，数组的维度由传入的参数数量决定：
- **`np.random.rand(4)`**：传入一个参数，生成形状为 `(4,)` 的一维数组。
- **`np.random.rand(1, 4)`**：传入两个参数，生成形状为 `(1, 4)` 的二维数组。

即使 `(1, 4)` 的数组看起来像“只有一行”，NumPy 仍然将其视为二维，因为它明确定义了两级结构（行和列）。一维数组的形状是 `(n,)`，没有额外的维度。

#### 一维 vs 二维对比

```python
import numpy as np

# 一维数组
array_1d = np.random.rand(4)
print("1D:", array_1d)
print("Shape:", array_1d.shape)  # (4,)
print("Dim:", array_1d.ndim)     # 1

# 二维数组
array_2d = np.random.rand(1, 4)
print("2D:", array_2d)
print("Shape:", array_2d.shape)  # (1, 4)
print("Dim:", array_2d.ndim)     # 2
```



------

## 14、np.any()方法

看起来你想问的是 NumPy 中的 `np.any()` 函数，而不是 pandas 的 `dropna()` 中的 `how='any'`。没问题，我来用中文为你详细解释 `np.any()` 的用法和功能！

#### `np.any()` 的定义
`numpy.any()` 是 NumPy 库中的一个函数，用于测试数组中是否至少有一个元素满足条件（即为 `True`）。简单来说，它检查数组中“哪怕有一个”元素是 `True`，就返回 `True`。

#### 语法
```python
numpy.any(a, axis=None, out=None, keepdims=False)
```

#### 参数说明
1. **`a`**：输入数组（可以是 NumPy 数组或其他可转换为数组的对象）。
2. **`axis`**：
   - 默认 `None`：对整个数组进行检查。
   - `0`：沿列方向检查。
   - `1`：沿行方向检查。
3. **`out`**：可选的输出数组，用于存储结果（一般很少用）。
4. **`keepdims`**：如果为 `True`，保留减少的维度（结果形状与输入一致）。

#### 返回值
- 如果数组中至少有一个元素为 `True`，返回 `True`。
- 如果所有元素都为 `False`，返回 `False`。

#### 示例
#### 1. 基本用法
```python
import numpy as np

arr = np.array([False, False, True, False])
result = np.any(arr)
print(result)  # 输出：True
```
解释：数组中哪怕有一个 `True`，`np.any()` 就返回 `True`。

#### 2. 全为 False
```python
arr = np.array([False, False, False])
result = np.any(arr)
print(result)  # 输出：False
```
解释：没有一个元素是 `True`，所以返回 `False`。

#### 3. 结合条件
常用来检查数组中是否满足某个条件：
```python
arr = np.array([1, 0, -3, 5])
result = np.any(arr < 0)
print(result)  # 输出：True
```
解释：数组中有一个元素 `-3` 小于 0，哪怕只有一个，`np.any()` 也返回 `True`。

#### 4. 指定轴
对于多维数组，可以沿特定轴检查：
```python
arr = np.array([[0, 0, 1],
                [0, 0, 0]])
result = np.any(arr, axis=0)
print(result)  # 输出：[False False  True]
```
解释：沿列检查（`axis=0`），第三列有 `1`（`True`），所以结果是 `[False, False, True]`。

```python
result = np.any(arr, axis=1)
print(result)  # 输出：[ True False]
```
解释：沿行检查（`axis=1`），第一行有 `1`，第二行全是 `0`，所以结果是 `[True, False]`。

#### 与 `np.all()` 的区别

- `np.any()`：只要有一个 `True` 就返回 `True`。
- `np.all()`：要求所有元素都是 `True` 才返回 `True`。

#### 实际应用
- 检查数组中是否有非零值：`np.any(arr != 0)`。
- 检查是否有缺失值（配合 pandas）：`np.any(pd.isna(df))`。
- 判断条件是否至少在某个地方成立。

## 15、洗牌操作

在 `NumPy` 中，洗牌操作可以通过 `numpy.random.shuffle()` 或 `numpy.random.permutation()` 来实现。两者的区别如下：

### 1. `numpy.random.shuffle()`

`shuffle()` 是一种就地（in-place）操作，它会随机打乱数组的顺序，修改原始数组。

**示例**：

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
np.random.shuffle(arr)  # 直接修改 arr 数组
print(arr)
```

输出（每次运行可能不同）：

```
[4 2 5 1 3]
```

**注意**：`shuffle()` 直接修改原数组，并且仅适用于一维数组。如果你想洗牌多维数组的行，可以对每一行单独进行操作。

### 2. `numpy.random.permutation()`

`permutation()` 返回一个新的数组，数组中的元素是原始数组的洗牌版本，不会修改原始数组。

**示例**：

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
shuffled_arr = np.random.permutation(arr)  # 返回一个新的洗牌数组
print("Original array:", arr)
print("Shuffled array:", shuffled_arr)
```

输出：

```
Original array: [1 2 3 4 5]
Shuffled array: [4 2 1 5 3]
```

### 区别总结：

- `shuffle()` 是就地修改，直接改变原数组。
- `permutation()` 返回一个新的数组，原数组不受影响。

这两个函数都可以用于打乱数据顺序，选择哪个取决于是否希望修改原始数据。

## 16、索引问题

在 NumPy 中，索引二维数组的某一行（self.theta[label_index]）返回一维数组 (3,)，而不是二维的 (1, 3)。这是 NumPy 的默认行为。如果你想要 (1, 3)，需要用切片（如 self.theta[label_index:label_index+1]）。

当你用单个索引（如 arr[0]）访问二维数组的某一行时，NumPy 返回一个一维数组 (3,)，而不是保持二维结构 (1, 3)。

如果你想要得到形状 (1, 3)，可以使用切片或显式索引

```python
import numpy as np
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr[0])
print(arr[0].shape)

print(arr[0:1])        # 输出: [[1 2 3]]
print(arr[0:1].shape)  # 输出: (1, 3)

print(arr[0][np.newaxis, :])        # 输出: [[1 2 3]]
print(arr[0][np.newaxis, :].shape)  # 输出: (1, 3)

```

## 17、meshgrid 

meshgrid 是 NumPy 库中的一个函数，用于从一维坐标数组生成多维网格坐标矩阵。np.meshgrid 的核心功能是将多个一维坐标数组扩展为多维网格的坐标矩阵。给定一维的 x 轴、y 轴（或其他维度）坐标，它生成对应的网格点坐标矩阵，每个矩阵表示一个维度的坐标值。

- **输入**：多个一维数组（通常表示不同维度的坐标）。
- **输出**：多个数组（通常是二维矩阵），表示网格点在每个维度上的坐标值。

最常见的情况是生成二维网格（x-y 平面），但它也支持更高维（如三维 x-y-z 空间）。

```
x0,x1=np.meshgrid(np.linspace(1,2,2).reshape(-1,1),
					np.linspace(10,20,3).reshape(-1,1))
np.c_[x0.ravel(),x1.ravel()]     #ravel，可以把二维矩阵平铺成一维数组
```

## 18、c_

### 功能

np.c_ 的核心功能是将多个数组（通常是一维或二维）沿着**第二个轴（列方向）**拼接，生成一个新的二维数组。每个输入数组被视为一列（或多列），拼接后形成一个矩阵。

- **输入**：多个 NumPy 数组（可以是一维或二维，形状需兼容）。
- **输出**：一个二维数组，列数是所有输入数组列数的总和，行数由输入数组的行数决定。

np.c_ 是 NumPy 索引表达式的简写，等价于使用 np.concatenate 或 np.hstack，但更简洁，专为列拼接设计。

------

### 用法

np.c_ 的基本语法如下：

```
import numpy as np result = np.c_[array1, array2, ..., arrayN]
```

- 参数：
  - array1, array2, ..., arrayN：要拼接的数组，可以是一维或二维。
  - 要求所有数组的**行数**（或一维数组的长度）相同，否则会抛出错误。
  - 如果输入是一维数组，会先将其转换为二维列向量（形状 (n, 1)）。
- 返回：
  - 一个二维数组，形状为 (n, m1 + m2 + ... + mN)，其中：
    - n 是输入数组的行数（或一维数组的长度）。
    - m1, m2, ..., mN 是每个输入数组的列数。



## 19、  ravel  和  flatten  对比

在 NumPy 中，`ravel()` 和 `flatten()` 是两个用于将多维数组展平为一维数组的方法。尽管它们的功能看起来类似（都返回一维数组），但在实现细节、性能和行为上存在一些关键差异。以下是对 `ravel()` 和 `flatten()` 的纯讲解，专注于它们的定义、用法、区别和应用场景。

---

### 1. **`ravel()`**

#### 定义
`ravel()` 是 NumPy 数组的一个方法，返回数组的展平视图（flattened view），即将多维数组转换为一维数组。它尽量返回原数组的视图（view），而不是复制数据，从而节省内存。

#### 用法
```python
import numpy as np
array.ravel(order='C')
```

- **参数**：
  - `order`：展平的顺序，默认为 `'C'`（按行优先，C 风格）。
    - `'C'`：按行展平（默认）。
    - `'F'`：按列展平（Fortran 风格）。
    - `'A'`：根据内存布局选择 `'C'` 或 `'F'`。
    - `'K'`：尽量按原始内存顺序。

- **返回**：
  - 一维数组（视图或复制，取决于输入数组的内存布局）。
  - 如果原数组是连续的（contiguous），返回视图；如果不是，可能返回复制。

#### 示例
```python
import numpy as np

# 二维数组
a = np.array([[1, 2, 3],
              [4, 5, 6]])

# 使用 ravel
b = a.ravel()
print("ravel 结果:", b)
print("形状:", b.shape)

# 修改 b 的值
b[0] = 99
print("修改后的原数组 a:\n", a)
```

**输出**：
```
ravel 结果: [1 2 3 4 5 6]
形状: (6,)
修改后的原数组 a:
 [[99  2  3]
  [ 4  5  6]]
```

- **解释**：
  - `a` 是 `(2, 3)` 的二维数组，`ravel()` 将其展平为长度 6 的一维数组。
  - `b` 是 `a` 的视图，修改 `b` 会影响原数组 `a`（因为是视图）。
  - 展平顺序是按行（`order='C'`）：`[1, 2, 3, 4, 5, 6]`。

#### 特点
- **视图优先**：尽量返回原数组的视图，修改结果会影响原数组。
- **内存效率**：如果返回视图，不复制数据，节省内存。
- **适用场景**：需要展平数组且允许修改原数组，或追求性能时。

---

### 2. **`flatten()`**

#### 定义
`flatten()` 是 NumPy 数组的一个方法，返回数组的展平复制（flattened copy），即将多维数组转换为一维数组。它总是返回数据的独立副本，修改结果不会影响原数组。

#### 用法
```python
import numpy as np
array.flatten(order='C')
```

- **参数**：
  - `order`：展平顺序，与 `ravel()` 相同：
    - `'C'`：按行展平（默认）。
    - `'F'`：按列展平。
    - `'A'`：根据内存布局。
    - `'K'`：按内存顺序。

- **返回**：
  - 一维数组，总是原数组的复制（copy）。

#### 示例
```python
import numpy as np

# 二维数组
a = np.array([[1, 2, 3],
              [4, 5, 6]])

# 使用 flatten
b = a.flatten()
print("flatten 结果:", b)
print("形状:", b.shape)

# 修改 b 的值
b[0] = 99
print("修改后的原数组 a:\n", a)
```

**输出**：
```
flatten 结果: [1 2 3 4 5 6]
形状: (6,)
修改后的原数组 a:
 [[1 2 3]
  [4 5 6]]
```

- **解释**：
  - `a` 是 `(2, 3)` 的二维数组，`flatten()` 将其展平为长度 6 的一维数组。
  - `b` 是 `a` 的复制，修改 `b` 不影响原数组 `a`。
  - 展平顺序是按行（`order='C'`）：`[1, 2, 3, 4, 5, 6]`。

#### 特点
- **复制优先**：总是返回原数组的复制，修改结果不影响原数组。
- **内存开销**：需要分配新内存，复制数据，内存使用较多。
- **适用场景**：需要展平数组且不希望影响原数组，或需要独立操作。

---

### `ravel()` vs. `flatten()`：关键区别

| 特性         | `ravel()`                            | `flatten()`                |
| ------------ | ------------------------------------ | -------------------------- |
| **返回类型** | 视图（优先），可能复制               | 总是复制                   |
| **修改影响** | 修改结果通常影响原数组（如果是视图） | 修改结果不影响原数组       |
| **内存效率** | 高（视图不复制数据）                 | 低（总是复制数据）         |
| **性能**     | 更快（避免复制）                     | 稍慢（需要复制）           |
| **方法来源** | NumPy 数组方法                       | NumPy 数组方法             |
| **参数**     | `order`（C, F, A, K）                | `order`（C, F, A, K）      |
| **适用场景** | 性能敏感、允许修改原数组             | 需要独立副本、不影响原数组 |

---

### 更详细的比较

1. **视图与复制**：
   - `ravel()`：
     - 如果原数组是连续存储（C 风格或 Fortran 风格），`ravel()` 返回视图。
     - 如果原数组非连续（例如切片或转置后的数组），可能返回复制。
     - 示例（非连续数组）：
       ```python
       a = np.array([[1, 2], [3, 4]])
       b = a[:, 0]  # 非连续
       c = b.ravel()
       c[0] = 99
       print("b:", b)  # 未改变，因为 c 是复制
       ```
   - `flatten()`：
     - 总是返回复制，无论数组是否连续。
     - 示例：
       ```python
       a = np.array([[1, 2], [3, 4]])
       b = a.flatten()
       b[0] = 99
       print("a:", a)  # 未改变
       ```

2. **性能差异**：
   - `ravel()` 通常更快，因为视图操作只需调整数组的元数据（strides），而不需要复制数据。
   - `flatten()` 需要分配新内存并复制所有元素，耗时更多，尤其在大数组上。
   - 示例（性能测试）：
     ```python
     import numpy as np
     import time
     
     a = np.random.rand(1000, 1000)
     start = time.time()
     for _ in range(1000):
         a.ravel()
     print("ravel 时间:", time.time() - start)
     
     start = time.time()
     for _ in range(1000):
         a.flatten()
     print("flatten 时间:", time.time() - start)
     ```
     - `ravel()` 通常比 `flatten()` 快数倍。

3. **内存使用**：
   - `ravel()`：视图只引用原数组数据，内存开销极小。
   - `flatten()`：复制数据需要额外内存，等于原数组大小。
   - 对于大数组（例如 1GB），`flatten()` 可能导致内存不足，而 `ravel()` 更安全。

4. **展平顺序**：
   - 两者都支持 `order` 参数，控制展平方式。
   - 示例（按列展平）：
     ```python
     a = np.array([[1, 2], [3, 4]])
     print(a.ravel(order='F'))   # [1, 3, 2, 4]
     print(a.flatten(order='F')) # [1, 3, 2, 4]
     ```

---

### 常见用途

1. **展平多维数组**：
   - 将高维数组（如矩阵或张量）转换为一维，用于后续计算。
   - 示例：
     ```python
     a = np.array([[1, 2], [3, 4]])
     flat = a.ravel()  # [1, 2, 3, 4]
     ```

2. **机器学习输入**：
   - 将网格坐标或其他多维数据展平，输入模型。
   - 示例：
     ```python
     x, y = np.meshgrid([1, 2], [3, 4])
     points = np.c_[x.ravel(), y.ravel()]  # 坐标点
     ```

3. **数据处理**：
   - 展平后进行统计计算（如均值、排序）。
   - 示例：
     ```python
     a = np.array([[1, 2], [3, 4]])
     mean = a.ravel().mean()  # 2.5
     ```

4. **可视化**：
   - 配合 `meshgrid`，展平网格点以计算函数值或绘制图形。
   - 示例：
     ```python
     x, y = np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3))
     z = (x.ravel()**2 + y.ravel()**2).reshape(x.shape)
     ```

---

### 注意事项

1. **选择依据**：
   - 用 `ravel()`：当性能和内存是优先考虑，且允许修改原数组。
   - 用 `flatten()`：当需要独立副本，或不希望影响原数组。

2. **视图的潜在风险**：
   - `ravel()` 返回视图时，修改结果会改变原数组，可能导致意外后果。
   - 示例：
     ```python
     a = np.array([[1, 2], [3, 4]])
     b = a.ravel()
     b[0] = 99  # a 变为 [[99, 2], [3, 4]]
     ```

3. **替代方法**：
   - `np.ravel(array)`：全局函数，类似 `array.ravel()`，但可直接应用于非数组对象。
   - `array.flat`：返回迭代器，适合逐元素访问。
   - `array.reshape(-1)`：功能类似 `ravel()`，但语法不同。
   - 示例：
     ```python
     a = np.array([[1, 2], [3, 4]])
     print(np.ravel(a))       # [1, 2, 3, 4]
     print(a.reshape(-1))     # [1, 2, 3, 4]
     ```

4. **非连续数组**：
   - 对于切片或转置数组，`ravel()` 可能返回复制（而非视图）。
   - 示例：
     ```python
     a = np.array([[1, 2], [3, 4]])
     b = a.T  # 转置，非连续
     c = b.ravel()
     c[0] = 99  # 不影响 a
     ```

---

### 总结

- **`ravel()`**：
  - 返回展平的视图（优先），修改可能影响原数组。
  - 内存高效，性能快。
  - 适合性能敏感场景。
- **`flatten()`**：
  - 返回展平的复制，修改不影响原数组。
  - 内存开销大，性能稍慢。
  - 适合需要独立副本的场景。
- **共同点**：
  - 将多维数组展平为一维。
  - 支持 `order` 参数（C, F, A, K）。
- **用途**：
  - 数据展平、模型输入、可视化、统计计算。

如果你对 `ravel()` 或 `flatten()` 的某个细节有疑问（比如内存管理、性能测试或高级用法），告诉我，我可以进一步讲解！

## 20、numpt.sum函数

`np.sum` 是 NumPy 库中的一个函数，用于计算数组元素的总和。它非常灵活，支持多维数组、指定轴、数据类型控制等功能，是科学计算中常用的工具。以下是对 `np.sum` 的纯讲解，专注于其功能、参数、使用场景和输出效果。

---

### 功能

`np.sum` 计算给定数组中所有元素或沿指定轴的元素之和，返回一个标量（对整个数组）或数组（沿轴求和）。

- **输入**：NumPy 数组或其他类似数组的对象。
- **输出**：总和（标量或数组），数据类型可控。
- **用途**：统计、矩阵运算、数据处理等。

---

### 用法

基本语法：
```python
import numpy as np
np.sum(a, axis=None, dtype=None, out=None, keepdims=False, initial=0, where=True)
```

- **参数**：
  - **`a`**：
    - 输入数组，必需参数。
    - 可以是 NumPy 数组、列表或其他可转换为数组的对象。
  - **`axis`**：
    - 指定沿哪个轴求和，默认为 `None`。
    - `None`：对整个数组展平后求和，返回标量。
    - 整数或元组：沿指定轴（或多个轴）求和，返回数组。
  - **`dtype`**：
    - 输出和的精确数据类型，默认为输入数组类型或推导类型。
    - 示例：`np.float64`, `np.int32`。
  - **`out`**：
    - 可选的输出数组，用于存储结果（需形状匹配）。
    - 默认 `None`，返回新数组。
  - **`keepdims`**：
    - 布尔值，默认为 `False`。
    - `True`：保留被求和轴的维度（长度为 1）。
    - `False`：移除被求和轴。
  - **`initial`**：
    - 求和的起始值，默认为 0。
    - 影响结果：`sum = initial + elements_sum`。
  - **`where`**：
    - 布尔数组，指定哪些元素参与求和。
    - 默认 `True`，包含所有元素。

- **返回**：
  - 标量（`axis=None` 时）或数组（沿轴求和时）。
  - 数据类型由 `dtype` 或输入数组决定。

---

### 工作原理

- **整体求和**：
  - 当 `axis=None` 时，`np.sum` 将数组展平为一维，然后计算所有元素之和。
  - 示例：`[[1, 2], [3, 4]]` 展平为 `[1, 2, 3, 4]`，和为 `10`。

- **按轴求和**：
  - 当指定 `axis` 时，沿该轴计算每个子数组的和。
  - 轴是数组的维度：
    - `axis=0`：沿行方向（跨行），对每列求和。
    - `axis=1`：沿列方向（跨列），对每行求和。
  - 输出形状移除被求和的轴（除非 `keepdims=True`）。

- **灵活性**：
  - 支持多维数组、条件求和（`where`）、自定义类型（`dtype`）。

---

### 示例

#### 示例 1：整体求和
```python
import numpy as np

a = np.array([[1, 2], [3, 4]])
result = np.sum(a)
print("Sum:", result)
```

**输出**：
```
Sum: 10
```

- **解释**：
  - `a` 是 `(2, 2)` 数组，展平为 `[1, 2, 3, 4]`。
  - `np.sum(a)` 计算 `1 + 2 + 3 + 4 = 10`，返回标量。

#### 示例 2：按轴求和
```python
a = np.array([[1, 2], [3, 4]])
sum_axis0 = np.sum(a, axis=0)
sum_axis1 = np.sum(a, axis=1)
print("Sum along axis 0:", sum_axis0)
print("Sum along axis 1:", sum_axis1)
```

**输出**：
```
Sum along axis 0: [4 6]
Sum along axis 1: [3 7]
```

- **解释**：
  - `a`：
    ```python
    [[1, 2],
     [3, 4]]
    ```
  - `axis=0`（跨行，对每列求和）：
    - 第 1 列：`1 + 3 = 4`。
    - 第 2 列：`2 + 4 = 6`。
    - 结果：`[4, 6]`，形状 `(2,)`。
  - `axis=1`（跨列，对每行求和）：
    - 第 1 行：`1 + 2 = 3`。
    - 第 2 行：`3 + 4 = 7`。
    - 结果：`[3, 7]`，形状 `(2,)`。

#### 示例 3：保留维度
```python
a = np.array([[1, 2], [3, 4]])
result = np.sum(a, axis=0, keepdims=True)
print("Sum with keepdims:", result)
print("Shape:", result.shape)
```

**输出**：
```
Sum with keepdims: [[4 6]]
Shape: (1, 2)
```

- **解释**：
  - `axis=0`：对每列求和，`[4, 6]`。
  - `keepdims=True`：保留轴 0，输出形状 `(1, 2)` 而不是 `(2,)`。
  - 结果是 `[[4, 6]]`，便于广播或后续操作。

#### 示例 4：指定数据类型
```python
a = np.array([1.5, 2.5])
result = np.sum(a, dtype=np.int32)
print("Sum with int32:", result)
```

**输出**：
```
Sum with int32: 4
```

- **解释**：
  - `a` 是浮点数组，`1.5 + 2.5 = 4.0`。
  - `dtype=np.int32`：强制结果为整数，`4.0` 转换为 `4`。

#### 示例 5：条件求和
```python
a = np.array([1, 2, 3, 4])
result = np.sum(a, where=a > 2)
print("Sum where a > 2:", result)
```

**输出**：
```
Sum where a > 2: 7
```

- **解释**：
  - `where=a > 2`：只对满足条件的元素求和。
  - `a > 2`：`[False, False, True, True]`。
  - 求和：`3 + 4 = 7`。

#### 示例 6：初始值
```python
a = np.array([1, 2])
result = np.sum(a, initial=10)
print("Sum with initial=10:", result)
```

**输出**：
```
Sum with initial=10: 13
```

- **解释**：
  - 数组和：`1 + 2 = 3`。
  - 加初始值：`3 + 10 = 13`。

---

### 常见用途

1. **统计计算**：
   - 计算数组总和、均值（结合 `np.mean`）、方差等。
   - 示例：`np.sum(data)` 计算数据集总和。

2. **矩阵运算**：
   - 按行或列求和，简化矩阵操作。
   - 示例：`np.sum(matrix, axis=1)` 计算每行和。

3. **机器学习**：
   - 计算损失函数、距离（如欧几里得距离）。
   - 示例：`np.sum((x - y)**2)` 计算平方误差。

4. **数据处理**：
   - 条件求和，过滤无效值。
   - 示例：`np.sum(data, where=data > 0)` 计算正值和。

5. **多维数组操作**：
   - 沿指定轴汇总数据，降维处理。
   - 示例：`np.sum(tensor, axis=(1, 2))` 降维求和。

---

### 注意事项

1. **轴的选择**：
   - `axis` 从 0 开始，需匹配数组维度。
   - 错误示例：
     ```python
     a = np.array([[1, 2]])
     np.sum(a, axis=2)  # 错误：轴超出范围
     ```

2. **数据类型**：
   - 默认 `dtype` 可能导致精度问题（如整数溢出）。
   - 示例：
     ```python
     a = np.array([2**31, 1])
     print(np.sum(a, dtype=np.int32))  # 溢出
     print(np.sum(a, dtype=np.int64))  # 正确
     ```

3. **空数组**：
   - 空数组求和返回 `initial` 值（默认 0）。
   - 示例：
     ```python
     np.sum([], initial=5)  # 返回 5
     ```

4. **性能**：
   - `np.sum` 比 Python 内置 `sum` 快，尤其在大数组上。
   - 示例：
     ```python
     a = np.random.rand(1000000)
     %timeit np.sum(a)  # 更快
     %timeit sum(a)     # 较慢
     ```

5. **与 np.add.reduce 的关系**：
   - `np.sum` 等价于 `np.add.reduce`：
     ```python
     np.sum(a) == np.add.reduce(a)
     ```
   - 但 `np.sum` 更直观，参数更丰富。

6. **多轴求和**：
   - 支持元组形式：
     ```python
     a = np.ones((2, 3, 4))
     np.sum(a, axis=(1, 2))  # 形状 (2,)
     ```

---

### 总结

- **功能**：
  - `np.sum` 计算数组元素总和，支持整体或按轴求和。
- **关键参数**：
  - `a`：输入数组。
  - `axis`：求和轴（`None`, 整数或元组）。
  - `dtype`：输出类型。
  - `keepdims`：保留维度。
  - `where`：条件求和。
  - `initial`：起始值。
- **输出**：
  - 标量（整体求和）或数组（按轴求和）。
- **用途**：
  - 统计、矩阵运算、机器学习、数据处理。
- **特点**：
  - 高效、灵活，支持多维数组和条件操作。

如果你对 `np.sum` 的某个细节有疑问（比如性能优化、轴操作或特殊用例），告诉我，我可以进一步讲解！

# 2、pandas包

------

`pandas` 是 Python 中一个强大的数据分析库，广泛应用于数据清洗、分析、可视化和建模。它提供了 **DataFrame** 和 **Series** 两种核心数据结构，使得处理和分析表格数据变得简单而高效。下面是一些 `pandas` 的常用知识和操作：

### 1. **安装 pandas**

首先确保你已安装了 `pandas`，可以通过以下命令安装：

```bash
pip install pandas
```

### 2. **导入 pandas**

通常，我们使用 `pd` 来作为 `pandas` 的别名：

```python
import pandas as pd
```

### 3. **pandas 核心数据结构**

- **Series**：一维数据结构，类似于 Python 的列表或数组。
- **Data Frame**：二维数据结构，类似于数据库中的表格、Excel 工作表，或 R 中的`data.frame`。
- **切片 `a:b` 总是按行操作**，相当于 `.iloc[a:b]`。

  **单个索引 `x` 默认按列标签取列**。

#### 创建 Series  

```python
# 使用列表创建 Series
s = pd.Series([1, 2, 3, 4, 5])
print(s)
```

#### 创建 Data Frame

```python
# 使用字典创建 DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [24, 27, 22], 'City': ['NY', 'LA', 'SF']}
df = pd.DataFrame(data,index=?,columns=?)
print(df)
```

### 4. **基本操作**

#### 查看数据

```python
# 查看 DataFrame 的前几行
print(df.head())  # 默认查看前 5 行

# 查看 DataFrame 的后几行
print(df.tail())

# 查看 DataFrame 的基本信息（列数、非空值等）
print(df.info())

# 查看数据的基本统计信息
print(df.describe())

#属性不需要括号,方法才需要
print(df.dtypes)#df的类型
print(df.index)#df的行标签
print(df.columns)#df的列标签
print(df.values)#df的值,可以转换为numpy矩阵
print(df.describe())#df的数值情况比如方差、平均值等等，编译器会忽略非数值数据,需要注意的是，describe是一个方法而非属性
print(df.T) #同样可以进行转置
```

#### 获取列

```python
# 获取单独的列
print(df['Name'])

# 使用属性方式获取列
print(df.Name)
```

#### 获取行

```python
# 通过索引获取单行
print(df.iloc[0])  # 获取第 1 行（按位置）

# 通过标签获取单行
print(df.loc[0])  # 获取第 1 行（按标签），pandas的切片操作是左闭右闭
print(df2.loc[0:2,['A','C']])
# 获取多个行
print(df.iloc[0:2])  # 获取前两行
print(df.iloc[1:2,2:3]) # 获取12行23列
print(df.iloc[1,2]) #iloc方法是通过位置索引查找
#增加一列
print(df['F']=np.nan)
```

#### 设置索引

```python
# 将列设置为索引
df.set_index('Name', inplace=True)
print(df)
```

#### 重置索引

```python
# 重置索引
df.reset_index(inplace=True)
print(df)
```

### 5. **数据选择与过滤**

#### 条件筛选

```python
# 选择符合条件的行
df_filtered = df[df['Age'] > 24]
print(df_filtered)
print(df.Age[df['Age']>24])
print(df.NUM[df['Age']>24])


```

#### 在 Pandas 中，可以用布尔数组筛选 Series 或 DataFrame 的行。

`data [x_axis][data['class']== iris_type] `提取 petal_length 中对应于 iris_type 类别的值。

#### 选择特定的列

```python
# 选择特定的列
print(df[['Name', 'Age']])
```

#### 使用多个条件筛选

```python
# 使用 & (and) 或 | (or) 进行多条件筛选
df_filtered = df[(df['Age'] > 24) & (df['City'] == 'NY')]
print(df_filtered)
```

### 6. **缺失数据处理**

#### 检查缺失数据

```python
# 检查是否有缺失值
print(df.isnull())  # 返回一个布尔值的 DataFrame

# 检查每列缺失数据的总数
print(df.isnull().sum())
```

#### 填充缺失值

```python
# 使用特定值填充缺失值
df.fillna(0, inplace=True)

# 使用前一个值填充缺失值
df.fillna(method='ffill', inplace=True)

# 使用后一个值填充缺失值
df.fillna(method='bfill', inplace=True)
```

#### 删除缺失数据

```python
# 删除包含缺失数据的行
df.dropna(inplace=True)
#需要注意是的，dropna方法并非直接删除，而是给你隐藏掉了，因此我们需要有一个新的变量去接收返回值
# 删除包含缺失数据的列
df.dropna(axis=1, inplace=True)
```

### 7. **数据清洗与处理**

#### 重命名列

```python
# 重命名列
df.rename(columns={'Age': 'Age (Years)'}, inplace=True)
```

#### 合并与连接

- **合并（Merge）**：类似 SQL 中的 `JOIN` 操作。

```python
df1 = pd.DataFrame({'ID': [1, 2, 3], 'Name': ['Alice', 'Bob', 'Charlie']})
df2 = pd.DataFrame({'ID': [2, 3, 4], 'Age': [24, 27, 22]})

# 按 'ID' 列合并
merged_df = pd.merge(df1, df2, on='ID', how='inner')  # 'inner' 表示内连接
print(merged_df)
```

- **连接（Concat）**：按行或列连接多个 DataFrame。

```python
# 按行连接
df_concat = pd.concat([df1, df2], axis=0)
df_concat = pd.concat([df1, df2], axis=0，ignore_idnex=True)#忽略原有序列的索引
print(df_concat)

# 按列连接
df_concat = pd.concat([df1, df2], axis=1)
print(df_concat)
```

### 8. **排序与排名**

#### 排序

```python
# 按照列排序
df.sort_values(by='Age', ascending=False, inplace=True)

# 按多个列排序
df.sort_values(by=['Age', 'Name'], ascending=[False, True], inplace=True)
```

#### 排名

```python
# 按列计算排名
df['Age Rank'] = df['Age'].rank(ascending=False)
print(df)
```

### 9. **分组操作（GroupBy）**

#### 分组并聚合

```python
# 根据某一列进行分组，并计算每组的统计信息
df_grouped = df.groupby('City')['Age'].mean()
print(df_grouped)
```

#### 多重聚合

```python
# 对每一组应用多个聚合函数
df_grouped = df.groupby('City').agg({'Age': ['mean', 'max', 'min']})
print(df_grouped)
```

### 10. **数据透视表（Pivot Table）**

```python
# 创建数据透视表
pivot_df = df.pivot_table(values='Age', index='City', aggfunc='mean')
print(pivot_df)
```

### 11. **时间序列**

#### 转换为日期时间格式

```python
# 将列转换为日期时间格式
df['Date'] = pd.to_datetime(df['Date'])
```

#### 生成时间序列

```python
# 生成日期范围
date_range = pd.date_range('2023-01-01', periods=5, freq='D')
print(date_range)
```

#### 时间切片

```python
# 设置时间列为索引
df.set_index('Date', inplace=True)

# 选择某个时间范围的数据
df_filtered = df['2023-01-01':'2023-01-03']
print(df_filtered)
```

### 12. **文件读取与写入**

#### 1、读取 CSV 文件

```python
# 从 CSV 文件读取数据
df = pd.read_csv('data.csv')

# 从 Excel 文件读取数据
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
```

#### 2、写入文件

```python
# 将 DataFrame 写入 CSV 文件
df.to_csv('output.csv', index=False)

# 将 DataFrame 写入 Excel 文件
df.to_excel('output.xlsx', index=False)
```

==以下是用 pandas 读取和保存 pickle 及 CSV 文件的语法，不含示例：==

- **读取 pickle 文件**
```python
pd.read_pickle('文件路径.pkl')
```

- **读取 CSV 文件**
```python
pd.read_csv('文件路径.csv', encoding='utf-8')
```

#### 3、保存文件

- **保存为 pickle 文件**
```python
DataFrame.to_pickle('输出路径.pkl', compression='infer', protocol=5)
```

- **保存为 CSV 文件**
```python
DataFrame.to_csv('输出路径.csv', index=False, encoding='utf-8')
```





## 13.sort_index 和 sort_values

 在 Pandas 中，`sort_index` 和 `sort_values` 是两个常用的方法，用于对 Data Frame 或 Series 进行排序。它们的功能和用法有所不同，下面我详细解释一下：

---

### 1. `sort_index`

- **作用**：按照索引（index）对 Data Frame 或 Series 进行排序。
- **适用于**：当你想基于行索引或列索引重新排列数据时。

#### 语法
```python
DataFrame.sort_index(axis=0, ascending=True, inplace=False, ...)
Series.sort_index(ascending=True, inplace=False, ...)
```
- **`axis`**：排序的轴，`0` 表示按行索引排序（默认），`1` 表示按列索引排序。
- **`ascending`**：是否升序排序，`True`（默认）为升序，`False` 为降序。
- **`inplace`**：是否修改原数据，`False`（默认）返回新对象，`True` 修改原对象。

#### 示例
```python
import pandas as pd

# 创建一个 DataFrame
df = pd.DataFrame({
    "A": [1, 3, 2],
    "B": [4, 6, 5]
}, index=[2, 0, 1])

print("原始 DataFrame:")
print(df)

# 按行索引排序
print("\n按行索引排序:")
print(df.sort_index())

# 按列索引排序
print("\n按列索引排序:")
print(df.sort_index(axis=1))
```

输出：
```
原始 DataFrame:
   A  B
2  1  4
0  3  6
1  2  5

按行索引排序:
   A  B
0  3  6
1  2  5
2  1  4

按列索引排序:
   A  B
2  1  4
0  3  6
1  2  5
```
- **解释**：`sort_index()` 按索引 `[2, 0, 1]` 重新排序为 `[0, 1, 2]`，列顺序不变。`axis=1` 时按列名排序（这里 `A` 和 `B` 已按字母顺序排列，所以不变）。

---

### 2. `sort_values`
- **作用**：按照某一列或多列的值（value）对 Data Frame 或 Series 进行排序。
- **适用于**：当你想基于数据内容排序时。

#### 语法
```python
DataFrame.sort_values(by, axis=0, ascending=True, inplace=False, ...)
Series.sort_values(ascending=True, inplace=False, ...)
```
- **`by`**：指定排序的列名（对 Data Frame）或列名列表。可以是字符串或字符串列表。
- **`axis`**：排序轴，`0`（默认）按列值排序，`1` 按行值排序。
- **`ascending`**：是否升序，`True`（默认）为升序，`False` 为降序。
- **`inplace`**：是否修改原数据。

#### 示例
```python
import pandas as pd

# 创建一个 DataFrame
df = pd.DataFrame({
    "A": [3, 1, 2],
    "B": [6, 4, 5]
}, index=[0, 1, 2])

print("原始 DataFrame:")
print(df)

# 按 A 列排序
print("\n按 A 列排序:")
print(df.sort_values(by="A"))

# 按 B 列降序排序
print("\n按 B 列降序排序:")
print(df.sort_values(by="B", ascending=False))

# 按多列排序
print("\n按 A 和 B 排序:")
print(df.sort_values(by=["A", "B"]))
```

输出：
```
原始 DataFrame:
   A  B
0  3  6
1  1  4
2  2  5

按 A 列排序:
   A  B
1  1  4
2  2  5
0  3  6

按 B 列降序排序:
   A  B
0  3  6
2  2  5
1  1  4

按 A 和 B 排序:
   A  B
1  1  4
2  2  5
0  3  6
```
- **解释**：
  - `by="A"`：按 `A` 列值升序排列。
  - `by="B", ascending=False`：按 `B` 列值降序排列。
  - `by=["A", "B"]`：先按 `A` 排序，`A` 相同时再按 `B` 排序。

---

### 主要区别
| 特性         | `sort_index`       | `sort_values`      |
| ------------ | ------------------ | ------------------ |
| **排序依据** | 索引（index）      | 数据值（values）   |
| **参数**     | 无需指定列名       | 需要 `by` 指定列名 |
| **适用场景** | 调整索引顺序       | 按内容排序         |
| **默认轴**   | `axis=0`（行索引） | `axis=0`（按列值） |

---

### 注意事项
1. **缺失值（Nan）**：
   
   - 默认情况下，`NaN` 会被排到最后，可以用 `na_position="first"` 将其排到开头。
   ```python
   df = pd.DataFrame({"A": [1, None, 3]})
   print(df.sort_values(by="A", na_position="first"))
   ```
   输出：
   ```
      A
   1  NaN
   0  1.0
   2  3.0
   ```
   
2. **Series vs Data Frame**：
   
   - Series 只有一列数据，`sort_index` 和 `sort_values` 直接作用于索引或值。
   - Data Frame 需要指定 `by`（`sort_values`）或 `axis`。
   
3. **性能**：
   - 对于大数据集，排序可能是耗时操作，尤其是多列排序。

如果你有具体代码或数据想用这两个方法处理，可以给我看，我帮你分析和实现！

### 小结

- **基本操作**：`head()`, `tail()`, `info()`, `describe()` 等方法帮助你快速了解数据。
- **选择和筛选**：使用 `.loc[]`, `.iloc[]`, 条件筛选来提取数据。
- **缺失值处理**：使用 `isnull()`, `fillna()`, `dropna()` 等方法处理缺失值。
- **合并和连接**：`merge()`, `concat()` 等方法可以帮助你合并多个数据集。
- **分组和聚合**：`groupby()` 和 `pivot_table()` 可以对数据进行分组、聚合和透视。
- **时间序列**：`pd.to_datetime()`, `pd.date_range()` 等方法帮助你处理时间数据。

掌握了这些基本操作，你就能高效地使用 `pandas` 进行数据分析、清洗和处理了。

## 14.dropna方法

在 pandas 中，`dropna()` 是一个非常实用的方法，用于处理 DataFrame 或 Series 中的缺失值（NaN）。让我为你详细解释一下它的用法：

### 基本用法
`dropna()` 会删除包含缺失值的行或列，默认情况下是删除任何包含 NaN 的行。

#### 语法
```python
DataFrame.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
```

#### 参数说明
1. **`axis`**:
   - `0` 或 `'index'`：删除包含 NaN 的行（默认）。
   - `1` 或 `'columns'`：删除包含 NaN 的列。
   
2. **`how`**:
   - `'any'`：只要有 NaN 就删除该行或列（默认）。
   - `'all'`：只有当整行或整列都是 NaN 时才删除。

3. **`thresh`**:
   - 指定一个整数，表示至少需要多少个非 NaN 值才能保留该行或列。

4. **`subset`**:
   - 指定特定的列或行标签，只在这些子集中检查 NaN。

5. **`inplace`**:
   - `False`：返回新的 DataFrame，原数据不变（默认）。
   - `True`：直接修改原 DataFrame。

### 示例
假设你有以下 DataFrame：
```python
import pandas as pd

data = {'A': [1, 2, None], 'B': [4, None, 6], 'C': [7, 8, 9]}
df = pd.DataFrame(data)
print(df)
```
输出：
```
     A    B  C
0  1.0  4.0  7
1  2.0  NaN  8
2  NaN  6.0  9
```

#### 1. 默认删除包含 NaN 的行
```python
df.dropna()
```
输出：
```
     A    B  C
0  1.0  4.0  7
```

#### 2. 删除包含 NaN 的列
```python
df.dropna(axis=1)
```
输出：
```
   C
0  7
1  8
2  9
```

#### 3. 只删除全为 NaN 的行
```python
df.dropna(how='all')
```
输出：
```
     A    B  C
0  1.0  4.0  7
1  2.0  NaN  8
2  NaN  6.0  9
```

#### 4. 指定子集
```python
df.dropna(subset=['A'])
```
输出：
```
     A    B  C
0  1.0  4.0  7
1  2.0  NaN  8
```

注意事项

- 如果数据中有大量缺失值，使用 `dropna()` 可能会导致数据丢失较多，可以考虑用 `fillna()` 填充缺失值作为替代方案。
- 你可以用 `df.isna()` 或 `df.isnull()` 检查缺失值的位置。

如果你有具体的 pandas 数据问题或代码需要帮助，随时告诉我，我可以用中文详细解答！

## 15、contact方法

以下是用 pandas 进行连接（concatenation）的 `concat()` 函数语法，用于合并多个 DataFrame 或 Series，不含示例：

### `pandas.concat()` 语法
```python
pd.concat(objs, axis=0, join='outer', ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, sort=False, copy=True)
```

#### 参数说明
- **`objs`**：要合并的对象列表或字典（如 `[df1, df2 ]`）。
- **`axis`**：合并方向，`0`（按行，垂直堆叠）或 `1`（按列，水平拼接）。
- **`join`**：合并方式，`'outer'`（并集）或 `'inner'`（交集）。
- **`ignore_index`**：是否重置索引，`True` 或 `False`。
- **`keys`**：为合并结果添加层次索引（如 `[‘group1’, ‘group2’]`）。
- **`verify_integrity`**：检查索引是否重复，`True` 时若重复则报错。
- **`sort`**：对非合并轴排序，`True` 或 `False`。
- **`copy`**：是否复制数据，`True` 或 `False`。

### 注意
- 需要导入 pandas：`import pandas as pd`。
- 可用于 DataFrame 或 Series。

## 16、merge方法

```python
merged_df = pd.merge(df1, df2, on='key_column', how='inner')
```

### 参数说明
- **df1, df2**: 要合并的两个数据框。
- **on**: 合并时使用的共同列名（键）。
- **how**: 合并方式，可选：
  -  `'inner'`: 内连接，仅保留键匹配的行。
  - `'outer'`: 外连接，保留所有行，未匹配处填 NaN。
  - `'left'`: 左连接，保留左数据框所有行。
  - `'right'`: 右连接，保留右数据框所有行。
- 是的，pandas 的 `merge` 函数还有其他参数。

  - **left_on**: 左数据框中用作键的列名（当左右键名不同时使用）。
  - **right_on**: 右数据框中用作键的列名（当左右键名不同时使用）。
  - **left_index**: 如果为 True，用左数据框的索引作为键。
  - **right_index**: 如果为 True，用右数据框的索引作为键。
  - **suffixes**: 当两数据框有相同列名时，添加的后缀，默认是 `('_x', '_y')`。 
  - **indicator**: 如果为 True，添加一列 `_merge`，显示每行数据的来源（`left_only`, `right_only`, `both`）。
  - **validate**: 检查合并的键是否符合预期关系，可选值包括：
    - `'one_to_one'`: 一对一。
    - `'one_to_many'`: 一对多。
    - `'many_to_one'`: 多对一。
    - `'many_to_many'`: 多对多。

# 3、matplotlib

Matplotlib 是 Python 中最常用的可视化库，提供了丰富的函数和功能，用于创建各种类型的图表，包括折线图、散点图、柱状图、等高线图等。以下是 Matplotlib 常用的函数和功能的整理，专注于核心绘图工具、图形设置和常见用途，保持清晰简洁。

---

### 1. 基本绘图函数

这些函数用于创建主要图形类型。

- **`plt.plot(x, y, fmt, **kwargs)`**

  - **功能**：绘制折线图或散点图（取决于格式字符串 `fmt`）。

  - **用途**：展示连续数据趋势或点分布。

  - **参数**：

    - `x, y`：x 和 y 坐标。
    - `fmt`：样式，如 `'b-'`（蓝色实线）、`'ro'`（红色圆点）。
    - `markersize`, `linewidth`, `color` 等。

  - **示例**：

    ```python
    plt.plot([1, 2, 3], [4, 5, 6], 'b--', linewidth=2)
    ```

- **`plt.scatter(x, y, s=None, c=None, marker=None, **kwargs)`**

  - **功能**：绘制散点图，支持动态大小和颜色。

  - **用途**：展示数据点分布，分类或聚类可视化。

  - **参数**：

    - `s`：点大小（标量或数组）。
    - `c`：颜色（标量、数组或颜色名称）。
    - `marker`：标记形状（如 `'o'`, `'x'`）。
    - `alpha`：透明度。

  - **示例**：

    ```python
    plt.scatter(x, y, s=50, c='red', alpha=0.5)
    ```

- **`plt.bar(x, height, width=0.8, **kwargs)`**

  - **功能**：绘制柱状图。

  - **用途**：比较类别数据或统计结果。

  - **参数**：

    - `x`：柱子位置。
    - `height`：柱子高度。
    - `width`：柱子宽度。
    - `color`, `edgecolor`, `align` 等。

  - **示例**：

    ```python
    plt.bar(['A', 'B', 'C'], [10, 20, 15], color='skyblue')
    ```

- **`plt.hist(x, bins=None, **kwargs)`**

  - **功能**：绘制直方图。

  - **用途**：展示数据分布或频率。

  - **参数**：

    - `x`：输入数据。
    - `bins`：分组数量或边界。
    - `density`：是否归一化。
    - `color`, `alpha` 等。

  - **示例**：

    ```python
    plt.hist(data, bins=30, color='green', alpha=0.7)
    ```

- **`plt.contour(X, Y, Z, levels=None, **kwargs)`**

  - **功能**：绘制等高线图（无填充）。

  - **用途**：展示二维网格数据的轮廓，如分类边界。

  - **参数**：

    - `X, Y`：网格坐标。
    - `Z`：网格值。
    - `levels`：等高线级别。
    - `colors`, `linewidths`, `cmap`。

  - **示例**：

    ```python
    plt.contour(X, Y, Z, levels=5, colors='k')
    ```

- **`plt.contourf(X, Y, Z, levels=None, **kwargs)`**

  - **功能**：绘制填充等高线图。

  - **用途**：显示网格值的颜色区域，如决策边界。

  - **参数**：

    - 同 `plt.contour`，另加 `cmap`（颜色映射）、`alpha`。

  - **示例**：

    ```python
    plt.contourf(X, Y, Z, cmap='Pastel2')
    ```

- **`plt.imshow(X, cmap=None, **kwargs)`**

  - **功能**：绘制图像或二维数组。

  - **用途**：显示矩阵数据、图像或热图。

  - **参数**：

    - `X`：二维数组。
    - `cmap`：颜色映射（如 `'viridis'`, `'hot'`）。
    - `interpolation`：插值方式（如 `'nearest'`）。

  - **示例**：

    ```python
    plt.imshow(matrix, cmap='gray')
    ```

- **`plt.pie(x, labels=None, **kwargs)`**

  - **功能**：绘制饼图。

  - **用途**：展示比例或百分比。

  - **参数**：

    - `x`：各部分值。
    - `labels`：标签。
    - `colors`, `autopct`（百分比格式）。

  - **示例**：

    ```python
    plt.pie([30, 40, 30], labels=['A', 'B', 'C'], autopct='%1.1f%%')
    ```

---

### 2. 图形布局与管理

这些函数用于组织图形和子图。

- **`plt.figure(figsize=None, **kwargs)`**

  - **功能**：创建新图形窗口。

  - **用途**：设置图形大小和属性。

  - **参数**：

    - `figsize`：宽度和高度（英寸），如 `(8, 6)`。
    - `dpi`：分辨率。

  - **示例**：

    ```python
    plt.figure(figsize=(10, 5))
    ```

- **`plt.subplot(nrows, ncols, index, **kwargs)`**

  - **功能**：创建并选择子图。

  - **用途**：在网格布局中绘制多个图表。

  - **参数**：

    - `nrows, ncols`：网格行数和列数。
    - `index`：子图位置（从 1 开始）。
    - `sharex`, `sharey`：共享轴。

  - **示例**：

    ```python
    plt.subplot(2, 1, 1)
    plt.plot([1, 2, 3])
    ```

- **`plt.subplots(nrows=1, ncols=1, figsize=None, **kwargs)`**

  - **功能**：一次性创建图形和子图数组。

  - **用途**：面向对象绘图，管理多子图。

  - **返回**：`Figure` 和 `Axes`（单个或数组）。

  - **示例**：

    ```python
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].plot([1, 2, 3])
    ```

- **`plt.tight_layout()`**

  - **功能**：自动调整子图间距。

  - **用途**：防止标题或标签重叠。

  - **示例**：

    ```python
    plt.tight_layout()
    ```

- **`plt.close(fig=None)`**

  - **功能**：关闭图形窗口。

  - **用途**：释放内存，防止图形堆积。

  - **参数**：

    - `fig`：指定图形，默认关闭当前。

  - **示例**：

    ```python
    plt.close()
    ```

---

### 3. 图形设置与装饰

这些函数用于调整图形外观和添加标注。

- **`plt.title(label, fontsize=None, **kwargs)`**

  - **功能**：设置图形标题。

  - **用途**：描述图形内容。

  - **示例**：

    ```python
    plt.title('My Plot', fontsize=16)
    ```

- **`plt.xlabel(label, fontsize=None, **kwargs)`**

  - **功能**：设置 x 轴标签。

  - **用途**：标注 x 轴含义。

  - **示例**：

    ```python
    plt.xlabel('X Axis', fontsize=12)
    ```

- **`plt.ylabel(label, fontsize=None, **kwargs)`**

  - **功能**：设置 y 轴标签。

  - **用途**：标注 y 轴含义。

  - **示例**：

    ```python
    plt.ylabel('Y Axis', fontsize=12)
    ```

- **`plt.legend(loc='best', **kwargs)`**

  - **功能**：显示图例。

  - **用途**：标注不同数据系列。

  - **参数**：

    - `loc`：位置，如 `'upper right'`, `'best'`。
    - `labels`：自定义标签。

  - **示例**：

    ```python
    plt.plot(x, y, label='Data')
    plt.legend()
    ```

- **`plt.grid(True, **kwargs)`**

  - **功能**：显示网格线。

  - **用途**：辅助数据读取。

  - **参数**：

    - `which`：`'major'`, `'minor'`。
    - `color`, `linestyle`。

  - **示例**：

    ```python
    plt.grid(True, linestyle='--', alpha=0.7)
    ```

- **`plt.colorbar(mappable=None, **kwargs)`**

  - **功能**：添加颜色条。

  - **用途**：显示颜色映射的值范围。

  - **示例**：

    ```python
    plt.contourf(X, Y, Z)
    plt.colorbar()
    ```

- **`plt.clabel(contour, inline=True, **kwargs)`**

  - **功能**：为等高线添加数值标签。

  - **用途**：标注等高线值。

  - **示例**：

    ```python
    cs = plt.contour(X, Y, Z)
    plt.clabel(cs, fmt='%.1f')
    ```

- **`plt.tick_params(**kwargs)`**

  - **功能**：调整刻度标签和样式。

  - **用途**：隐藏或格式化刻度。

  - **参数**：

    - `labelbottom`, `labelleft`：显示/隐藏。
    - `length`, `width`, `color`。

  - **示例**：

    ```python
    plt.tick_params(labelbottom='off')
    ```

- **`plt.xlim(left=None, right=None)` / `plt.ylim(bottom=None, top=None)`**

  - **功能**：设置 x/y 轴范围。

  - **用途**：控制显示区域。

  - **示例**：

    ```python
    plt.xlim(0, 10)
    plt.ylim(-1, 1)
    ```

---

### 4. 数据处理与辅助函数

这些函数用于数据准备或绘图辅助。

- **`plt.text(x, y, s, **kwargs)`**

  - **功能**：在指定位置添加文本。

  - **用途**：标注数据点或说明。

  - **示例**：

    ```python
    plt.text(1, 2, 'Point A', fontsize=12)
    ```

- **`plt.annotate(s, xy, xytext=None, arrowprops=None, **kwargs)`**

  - **功能**：添加带箭头的注释。

  - **用途**：指向特定点并描述。

  - **示例**：

    ```python
    plt.annotate('Max', xy=(2, 3), xytext=(3, 4), arrowprops={'arrowstyle': '->'})
    ```

- **`plt.savefig(fname, dpi=None, **kwargs)`**

  - **功能**：保存图形到文件。

  - **用途**：导出 PNG、PDF 等格式。

  - **参数**：

    - `fname`：文件名（如 `'plot.png'`）。
    - `dpi`：分辨率。
    - `bbox_inches='tight'`：调整边距。

  - **示例**：

    ```python
    plt.savefig('output.png', dpi=300)
    ```

- **`plt.show()`**

  - **功能**：显示图形。

  - **用途**：渲染所有绘图。

  - **示例**：

    ```python
    plt.plot([1, 2, 3])
    plt.show()
    ```

---

### 5. 高级可视化

这些功能用于复杂或特定场景。

- **`plt.boxplot(x, **kwargs)`**

  - **功能**：绘制箱线图。

  - **用途**：展示数据分布和异常值。

  - **示例**：

    ```python
    plt.boxplot([data1, data2], labels=['A', 'B'])
    ```

- **`plt.errorbar(x, y, yerr=None, xerr=None, **kwargs)`**

  - **功能**：绘制带误差条的图。

  - **用途**：显示数据的不确定性。

  - **示例**：

    ```python
    plt.errorbar(x, y, yerr=0.1, fmt='o')
    ```

- **`plt.fill_between(x, y1, y2=0, **kwargs)`**

  - **功能**：填充两条曲线间的区域。

  - **用途**：高亮范围或区域。

  - **示例**：

    ```python
    plt.fill_between(x, y, 0, alpha=0.3)
    ```

- **`plt.streamplot(x, y, u, v, **kwargs)`**

  - **功能**：绘制流线图。

  - **用途**：展示向量场。

  - **示例**：

    ```python
    plt.streamplot(X, Y, U, V, color='b')
    ```

---

### 6. 常用功能与技巧

- **颜色管理**：

  - 颜色名称：`'red'`, `'blue'`, `'k'`（黑色）。

  - 十六进制：`'#ff0000'`。

  - 颜色映射：`plt.cm.viridis`, `plt.cm.Pastel2`。

  - 示例：

    ```python
    plt.scatter(x, y, c='tab:blue')
    ```

- **样式设置**：

  - 使用预定义样式：

    ```python
    plt.style.use('ggplot')
    ```

  - 自定义样式：

    ```python
    plt.rcParams['font.size'] = 12
    ```

- **多图叠加**：

  - 多次调用绘图函数（如 `plt.plot`, `plt.scatter`）叠加内容。

  - 示例：

    ```python
    plt.plot(x, y1, 'b-')
    plt.scatter(x, y2, c='r')
    ```

- **动态绘图**：

  - 使用 `FuncAnimation` 实现动画。

  - 示例：

    ```python
    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(fig, update, frames=100)
    ```

- **三维绘图**：

  - 使用 `mpl_toolkits.mplot3d`：

    ```python
    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.subplot(projection='3d')
    ax.plot3D(x, y, z)
    ```

---

### 总结

- **核心绘图**：
  - `plot`, `scatter`, `bar`, `hist`, `contour`, `contourf`, `imshow`, `pie`。
- **布局管理**：
  - `figure`, `subplot`, `subplots`, `tight_layout`, `close`。
- **图形装饰**：
  - `title`, `xlabel`, `ylabel`, `legend`, `grid`, `colorbar`, `clabel`, `tick_params`, `xlim`, `ylim`.
- **辅助功能**：
  - `text`, `annotate`, `savefig`, `show`.
- **高级绘图**：
  - `boxplot`, `errorbar`, `fill_between`, `streamplot`, 3D 绘图，动画。
- **用途**：
  - 数据分析、机器学习、科学可视化、报告制作。

如果你对某个函数或功能有具体疑问（比如高级用法、优化或案例

# 4、python中的导包

### 1、ImportError: attempted relative import beyond top-level packag
发生原因该错误通常发生在以下情况：

1. 直接运行子包内的脚本：如果尝试直接运行位于包内（而非顶级包）的脚本，Python不会将该目录视为包，因此相对导入会失败。

2. 没有正确设置包结构：确保每个包（包括子包）都有一个__init__.py文件，这是Python识别目录为包的标志。
3. 相对导入超出顶级包：如错误代码示范所示，尝试使用两个点(..)的相对导入表示向上一级目录寻找模块，但如果执行环境不在包的上下文中，这样的导入会失败。

# 5、jupyter中的快捷键或者快捷指令

在 **Jupyter Notebook** 或 **JupyterLab** 中，使用快捷键可以提高工作效率。下面是一些常用的快捷键和指令，分为 **命令模式** 和 **编辑模式**。

### 1. **命令模式快捷键**

命令模式是你选中一个单元格而没有进入单元格的编辑状态时的模式。此时，你可以对单元格执行诸如删除、复制、粘贴等操作。

- **Enter**：进入编辑模式（在当前单元格中编辑代码）。
- **Esc**：进入命令模式（退出编辑模式）。
- **Shift + Enter**：运行当前单元格并选择下一个单元格。
- **Ctrl + Enter**：运行当前单元格并保持焦点在当前单元格上。
- **Alt + Enter**：运行当前单元格并在下方插入一个新单元格。
- **A**：在当前单元格上方插入一个新单元格。
- **B**：在当前单元格下方插入一个新单元格。
- **X**：剪切当前单元格。
- **C**：复制当前单元格。
- **V**：粘贴剪切或复制的单元格。
- **D, D**（按两次 D）：删除当前单元格。
- **Z**：撤销删除操作。
- **M**：将当前单元格的类型改为 Markdown。
- **Y**：将当前单元格的类型改为 Code（代码）。
- **Shift + M**：合并选中的多个单元格。
- **Ctrl + S**：保存当前 Notebook。
- **Shift + Up/Down**：选择多个单元格（上下箭头选择多个单元格）。
- **L**：显示/隐藏行号（仅适用于代码单元格）。
- **H**：显示所有快捷键帮助。
- **I, I**（按两次 I）：中断当前单元格的执行。

### 2. **编辑模式快捷键**

编辑模式是你点击进入代码单元格进行代码编辑时的模式。你可以在此模式下修改单元格内容。

- **Tab**：代码补全（自动补全代码、函数名、变量名等）。
- **Shift + Tab**：显示函数或方法的文档说明（如果光标在函数或方法名上）。
- **Ctrl + /**：注释或取消注释选中的行（使用 `#`）。
- **Ctrl + Shift + -**：在当前光标位置拆分代码单元格。
- **Ctrl + Z**：撤销上一步的编辑操作。
- **Ctrl + Y**：重做上一步的编辑操作。

### 3. **其他常用操作**

- **Shift + Space**：向上滚动一个屏幕（页面）。
- **Space**：向下滚动一个屏幕（页面）。
- **Ctrl + F**：在当前 Notebook 中搜索文本。
- **Ctrl + G**：跳转到匹配的下一个搜索结果。

### 4. **Markdown 常用快捷键**

在 Jupyter Notebook 中，Markdown 格式用于写文档说明或注释。

- **#**：一级标题（例如 `# 标题1`）。
- **##**：二级标题（例如 `## 标题2`）。
- **###**：三级标题（例如 `### 标题3`）。
- **- 或 \*（星号）**：无序列表（例如 `- 项目 1` 或 `* 项目 1`）。
- **1. 2. 3.**：有序列表（例如 `1. 第一项`）。
- **[链接文字](https://chatgpt.com/c/链接地址)**：插入链接。
- **`代码块`**：行内代码（例如 ``print()``）。
- **`代码块`**：多行代码块（例如 ` ```python` 写代码，````` 结束代码块）。
- ***斜体\*** 或 ***斜体\***：斜体。
- ***\*粗体\**** 或 ***\*粗体\****：粗体。
- **`~~删除线~~`**：删除线。

### 5. **Cell 魔法命令**

Jupyter Notebook 还支持一些魔法命令，可以方便地执行一些特定的操作，通常以 `%` 或 `%%` 开头。

- **%time**：计时执行一个语句的运行时间（例如 `%time x = sum(range(1000))`）。
- **%timeit**：重复执行语句并返回平均时间（例如 `%timeit x = sum(range(1000))`）。
- **%run <file.py>**：运行 Python 脚本文件（例如 `%run script.py`）。
- **%matplotlib inline**：使得 matplotlib 绘制的图像显示在 Notebook 内。
- **%load <file.py>**：加载 Python 脚本文件到单元格中（例如 `%load script.py`）。
- **%who**：查看当前 Notebook 中所有的变量。
- **%reset**：清除当前 Notebook 的所有变量和状态。

# 6、skit-learn

## 1、混淆矩阵基础

在二分类任务中（例如判断数字是否为 5，y_train_5 是 True 或 False），混淆矩阵用来评估模型的预测结果。它包含以下四个元素：

1. TP（True Positive，真正例）

   ：

   - 实际为正类（True，即数字 5），模型也预测为正类（True）。
   - 示例：一个样本的真实标签是 5，模型预测也是 5。

2. TN（True Negative，真负例）

   ：

   - 实际为负类（False，即非 5），模型也预测为负类（False）。
   - 示例：一个样本的真实标签是 3，模型预测不是 5。

3. FP（False Positive，假正例）

   ：

   - 实际为负类（False，即非 5），模型错误预测为正类（True）。
   - 示例：一个样本的真实标签是 7，模型预测是 5。

4. FN（False Negative，假负例）

   ：

   - 实际为正类（True，即 5），模型错误预测为负类（False）。
   - 示例：一个样本的真实标签是 5，模型预测不是 5。

## 2、常见参数

1. 学习模型参数（fit）。
2. 预测新数据（predict）。
3. 评估性能（score 或 cross_val_score）。
4. 

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



# 7、神经网络的基本参数

损失函数指导方向，学习率控制步长，初始化器提供起点，正则化器防止过度拟合。

---

### 1. 什么是神经网络？
神经网络是一种模仿人脑神经元工作方式的计算模型，用于处理复杂的模式识别任务（如图像分类）。它由多层节点（神经元）组成，通过权重、偏置和激活函数将输入数据逐步转换为输出（预测结果）。

在您提供的代码中，神经网络用于将 32x32 像素的图像（展平为 3072 维向量）分类为特定类别（如猫、狗）。网络包含输入层、隐藏层和输出层，通过训练调整参数以提高分类准确率。

---

### 2. 激活函数
#### 定义
激活函数（Activation Function）是神经网络中每个神经元的非线性变换函数，决定该神经元是否“激活”以及输出什么值。它引入非线性，使神经网络能够解决复杂问题（如图像分类）。

#### 常见激活函数
在您的代码中使用了以下激活函数：
1. **ReLU（Rectified Linear Unit）**：
   - 公式：`f(x) = max(0, x)`
   - 作用：将负输入置为 0，正输入保持不变。简单高效，加速收敛，缓解梯度消失问题。
   - 代码示例：
     ```python
     model.add(Dense(512, activation="relu"))
     ```
   - 适用场景：隐藏层（如代码中的 512 和 256 单元层），常用于图像处理。

2. **Softmax**：
   - 公式：对于输入向量 \( x = [x_1, x_2, ..., x_n] \)，输出第 \( i \) 个类别的概率：
     \[
     \text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}
     \]
   - 作用：将输出转换为概率分布，总和为 1，适合多分类任务。
   - 代码示例：
     ```python
     model.add(Dense(len(lb.classes_), activation="softmax"))
     ```
   - 适用场景：输出层（如代码中用于预测类别）。

#### 其他常见激活函数
- **Sigmoid**：输出 [0, 1]，适合二分类，但易导致梯度消失。
- **Tanh**：输出 [-1, 1]，比 Sigmoid 更适合隐藏层，但计算复杂。
- **Leaky ReLU**：改进 ReLU，允许负输入有小斜率，避免“神经元死亡”。

#### 为什么需要激活函数？
- **非线性**：没有激活函数，神经网络只是线性变换，无法处理复杂模式（如图像特征）。
- **控制输出**：激活函数决定哪些神经元对结果贡献更大。

---

### 3. 全连接层
#### 定义
全连接层（Fully Connected Layer，或 Dense 层）是神经网络中一种基本层类型，其中每个输入神经元与每个输出神经元都有连接。每个连接有一个权重，输出通过激活函数处理。

#### 在代码中的体现
您的代码使用 `Dense` 层构建全连接网络：
```python
model.add(Dense(512, input_shape=(3072,), activation="relu"))
model.add(Dense(256, activation="relu"))
model.add(Dense(len(lb.classes_), activation="softmax"))
```
- **输入层**：接收 3072 维向量（32x32x3 的展平图像）。
- **隐藏层 1**：512 个神经元，每个神经元接收 3072 个输入，输出经过 ReLU 激活。
- **隐藏层 2**：256 个神经元，接收前一层的 512 个输出。
- **输出层**：神经元数等于类别数，输出概率分布。

#### 全连接层的计算
对于一个全连接层：
- 输入：向量 \( x \)（维度 \( n \)）。
- 输出：向量 \( y \)（维度 \( m \)）。
- 计算公式：
  \[
  y = \text{Activation}(W \cdot x + b)
  \]
  - \( W \): 权重矩阵（维度 \( m \times n \)）。
  - \( b \): 偏置向量（维度 \( m \)）。
  - \(\text{Activation}\): 激活函数（如 ReLU 或 Softmax）。

#### 特点
- **参数多**：全连接层的参数量为 \( n \times m + m \)。例如，代码中第一层有 \( 3072 \times 512 + 512 \approx 1.57 \) 百万个参数。
- **适合简单任务**：但对图像分类，卷积层（CNN）通常更高效，因为它们捕捉空间特征。

---

### 4. 神经网络的基本参数概念
神经网络的性能和行为由以下参数和超参数控制：

#### （1）权重（Weights）
- **定义**：每个连接的强度，决定输入对输出的影响。
- **初始化**：代码中使用 `TruncatedNormal` 初始化权重（均值 0，标准差 0.05）：
  ```python
  kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.05)
  ```
- **作用**：通过训练调整权重以最小化损失函数。
- **优化**：权重通过反向传播和优化器（如 SGD）更新。

#### （2）偏置（Bias）
- **定义**：每个神经元的偏移量，帮助模型拟合数据。
- **初始化**：通常初始化为 0 或小值，代码中未显式指定（Keras 默认 0）。
- **作用**：增加模型灵活性，避免输出始终通过原点。

#### （3）学习率（Learning Rate）
- **定义**：控制权重更新的步长。
- **代码中**：
  ```python
  INIT_LR = 0.001
  opt = SGD(learning_rate=INIT_LR)
  ```
- **作用**：学习率太高可能导致震荡，太低则收敛慢。代码中 0.001 对于 SGD 较小，适合稳定训练。

#### （4）损失函数（Loss Function）
- **定义**：衡量模型预测与真实标签的差距。
- **代码中**：
  ```python
  model.compile(loss="categorical_crossentropy", ...)
  ```
- **分类交叉熵**（Categorical Crossentropy）：用于多分类任务，计算预测概率与真实标签的差异。
- **作用**：训练目标是最小化损失函数。

#### （5）优化器（Optimizer）
- **定义**：调整权重以最小化损失的算法。
- **代码中**：使用 SGD（随机梯度下降）：
  ```python
  opt = SGD(learning_rate=INIT_LR)
  ```
- **其他优化器**：Adam、RMSprop 等通常收敛更快，可能比 SGD 更适合图像任务。

#### （6）正则化（Regularization）
- **定义**：防止过拟合的技术。
- **代码中**：
  - **L2 正则化**：对权重施加惩罚，防止权重过大：
    ```python
    kernel_regularizer=regularizers.l2(0.01)
    ```
  - **Dropout**：随机丢弃神经元（概率 0.5），增强泛化能力：
    ```python
    model.add(Dropout(0.5))
    ```

#### （7）Epoch 和 Batch Size
- **Epoch**：整个训练数据集经过一次前向和反向传播的次数。代码中：
  ```python
  EPOCHS = 2000
  ```
  - 2000 可能过多，建议用早停机制。
- **Batch Size**：每次训练处理的样本数。代码中：
  ```python
  batch_size=32
  ```
  - 32 是常见选择，平衡内存和收敛速度。

#### （8）评估指标（Metrics）
- **定义**：衡量模型性能的指标。
- **代码中**：
  
  ```python
  model.compile(..., metrics=["accuracy"])
  ```
- **准确率**（Accuracy）：预测正确的样本比例。
- **分类报告**：提供精确率、召回率、F1 分数：
  ```python
  print(classification_report(...))
  ```

---

#### 5、结合代码的例子

以下是代码中神经网络的结构和参数如何协同工作：
1. **输入**：32x32 像素 RGB 图像，展平为 3072 维向量。
2. **第一层（Dense 512, ReLU）**：
   - 输入 3072 维，输出 512 维。
   - 参数量：\( 3072 \times 512 + 512 \approx 1.57 \) 百万。
   - ReLU 激活：将负值置为 0。
   - Dropout (0.5)：随机丢弃 50% 的输出。
3. **第二层（Dense 256, ReLU）**：
   - 输入 512 维，输出 256 维。
   - 参数量：\( 512 \times 256 + 256 \approx 0.13 \) 百万。
4. **输出层（Dense num_classes, Softmax）**：
   - 输出类别数的概率分布。
   - 例如，若有 3 个类别，参数量为 \( 256 \times 3 + 3 = 771 \)。

**训练过程**：
- 使用 SGD 优化器，学习率 0.001，更新权重以最小化分类交叉熵损失。
- 每 32 个样本（batch_size）更新一次权重，重复 2000 次（epochs）。
- L2 正则化和 Dropout 防止过拟合。

---

#### 6、常见问题解答

1. **为什么不用卷积层？**
   - 您的代码使用全连接层，适合简单任务，但对图像分类，卷积神经网络（CNN）更高效，因为它们捕捉空间特征。CNN 使用 `Conv2D` 层，参数量更少，效果更好。

2. **如何选择激活函数？**
   - 隐藏层：ReLU 是默认选择，简单高效。
   - 输出层：多分类用 Softmax，二分类用 Sigmoid。

3. **如何调整参数？**
   - **学习率**：尝试 0.01 或 0.0001，观察收敛。
   - **正则化**：L2 的 0.01 可能过强，试 0.001。
   - **Epoch**：使用早停（EarlyStopping）自动选择最佳 epoch。

4. **全连接层的局限性？**
   - 参数量大（代码中约 1.7 百万参数），计算成本高。
   - 对图像的空间结构不敏感，建议用 CNN。

---

### 5、激活函数的选取

激活函数是神经网络中引入非线性的关键元素，它们使网络能够学习复杂的模式。下面是深度学习中最常用的激活函数及其特点：

#### ReLU (修正线性单元)

**函数表达式**：f(x) = max(0, x)

**特点**：

- 计算效率高，仅需简单的阈值操作
- 对于正输入可以缓解梯度消失问题
- 产生稀疏激活（许多神经元输出为0），这可能有益于模型
- 在x=0处不可微，但实际应用中很少造成问题
- 隐藏层中最广泛使用的激活函数
- 输出范围：[0, ∞)

**局限性**：

- "死亡ReLU"问题 - 当神经元持续接收负输入时，可能永久性失活
- 输出不以零为中心，可能导致训练过程中的锯齿状动态

#### Leaky ReLU (渗漏ReLU)

**函数表达式**：f(x) = max(αx, x)，其中α通常是一个小值，如0.01

**特点**：

- 通过允许负输入有小梯度来解决死亡ReLU问题
- 保留ReLU的大部分计算优势
- 输出范围：(-∞, ∞)
- 除x=0外处处可微（与ReLU类似）

#### PReLU (参数化ReLU)

**函数表达式**：与Leaky ReLU类似，但α是一个可学习参数

**特点**：

- 允许模型学习负值的最佳斜率
- 可以适应数据，潜在地提高性能
- 比固定α值的Leaky ReLU更灵活

#### ELU (指数线性单元)

**函数表达式**：f(x) = x if x > 0 else α(e^x - 1)

**特点**：

- 对负值有软饱和效应，使其对噪声更鲁棒
- 输出均值接近于零，有助于加速学习
- 在所有点都可微
- 计算成本高于ReLU

#### SELU (缩放指数线性单元)

**函数表达式**：f(x) = λx if x > 0 else λα(e^x - 1)

**特点**：

- 自归一化性质，有助于深层网络的训练稳定性
- 被设计用来保持输入和输出的均值和方差
- 在某些情况下可以不使用批量归一化

#### Sigmoid

**函数表达式**：f(x) = 1 / (1 + e^(-x))

**特点**：

- 输出范围为(0, 1)，可解释为概率
- 在二元分类问题的输出层常用
- 处处可微
- 在输入绝对值较大时会饱和，导致梯度几乎为零

**局限性**：

- 容易引起梯度消失问题
- 输出不以零为中心
- 计算指数函数较为昂贵

#### Tanh (双曲正切)

**函数表达式**：f(x) = (e^x - e^(-x)) / (e^x + e^(-x))

**特点**：

- 输出范围为(-1, 1)，以零为中心
- 比Sigmoid在隐藏层表现更好
- 处处可微
- 仍然存在饱和问题

#### Softmax

**函数表达式**：f(x_i) = e^(x_i) / Σ(e^(x_j))，对所有j求和

**特点**：

- 将输入转换为概率分布（所有输出之和为1）
- 多分类问题输出层的标准选择
- 强调最大值，抑制其他值
- 与交叉熵损失函数配合良好

### 6、compile

在深度学习中，`compile()`是模型训练前的关键准备步骤，相当于为神经网络设置学习规则和目标。我来深入解释这个函数的作用、参数和内部机制。

#### compile()方法的本质

`compile()`方法本质上是在告诉模型三件重要的事：

1. 如何测量成功与失败（损失函数）
2. 如何改进自己（优化器）
3. 关注哪些性能指标（评估指标）

这就像是给学生制定学习计划：你需要知道考试如何评分（损失函数），采用什么学习方法（优化器），以及除了考试成绩外还要关注哪些能力的提升（评估指标）。

#### 主要参数详解

##### 1. 损失函数 (loss)

损失函数定义了模型预测值与真实值之间的差距计算方法。常见选项包括：

- `categorical_crossentropy`：适用于多分类问题，目标是one-hot编码形式
- `sparse_categorical_crossentropy`：也用于多分类，但目标是整数标签而非one-hot向量
- `binary_crossentropy`：二元分类问题的标准选择
- `mean_squared_error`：回归问题的常用损失函数
- `mean_absolute_error`：对异常值不那么敏感的回归损失函数

每种损失函数都有其适用场景。例如，你的代码中使用的`categorical_crossentropy`专为多类别分类设计，非常适合与softmax激活函数配合使用。

##### 2. 优化器 (optimizer)

优化器决定了如何基于损失函数的梯度更新模型权重。常见的优化器包括：

- `SGD`：最基础的随机梯度下降，可以配置动量和学习率衰减
- `Adam`：自适应学习率的优化器，结合了动量和RMSprop的优势
- `RMSprop`：自适应学习率方法，适合处理非平稳目标
- `Adagrad`：为不同参数自动调整学习率的方法

##### 3. 评估指标 (metrics)

评估指标用于监控训练和测试过程。与损失函数不同，它们通常不用于梯度更新，而是为了提供更直观的性能度量：

- `accuracy`：分类准确率
- `precision`：精确率
- `recall`：召回率
- `AUC`：ROC曲线下面积
- `mae`：平均绝对误差（回归问题）

这些指标会在训练期间定期计算并显示，帮助你了解模型学习进展。

#### compile()内部工作机制

当你调用`compile()`时，Keras会：

1. **构建计算图**：根据你的损失函数和优化器设置，准备好用于前向和反向传播的计算路径
2. **准备梯度计算**：设置自动微分所需的内部状态
3. **优化器初始化**：初始化优化器所需的状态变量（如动量缓存等）
4. **确定输出格式**：根据损失函数和度量指标，确定模型应输出什么样的预测结果

这就像是一位教练在训练开始前，确定训练计划、评价标准和改进策略。

#### compile()与fit()的区别

很多初学者会混淆`compile()`和`fit()`这两个函数：

- `compile()`是训练前的准备工作，定义学习规则
- `fit()`是实际执行训练的函数，将数据喂给模型并迭代更新权重

换句话说，`compile()`设定目标和方法，而`fit()`则是做实际工作的函数。

#### 一个类比：烹饪过程

想象你要烹饪一道菜：

- 构建模型（定义网络结构）相当于准备食材和厨具
- `compile()`相当于选择食谱和烹饪方法
- `fit()`相当于实际的烹饪过程

没有正确的`compile()`设置，模型就无法有效学习，就像没有清晰的食谱，再好的食材也难以做出美味佳肴。

#### 在你代码中的具体应用

```python
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
```

这行代码告诉你的神经网络：

- 使用分类交叉熵来衡量预测错误（适合多分类问题）
- 使用前面创建的SGD优化器来更新权重
- 除了损失值外，还要跟踪和报告准确率这一指标

这些设置与你模型的结构紧密相关：最后一层使用softmax激活函数，刚好与categorical_crossentropy损失函数匹配；而accuracy度量指标则适合分类任务的评估。

### 7、fit

`model.fit()` 方法是实际开始训练神经网络的命令，它启动了迭代学习过程，数据在网络中不断前向传播和反向传播，逐步调整权重以最小化损失函数。这个函数的工作原理可以比作一个学生通过反复练习和修正来掌握知识点。

#### 主要参数解释

1. **trainX, trainY**： 这是训练数据和对应的标签。`trainX` 包含输入特征（比如图像像素值），而 `trainY` 包含目标输出（比如类别的one-hot编码）。模型将学习从 `trainX` 到 `trainY` 的映射关系。
2. **validation_data=(testX, testY)**： 这是用于验证的数据集和标签，模型不会用它们来学习（调整权重），而是用来评估当前模型的泛化能力。验证集就像是"模拟考试"，帮你检验学习效果但不会直接影响学习过程。
3. **epochs=EPOCHS**： 轮次数，表示整个训练数据集被完整处理的次数。每个轮次，模型会查看所有训练样本一次。如果 `EPOCHS=20`，意味着整个训练集会被处理20次。这就像学生反复复习教材20遍，每遍都能加深理解。
4. **batch_size=32**： 批量大小，表示一次更新模型参数前处理的样本数。这里设置为32意味着模型会看32个样本，计算它们的平均梯度，然后更新一次参数。较大的批量提供更稳定但可能较慢的学习；较小的批量学习更快但波动更大。

#### 返回值 H

`H = model.fit(...)` 中的 `H` 是一个历史对象（History），它记录了训练过程中每个轮次（epoch）的损失值和指标值（如准确率）。H 是 model.fit 的返回值（History 对象），H.history 是一个字典，包含训练过程中的指标：

- "accuracy"：训练集准确率（每 epoch 的值）。
- "val_accuracy"：验证集准确率。
- "loss"：训练集损失（注释中未用）。
- "val_loss"：验证集损失（注释中未用）。

#### model.fit() 内部工作机制

当你调用 `model.fit()` 时，以下是内部发生的事情：

1. **数据准备**：将数据分成指定大小的批次

2. 对于每个轮次（epoch）

   ： a. 

   对于每个批次

   ：

   - 前向传播：输入数据通过网络，生成预测
   - 计算损失：比较预测与真实标签
   - 反向传播：计算梯度
   - 参数更新：优化器根据梯度更新权重 b. 完成所有批次后，在验证集上评估模型 c. 显示该轮次的训练和验证指标（损失值、准确率等） d. 更新历史对象

这就像学生的学习过程：阅读一些材料（批次）→ 尝试理解 → foo测验自己 → 调整理解方式 → 重复，直到完成整本书（一个轮次）→ 进行模拟考试（验证）→ 开始下一轮学习。

#### 一个深入的类比：驾驶培训

想象 `model.fit()` 就像学开车的过程：

- **训练数据**（trainX/trainY）是你练习的各种道路和正确的驾驶方式
- **验证数据**（testX/testY）是模拟考试路线
- **epochs** 是你练习驾驶的天数
- **batch_size** 是每次练习连续开车的时间（比如32分钟）
- **前向传播**是你的实际驾驶尝试
- **损失计算**是教练指出你的错误
- **反向传播和参数更新**是你根据反馈调整驾驶习惯
- **历史对象H**是记录你每天进步情况的学习日志

随着练习天数增加，你的驾驶技能应该提高，错误减少，但如果只在固定路线上反复练习（过拟合），在新路线（测试数据）上的表现可能不佳。

#### fit() 在深度学习工作流中的位置

在典型的深度学习工作流程中：

1. 准备数据（加载、预处理、划分训练/验证集）
2. 构建模型（定义层、激活函数等）
3. 编译模型（设置损失函数、优化器、指标）
4. **训练模型**（使用 `model.fit()`）← 你的代码在这里
5. 评估模型（在测试集上验证性能）
6. 预测和应用（部署模型进行实际预测）

`model.fit()` 是耗时最长、计算最密集的步骤，在大型数据集上可能需要数小时甚至数天。

#### 训练过程的监控和调整

当 `model.fit()` 运行时，你通常会看到如下输出：

```
Epoch 1/20
500/500 [==============================] - 2s 4ms/step - loss: 2.1847 - accuracy: 0.3254 - val_loss: 1.8932 - val_accuracy: 0.4102
Epoch 2/20
500/500 [==============================] - 2s 4ms/step - loss: 1.7539 - accuracy: 0.4872 - val_loss: 1.6243 - val_accuracy: 0.5230
...
```

这些输出告诉你：

- 当前轮次进度
- 训练损失和准确率
- 验证损失和准确率
- 每步（批次）处理时间

通过观察这些数值，你可以：

1. 确认模型正在学习（损失值下降）
2. 检测过拟合（训练指标改善但验证指标恶化）
3. 决定是否提前停止训练（验证指标停止改善）

### 8、predict 和 classification_report

这段代码是深度学习项目中的模型评估阶段，用于测试训练好的模型在未见数据上的表现并生成详细的性能报告。

```python
predictions = model.predict(testX, batch_size=32)
```

这行代码使用训练好的模型对测试数据进行预测。`model.predict()`函数接收测试数据`testX`，并返回模型的预测结果。与训练阶段类似，这里也设置了`batch_size=32`，表示一次处理32个样本，这样可以高效利用计算资源，特别是当测试数据量较大时。

返回的`predictions`是一个概率矩阵，其中每行对应一个测试样本，每列对应一个类别的预测概率。例如，如果有3个类别，一个样本的预测结果可能是[0.1, 0.7, 0.2]，表示模型认为这个样本有70%的概率属于第二类。

```python
print(classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1), target_names=lb.classes_))
```

这行代码生成并打印一个详细的分类报告，对模型性能进行全面评估。让我们分解其参数：

1. `testY.argmax(axis=1)`：将one-hot编码的真实标签转换回类别索引。例如，[0,1,0]变成1，表示第二类。
2. `predictions.argmax(axis=1)`：同样地，将模型预测的概率向量转换为类别索引，选择概率最高的类别。
3. `target_names=lb.classes_`：指定类别的名称，而不是仅显示索引号，使报告更具可读性。`lb.classes_`来自于标签二值化器（LabelBinarizer），包含了所有类别的名称。

#### 分类报告的理解

`classification_report`函数生成的报告包含以下关键指标：

1. **精确率(Precision)**：预测为某类的样本中，真正属于该类的比例。
   - 公式：TP/(TP+FP)
   - 高精确率意味着低误报率（很少将其他类错误地识别为此类）
2. **召回率(Recall)**：真正属于某类的样本中，被正确预测为该类的比例。
   - 公式：TP/(TP+FN)
   - 高召回率意味着低漏报率（很少漏掉真正属于此类的样本）
3. **F1分数**：精确率和召回率的调和平均值，提供了一个平衡的性能指标。
   - 公式：2 * (precision * recall) / (precision + recall)
   - 当类别不平衡时，F1比简单的准确率更有意义
4. **支持度(Support)**：每个类别的测试样本数量，帮助理解数据分布
5. **准确率(Accuracy)**：整体正确预测的比例，报告末尾会给出
6. **加权平均(Weighted Avg)**：考虑了各类别样本数量的指标平均值

#### 一个形象的类比

想象你是一位医生，使用一个诊断系统来判断患者是否患有某种疾病：

- **精确率** 回答的是："系统诊断为阳性的患者中，有多少真的患病？"这关系到不必要的治疗和焦虑。
- **召回率** 回答的是："真正患病的人中，系统能识别出多少？"这关系到漏诊的严重后果。
- **F1分数** 帮助你在这两者之间找到平衡点，特别是当疾病稀有但漏诊成本高时。

#### 实际输出示例及理解

分类报告的输出可能如下所示：

```
              precision    recall  f1-score   support

        猫       0.92      0.88      0.90       100
        狗       0.85      0.91      0.88       100
        鸟       0.94      0.82      0.88        50

    accuracy                          0.88       250
   macro avg       0.90      0.87      0.88       250
weighted avg       0.89      0.88      0.88       250
```

从这个报告中，我们可以得出以下洞见：

- 模型在识别猫时精确率很高(0.92)，意味着当它说"这是猫"时，大概率是对的
- 但它的召回率稍低(0.88)，表明有些猫被错误地归类为其他动物
- 对于狗，情况相反：召回率高(0.91)但精确率低(0.85)，意味着模型倾向于将其他动物误判为狗
- 鸟类有最高的精确率(0.94)但最低的召回率(0.82)，表明模型非常谨慎地预测鸟类，但因此会漏掉一些真正的鸟

### 9、卷积神经网络 CNN

这是卷积神经网络 (CNN) 的简短精辟流程：

1. **卷积 (Convolution):** 提取图像特征（边缘、纹理等）。
2. **激活 (Activation):** 引入非线性，使网络学习复杂模式（常用 ReLU）。
3. **池化 (Pooling):** 减小特征图尺寸，降低计算复杂度。
4. **重复 1-3:** 多次堆叠，逐层提取更抽象的特征。
5. **展平 (Flatten):** 将多维特征图转为一维向量。
6. **全连接 (Fully Connected):** 像传统神经网络一样，进行分类或回归。
7. **输出 (Output):** 得到最终结果（类别概率等）。

---



# 8、argparse

在深度学习中，`argparse` 是 Python 标准库中用于解析命令行参数的模块，广泛用于配置深度学习脚本的参数（如数据集路径、模型设置、超参数等）。以下是 `argparse` 在深度学习中的常用知识点总结，简洁清晰，涵盖核心用法和实践技巧。

---

### 1. 基本概念
- **作用**：通过命令行接收用户输入参数，使深度学习脚本灵活、可重用，无需修改代码即可更改配置。
- **核心类**：`argparse.ArgumentParser`，用于定义和解析参数。
- **典型场景**：
  - 指定数据集路径、模型保存路径。
  - 配置超参数（如学习率、批次大小、epoch 数）。
  - 控制训练/测试模式、设备选择（CPU/GPU）。

---

### 2. 常用功能与用法
#### （1）创建解析器
```python
import argparse
parser = argparse.ArgumentParser(description="Train a deep learning model")
```
- `description`：脚本用途描述，显示在帮助信息中（`python script.py -h`）。

#### （2）添加参数
使用 `parser.add_argument()` 定义参数：
```python
parser.add_argument('--data', type=str, required=True, help='Path to dataset')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
```
- **参数选项**：
  - **名称**：长选项（如 `--data`）以 `--` 开头，短选项（如 `-d`）以 `-` 开头。
  - **type**：指定参数类型（如 `str`、`int`、`float`）。
  - **required**：是否必须提供（`True`/`False`）。
  - **default**：默认值，参数未提供时使用。
  - **help**：参数描述，显示在帮助信息中。

#### （3）解析参数
```python
args = parser.parse_args()
```
- 返回 `Namespace` 对象，参数通过 `args.data`、`args.lr` 等访问。
- 转换为字典：
  ```python
  args_dict = vars(args)
  ```

#### （4）示例命令
```bash
python train.py --data ./dataset --lr 0.01 --epochs 50
```

---

### 3. 深度学习中的常见参数类型
1. **路径参数**：
   - 数据集路径、模型保存路径、日志路径。
   - 示例：
     ```python
     parser.add_argument('--data', type=str, required=True, help='Dataset path')
     parser.add_argument('--model', type=str, default='model.pth', help='Model save path')
     ```

2. **超参数**：
   - 学习率、批次大小、epoch 数、优化器类型。
   - 示例：
     ```python
     parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
     parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
     ```

3. **模型配置**：
   - 模型架构、层数、隐藏单元数。
   - 示例：
     ```python
     parser.add_argument('--model-type', type=str, choices=['resnet', 'vgg'], help='Model architecture')
     parser.add_argument('--hidden-units', type=int, default=512, help='Number of hidden units')
     ```

4. **训练选项**：
   - 训练/测试模式、设备（CPU/GPU）、是否恢复训练。
   - 示例：
     ```python
     parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train', help='Mode')
     parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
     parser.add_argument('--resume', type=str, help='Path to checkpoint to resume training')
     ```

5. **布尔参数**：
   - 开关选项（如是否使用数据增强）。
   - 示例：
     ```python
     parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
     ```
     - 使用 `--augment` 设为 `True`，不提供则为 `False`。

---

### 4. 高级用法
1. **互斥参数**：
   - 确保某些参数不能同时使用。
   - 示例：选择训练或测试模式：
     ```python
     group = parser.add_mutually_exclusive_group(required=True)
     group.add_argument('--train', action='store_true', help='Train mode')
     group.add_argument('--test', action='store_true', help='Test mode')
     ```

2. **参数选择范围**：
   - 限制参数值到特定选项。
   - 示例：
     ```python
     parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='adam', help='Optimizer')
     ```

3. **列表参数**：
   - 接受多个值（如卷积核大小）。
   - 示例：
     ```python
     parser.add_argument('--kernel-sizes', type=int, nargs='+', default=[3, 5], help='List of kernel sizes')
     ```
     - 命令：`--kernel-sizes 3 5 7`

4. **文件路径验证**：
   - 确保路径存在。
   - 示例：
     ```python
     def valid_path(path):
         if not os.path.exists(path):
             raise argparse.ArgumentTypeError(f"Path {path} does not exist")
         return path
     parser.add_argument('--data', type=valid_path, required=True)
     ```

5. **子命令**：
   - 支持不同任务（如训练、推理、评估）。
   - 示例：
     ```python
     subparsers = parser.add_subparsers(dest='command')
     train_parser = subparsers.add_parser('train', help='Train model')
     train_parser.add_argument('--data', type=str, required=True)
     test_parser = subparsers.add_parser('test', help='Test model')
     test_parser.add_argument('--model', type=str, required=True)
     ```
     - 命令：
       ```bash
       python script.py train --data ./dataset
       python script.py test --model model.pth
       ```

---

### 5. 最佳实践
1. **清晰的帮助信息**：
   - 为每个参数提供详细的 `help` 描述，方便用户运行 `python script.py -h` 查看。

2. **短选项与长选项**：
   - 常用参数用短选项（如 `-d`），非常用参数只用长选项（如 `--log-dir`）。
   - 示例：
     ```python
     parser.add_argument('-d', '--data', type=str, required=True, help='Dataset path')
     ```

3. **默认值**：
   - 为非必需参数设置合理默认值，减少用户输入。
   - 示例：
     ```python
     parser.add_argument('--batch-size', type=int, default=32)
     ```

4. **类型检查**：
   - 使用 `type` 确保参数格式正确（如 `float` 给学习率）。
   - 示例：
     ```python
     parser.add_argument('--lr', type=float, default=0.001)
     ```

5. **参数分组**：
   - 使用 `add_argument_group` 将参数分组，提高可读性。
   - 示例：
     ```python
     data_group = parser.add_argument_group('Data options')
     data_group.add_argument('--data', type=str, required=True)
     model_group = parser.add_argument_group('Model options')
     model_group.add_argument('--model-type', type=str, default='resnet')
     ```

6. **错误处理**：
   - 验证参数值（如路径存在、学习率正数）。
   - 示例：
     ```python
     if args.lr <= 0:
         raise ValueError("Learning rate must be positive")
     ```

7. **配置文件结合**：
   - 结合 `yaml` 或 `json` 文件加载默认参数，`argparse` 用于覆盖。
   - 示例：
     ```python
     import yaml
     with open('config.yaml') as f:
         config = yaml.safe_load(f)
     parser.set_defaults(**config)
     args = parser.parse_args()
     ```

---

### 6. 常见问题
1. **参数未提供怎么办？**
   - 使用 `required=True` 强制提供，或设置 `default` 值。
2. **如何调试？**
   - 打印 `args`：`print(vars(args))`。
   - 查看帮助：`python script.py -h`。
3. **如何处理大量参数？**
   - 使用配置文件（如 `yaml`）存储默认值，`argparse` 只覆盖部分参数。
4. **如何支持可选参数？**
   - 使用 `action='store_true'` 或 `default` 值。

---

### 7. 总结
`argparse` 在深度学习中是配置脚本的标配工具，常用场景包括指定路径、超参数、模型选项和运行模式。其核心优势是灵活性、可读性和标准化。掌握以下关键点即可高效使用：
- 使用 `add_argument` 定义参数，支持短/长选项、类型、默认值。
- 提供清晰的 `help` 信息，方便用户。
- 结合路径验证、选择范围、布尔参数等高级功能。
- 通过配置文件或分组管理复杂参数。

通过合理设计 `argparse`，深度学习脚本可以轻松适应不同数据集、模型和实验设置，提高开发效率和代码复用性。

# 9、cv

OpenCV（`cv2`）是计算机视觉领域的开源库，广泛用于图像和视频处理，尤其在深度学习中用于数据预处理、可视化等。以下是 `cv2` 在深度学习中的常用知识点总结，简洁清晰，涵盖核心功能和实践技巧。

---

### 1. 基本概念
- **作用**：提供图像/视频读取、处理、转换、特征提取等功能，适合深度学习中的数据准备和结果可视化。
- **依赖**：需要安装 `opencv-python`（`pip install opencv-python`）。
- **核心模块**：图像读取/写入、颜色空间转换、几何变换、特征检测、视频处理等。

---

### 2. 常用功能与用法
#### （1）图像读取与保存
- **读取图像**：
  ```python
  import cv2
  img = cv2.imread('image.jpg', cv2.IMREAD_COLOR)  # 彩色图像（BGR）
  # 其他模式：cv2.IMREAD_GRAYSCALE（灰度），cv2.IMREAD_UNCHANGED（包含透明通道）
  ```
  - 返回：NumPy 数组，形状 `(height, width, channels)`（彩色为 3，灰度为 1）。
  - 注意：OpenCV 使用 BGR 颜色顺序（非 RGB）。

- **保存图像**：
  ```python
  cv2.imwrite('output.jpg', img)
  ```
  - 支持格式：`.jpg`、`.png`、`.bmp` 等。

#### （2）颜色空间转换
- **BGR ↔ RGB**（深度学习常用 RGB）：
  ```python
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
  ```
- **BGR ↔ 灰度**：
  ```python
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  ```
- **其他**：`COLOR_BGR2HSV`、`COLOR_BGR2LAB` 等。

#### （3）图像缩放与裁剪
- **缩放**：
  ```python
  img_resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
  # 常见插值：INTER_NEAREST（最近邻）、INTER_LINEAR（双线性）、INTER_CUBIC（双三次）
  ```
  - 深度学习中常用于统一输入尺寸（如 224x224）。
- **裁剪**：
  ```python
  img_cropped = img[y:y+h, x:x+w]  # NumPy 切片，[y, x, channels]
  ```

#### （4）图像预处理
- **归一化**（深度学习输入通常 [0, 1] 或 [-1, 1]）：
  ```python
  img_normalized = img.astype(float) / 255.0
  ```
- **标准化**（减均值、除标准差）：
  ```python
  img_standardized = (img - mean) / std
  ```
- **数据增强**：
  - 旋转：
    ```python
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle=45, scale=1.0)
    img_rotated = cv2.warpAffine(img, M, (w, h))
    ```
  - 翻转：
    ```python
    img_flipped = cv2.flip(img, 1)  # 1: 水平翻转，0: 垂直翻转，-1: 两者
    ```
  - 随机裁剪、亮度/对比度调整：
    ```python
    img_adjusted = cv2.convertScaleAbs(img, alpha=1.2, beta=10)  # alpha: 对比度，beta: 亮度
    ```

#### （5）绘制与标注
- **绘制矩形**（用于可视化边界框）：
  ```python
  cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
  ```
- **绘制文本**（标注类别或分数）：
  ```python
  cv2.putText(img, text="Cat", org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
              fontScale=1.0, color=(0, 0, 255), thickness=2)
  ```
- **绘制线条/圆形**：
  ```python
  cv2.line(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
  cv2.circle(img, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
  ```

#### （6）特征检测与匹配
- **边缘检测**（Canny）：
  ```python
  edges = cv2.Canny(img_gray, threshold1=100, threshold2=200)
  ```
- **角点检测**（Harris）：
  ```python
  corners = cv2.cornerHarris(img_gray, blockSize=2, ksize=3, k=0.04)
  ```
- **SIFT/ORB 特征**（用于传统视觉任务）：
  ```python
  orb = cv2.ORB_create()
  keypoints, descriptors = orb.detectAndCompute(img_gray, None)
  ```

#### （7）视频处理
- **读取视频**：
  ```python
  cap = cv2.VideoCapture('video.mp4')  # 或 0 表示摄像头
  while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
          break
      # 处理 frame
  cap.release()
  ```
- **保存视频**：
  ```python
  out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps=30, frameSize=(width, height))
  out.write(frame)
  out.release()
  ```

#### （8）显示图像
- **显示窗口**：
  ```python
  cv2.imshow('Image', img)
  cv2.waitKey(0)  # 等待按键（0 表示无限等待）
  cv2.destroyAllWindows()
  ```
  - 注意：Jupyter 环境可能不支持 `imshow`，建议用 `matplotlib` 显示：
    ```python
    import matplotlib.pyplot as plt
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    ```

---

### 3. 深度学习中的常见应用
1. **数据预处理**：
   - 读取图像、调整大小、归一化、颜色转换（如 BGR 到 RGB）。
   - 数据增强（旋转、翻转、随机裁剪）以提高模型泛化能力。
2. **可视化**：
   - 绘制预测边界框、分割掩码、关键点。
   - 显示训练过程中的图像或中间特征图。
3. **视频处理**：
   - 实时目标检测、跟踪或动作识别。
   - 提取视频帧作为训练数据。
4. **传统特征提取**：
   - 使用 SIFT/ORB 提取特征，结合深度学习模型（如特征融合）。

---

### 4. 高级功能
1. **图像滤波**：
   - **高斯模糊**：
     ```python
     img_blur = cv2.GaussianBlur(img, (5, 5), sigmaX=0)
     ```
   - **中值滤波**（去噪）：
     ```python
     img_median = cv2.medianBlur(img, ksize=5)
     ```
2. **阈值处理**：
   - **全局阈值**：
     ```python
     _, img_thresh = cv2.threshold(img_gray, thresh=127, maxval=255, type=cv2.THRESH_BINARY)
     ```
   - **自适应阈值**：
     ```python
     img_adaptive = cv2.adaptiveThreshold(img_gray, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         thresholdType=cv2.THRESH_BINARY, blockSize=11, C=2)
     ```
3. **轮廓检测**：
   ```python
   contours, _ = cv2.findContours(edges, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
   cv2.drawContours(img, contours, contourIdx=-1, color=(0, 255, 0), thickness=2)
   ```
4. **模板匹配**：
   ```python
   template = cv2.imread('template.jpg', cv2.IMREAD_GRAYSCALE)
   result = cv2.matchTemplate(img_gray, template, method=cv2.TM_CCOEFF_NORMED)
   min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
   ```

---

### 5. 最佳实践
1. **颜色空间**：
   
   - 始终确认颜色顺序（OpenCV 用 BGR，深度学习框架如 PyTorch/TensorFlow 用 RGB）。
   - 转换后检查：`img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`。
2. **内存管理**：
   - 大量图像处理时，释放资源（如 `cap.release()`、`cv2.destroyAllWindows()`）。
   - 使用生成器加载图像，减少内存占用。
3. **性能优化**：
   - 批量处理图像时，使用 NumPy 矢量化操作而非循环。
   - 对于实时应用，优先选择快速插值（如 `INTER_NEAREST`）。
4. **错误处理**：
   - 检查图像是否成功加载：
     ```python
     if img is None:
         raise ValueError("Failed to load image")
     ```
5. **与深度学习框架结合**：
   - 转换为张量：
     ```python
     import torch
     img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
     ```
   - 配合 `PIL` 或 `torchvision` 进行增强。

---

### 6. 常见问题
1. **图像加载失败**：
   - 检查路径是否正确（支持相对/绝对路径）。
   - 确保文件格式支持（`.jpg`、`.png` 等）。
2. **颜色错误**：
   - 显示或输入模型时，确认 BGR/RGB 转换。
3. **性能瓶颈**：
   - 使用多线程或 `multiprocessing` 加速图像加载/处理。
4. **窗口无法显示**：
   - Jupyter 中用 `matplotlib`，或确保 `cv2.waitKey()` 正确调用。

---

### 7. 总结
`cv2` 是深度学习中图像处理的强大工具，常用功能包括：

- **图像读写**：`imread`、`imwrite`。
- **预处理**：缩放、归一化、颜色转换、数据增强。
- **可视化**：绘制矩形、文本、边界框。
- **视频处理**：帧读取、保存。
- **特征提取**：边缘、角点、SIFT/ORB。
通过熟练掌握这些功能，`cv2` 可无缝集成到深度学习工作流中，提升数据处理效率和结果呈现效果。

# 8.kreas

## 1.超参数优化

**Keras Tuner 简介**

Keras Tuner 是一个专为 TensorFlow/Keras 设计的超参数优化库，简化了模型超参数和架构的调优，支持随机搜索、贝叶斯优化和 Hyperband 算法。适用于优化学习率、层数、神经元数量等。

---

### **核心功能**
- **超参数定义**：
  - `hp.Int(name, min, max, step)`：整数型（如层数、神经元数）。
  - `hp.Float(name, min, max, sampling='linear'/'log')`：浮点型（如学习率）。
  - `hp.Choice(name, values)`：离散值（如激活函数 ['relu', 'tanh']）。
  - `hp.Boolean(name)`：布尔值（如是否使用 Dropout）。
- **条件超参数**：通过 `hp.ConditionalScope` 定义参数依赖。
- **优化算法**：
  - **RandomSearch**：随机采样超参数。
  - **BayesianOptimization**：基于概率模型高效搜索。
  - **Hyperband**：动态分配资源，优先评估高潜力配置。
- **目标指标**：优化验证集指标（如 `val_accuracy`、`val_loss`）或自定义指标。
- **早停支持**：结合 `EarlyStopping` 回调减少无效试验。

---

### **使用步骤**
1. **定义模型**：创建函数 `build_model(hp)`，用 `hp` 指定超参数。
2. **初始化 Tuner**：
   ```python
   tuner = kt.Hyperband(build_model, objective='val_accuracy', max_epochs=10, directory='my_dir', project_name='tune')
   ```
3. **搜索**：`tuner.search(X, y, validation_data=(X_val, y_val), epochs=50)`。
4. **获取结果**：
   - 最佳超参数：`tuner.get_best_hyperparameters()[0].values`。
   - 最佳模型：`tuner.get_best_models()[0]`。

---

### **关键参数**
- **Tuner 初始化**：
  - `objective`：优化目标（如 `'val_accuracy'`）。
  - `max_epochs`：最大训练轮次（Hyperband 专用）。
  - `max_trials`：最大试验次数（RandomSearch/Bayesian）。
  - `executions_per_trial`：每组超参数重复试验次数，减少随机性。
  - `directory/project_name`：保存搜索结果。
- **搜索方法**：
  - `search(X, y, epochs, validation_data, callbacks)`：支持早停、自定义回调。

---

### **安装**
```bash
pip install keras-tuner
```

---

### **简短示例**
```python
import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras

def build_model(hp):
    model = keras.Sequential([
        keras.layers.Dense(hp.Int('units', 32, 128, step=32), activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=keras.optimizers.Adam(hp.Float('lr', 1e-4, 1e-2, sampling='log')),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

tuner = kt.Hyperband(build_model, objective='val_accuracy', max_epochs=10)
tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=50)
best_model = tuner.get_best_models()[0]
```

---

### **优点**
- 简单易用，集成 TensorFlow/Keras。
- 灵活支持多种优化算法。
- 可扩展至复杂模型（如 CNN、RNN）。

### **资源**
- 官方文档：https://keras.io/keras_tuner/
- GitHub：https://github.com/keras-team/keras-tuner
- TensorFlow 教程：https://www.tensorflow.org/tutorials/keras/keras_tuner

---

## 2、Deep Learning  Python

### 1、张量 tensor

#### 张量是矩阵向任意维度的推广。

1. 仅包含一个数字的张量叫做标量，或者0D张量。张量轴的个数也叫做阶数 `rank`。

2. 数字组成的数组叫做向量，或者1D张量。
3. 向量组成的数组叫做矩阵，或者2D张量

#### 张量的关键属性

轴的个数（阶），形状，数据类型

#### 张量运算

1、张量变形：改变张量的额行和列，以得到想要的形状。变形后的张量的元素总个数和之前一样。

2、张量点积：A@B

### 2、广播

**广播（Broadcasting）**是指在进行数组或张量运算时，**自动扩展维度**以便形状不一致的数据能进行运算的机制。

------

#### 简化理解：

当两个形状不同的数组进行操作时，系统会**自动复制较小数组的维度**，使它们形状一致，再执行逐元素操作。

------

#### NumPy 例子：

```python
import numpy as np

a = np.array([1, 2, 3])
b = 2
print(a + b)  # 输出: [3 4 5]
```

`b` 被广播为 `[2, 2, 2]`，然后与 `a` 相加。

------

广播让你**无需手动调整形状**，简化了代码逻辑。

### 3、处理标签和损失

编码标签的方法有两种，一种是分类编码，e.g `to_categorical	` ,相对应的损失函数就应该是 `categorical_crossentropy` ；另一种是整数编码 ，但需要利用numpy将标签转换为张量，相对应的损失函数就是 `sparse_categorical_crossentropy`

### 4、经验之谈

1、中间隐藏层的神经元个数如果比较小，很容易引起信息瓶颈，前一层网络将信息传入该层时，由于神经元数量受限，导致传入信息不完整，导致训练出的模型精度下降。

2、如果神经网络最后一层是纯线性的，没有激活函数，网络可以学会预测任意范围中的值。

3、如果输入数据的特征具有不同的取值范围，应该先进行预处理，即标准化。

4、如果可用的数据很少，可以采用k折交叉验证法。

5、评估模型时，要保证数据具有代表性，在划分训练集和测试集之前，通常应该随机打乱数据，并且要保证训练集和验证集之间没有交集。

6、数据预处理中，神经网络的输入和目标都必须是浮点数张量(有些情况下可以为整数张量)，需要对数据进行标准化，也需要注意缺失值。

### 5、evaluate评估模型

------

#### **1. 在机器学习中：`evaluate()` 的作用**

这是**评估模型在新数据上的表现**，通常用于验证模型是否泛化良好。

例子（Keras）：

```python
loss, accuracy = model.evaluate(x_test, y_test)
print("测试损失：", loss)
print("测试准确率：", accuracy)
```

- `x_test`: 输入测试数据
- `y_test`: 对应的真实标签
- `loss`: 模型在测试集上的损失值（越小越好）
- `accuracy`: 模型在测试集上的预测准确率（越高越好）

------

#### **2. 为什么用 `evaluate()`？**

- 训练后检查模型效果
- 防止过拟合（仅在训练集上好不算好）
- 用于模型对比（哪个模型在测试集上表现最好）

#### **3. 对比训练与评估**

| 操作         | 目的             | 使用的数据       |
| ------------ | ---------------- | ---------------- |
| `fit()`      | 训练模型         | 训练数据         |
| `evaluate()` | 评估模型性能     | 验证/测试数据    |
| `predict()`  | 输出模型预测结果 | 新的未知输入数据 |

### 6、过拟合与欠拟合

机器学习的根本问题是优化与泛化的对立。优化是指调节模型以在训练数据上得到最佳性能，而泛化则是指训练好的模型在前所未见的数据上的性能的好坏，实际上我们往往需要解决过拟合的问题。下面是一些方法

#### 1、减少网络容量

#### 2、添加权重正则化

**也就是强制让模型权重只能取较小的值。从而限制模型的复杂度，这使得权重的分布更加规则**

------

🔧 常见的两种权重正则化方法：

| 名称          | 数学形式   | 常见名称 | 作用说明                     |
| ------------- | ---------- | -------- | ---------------------------- |
| **L1 正则化** | `λ * Σ     | w        | `                            |
| **L2 正则化** | `λ * Σ w²` | Ridge    | 抑制过大的权重，更稳定、常用 |

其中 `λ` 是正则化强度，`w` 是模型的权重。

------

✅ Keras 示例（使用 L2 正则化）：

```python
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense

Dense(64, activation='relu',
      kernel_regularizer=regularizers.l2(0.01))  # L2 正则项
```



------

📌 小结

| 正则化类型 | 关键目的               | 是否常用 |
| ---------- | ---------------------- | -------- |
| L1         | 稀疏权重（特征选择）   | 中等     |
| L2         | 抑制大权重（平滑模型） | 非常常用 |

---

#### 3、 dropout正则化

对某一层使用dropout，就是在训练过程中随机将该层的一些输出特征舍弃（置为0），dropout比率就是被设为0的特征所占的比例

#### 4、获取更多的训练数据

### 7、数据处理

#### 1、卷积网络与全连接网络 对比

Dense层从输入特征空间中学到的是全局模式，而卷积层学习到的是局部模式。这就使得神经网络学到的模式具有平移不变性，卷积完网络学习了某个模式，它可以在全局任何位置识别这个模式。而全连接网络只能重新学习这个模式。另一方面，卷积神经网络可以学习模式的空间层次结构，这时深度轴的通道不再像RGB那样代表特定的颜色，而是代表滤波器。

#### 2、最大池化操作 MaxPooling

利用池化操作来进行下采样的原因，一是减少需要处理的特征图的元素个数，二是通过让连续卷积层的观察窗口越来越大（即窗口覆盖原始输入的比例会越来越大），从而引入空间过滤器的层级结构

####  3、数据预处理

图像处理辅助工具模块 `ImageDataGenerator`

```python
from keras.preprocessing.image import ImageDataGenerator

datagen=ImageDataGenerator(rescale=1./255)
gengerator=datagen.flow_from_directory(
					path,
					target_size=(150,150),
					batch_size=32,
					class_mode="binary")
...
# 利用生成器来训练模型
history = model.fit(
			gengerator,
			steps_per_epoch=100,
			epochs=100,
			validation_data=val_gengerator,
			validation_steps=50)
```

✨ImageDataGenerator进行数据增强  常用的数据增强参数：

| 参数名               | 作用                     |
| -------------------- | ------------------------ |
| `rotation_range`     | 随机旋转角度（0~指定角） |
| `width_shift_range`  | 水平平移（比例或像素）   |
| `height_shift_range` | 垂直平移                 |
| `shear_range`        | 剪切变换角度             |
| `zoom_range`         | 随机缩放                 |
| `horizontal_flip`    | 随机水平翻转             |
| `fill_mode`          | 填充方式（如 'nearest'） |

------

✅ 示例代码：

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 定义数据增强
train_datagen = ImageDataGenerator(
    rescale=1./255,                  # 归一化
    rotation_range=20,              # 随机旋转
    width_shift_range=0.2,          # 水平平移
    height_shift_range=0.2,         # 垂直平移
    zoom_range=0.2,                 # 随机缩放
    horizontal_flip=True,           # 水平翻转
    fill_mode='nearest'             # 边界填充
)

# 加载图片数据
train_generator = train_datagen.flow_from_directory(
    'data/train',                   # 图像所在目录
    target_size=(150, 150),         # 调整大小
    batch_size=32,
    class_mode='categorical'        # 多分类
)
```

------

🎯 用法总结

- 训练数据增强只用在训练集，不用于验证/测试集
- 它是**边训练边增强**（实时生成），不会生成新文件
- 搭配 `model.fit(train_generator)` 使用即可

### 8、预训练的卷积神经网络

预训练网络是一个保存好的网络，之前已经在大型数据集上训练好。使用预训练网络的方法有特征提取和微调模型。

#### 1、特征提取

特征提取是使用之前网络学到的表示来从新样本中提取出的有趣的特征，然后将这些特征输入到一个新的分类器中。

📌 

- 预训练模型提取的特征很通用（比如边缘、结构、语义）
- 节省计算资源，不用从头训练深度网络
- 即使数据少，也能有不错的效果

卷积基学到的表示可能更加通用，因此更适合重复使用。卷积神经网路的特征图表示通用概念在图像中是否存在，是通用的。但是分类器学到的表示是针对于模型训练的类别，其中仅包含某个类别的概率信息，丢失了空间的概念。

值得注意的是，某个卷积层提取的表示的通用性以及可复用性，取决于该层在1模型中深度。模型中更靠近底部的层提取的特征是局部的，比如边缘、颜色、纹理等。而更靠近顶部的层提取的是更加抽象的概念，比如“猫耳朵”，因此如果新的数据集和原先训练的数据集差异比较大的话，最好使用前几层来进行特征提取，而不是选用整个卷积基。

==在这里，更靠近底部的层是指先被添加进模型中的层。==

数据增强的特征提取是指扩展卷积基，你需要将convbase的trainable属性设置为False，保证其在训练过程中参数权重不发生改变，并且这个操作应该在compile之前生效。 关键点：**数据增强 = 训练时实时变化**，但预训练卷积基如果不训练，根本“感知不到这些变化”。

------

🔍 分情况解释：

------

❌ 情况 1：**冻结卷积基，只用于提特征**

- 做法：你把卷积层当作固定特征提取器

  ```python
  conv_base.trainable = False
  ```

- 只对每张图像“提一次特征”

- 数据增强在这里没意义，因为增强图像**不会传递到训练过程**（特征已经提完了）

#### 🧨 举个比喻：

> 你给模型准备了各种不同版本的图像（旋转、缩放）
>  但你只提一次特征 → 模型根本没“看到”这些变化

------

✅ 情况 2：**扩展模型、微调卷积基（fine-tune）**

- 卷积基继续训练 → 每次看到的增强图像都能更新权重
- 增强图像 → 训练更稳健、抗干扰能力更强
- 数据增强能**真正帮助模型学习**

**只有当卷积基参与训练（即扩展/微调时）**，增强数据的“变化”才会传递到模型中，帮助它学得更鲁棒。 如果只是“提特征”，增强图像再多也没用，因为模型只看了一次静态图像。

#### 2、模型微调

将卷积层顶部的几个层解冻，让他们一起参与训练，但应该把优化器的学习率设置的小一点。 

### 9、深度学习用于处理文本和实践序列

#### 1、基本概念

1. 标记：将文本分解而成的单元（单词、字符、序列）叫做标记（token）
2. 分词：将文本分解成标记的过程叫做分词
3. 将向量与标记关联主要有两种方法，one-hot编码和词嵌入，one-hot编码得到的向量是稀疏的（绝大部分元素为0），二进制的，而词嵌入是地位的浮点数向量（即密集的向量），可以将更多信息塞入更低维度中。

#### 2、常见内置方法

```python
preprocessing.sequence.pad_sequences(data,maxlen=100)
# 将整数列表转换成形状为 (samples,maxlen)

Embedding(10000,8,input_length=maxlen)
# 指定Embedding层最大输入长度，一边后面将嵌入输入展平。Embedding层的激活形状为
# (samples,maxlen,Embedding_dim)

```

#### 3、词嵌入

```python
model.layers[0].set_weights(embdedding_matrix)
model.layers[0].trainable=False
```

与卷积网络类似，使用预训练的词嵌入矩阵时，我们需要设置Embedding层的权重为我们构建好的矩阵，并且需要将其冻结，使模型在训练过程中权重不会发生改变。

#### 4、简单循环神经网络（SimpleRNN）

遍历所有的序列元素并保存一个状态，与前馈网络（FFNN）不同，RNN会对序列内部元素进行遍历。在keras中，`RNN`实际上对应着`SimpleRNN`。SimpleRNN层能够像keras中的其他层一样处理序列批量，因此他的接受形状为（batch_size,timesteps,input_features）。SimpleRNN可以在两种不同的模式下运行：一种是返回每个时间不连续输出的完整序列，即形状为（batch_size , timeseps , output_features）的三位张量，另一种是只返回每个输入序列的输出，即形状为（batch_size , output_features）的二维张量。这两种模式由return_sequences这个构造函数来控制。一般而言中间RNN层保持这个参数为`True`，即返回完整的时间序列，并让最后一层为False（default），只输出最终的结果。

但是SImple RNN存在梯度消失问题，随着训练层数的增加，网络最终会无法训练

#### 5、长短期记忆网络（LSTM）

增加了一条平行于序列的“传送带”，序列中的信息可以在任意位置跳上“传送带”，然后被传到更晚的时间步，当需要时又会原封不动地跳回来。LSTM适合评论全局的长期性结构

#### 6、门控循环单元（GRU）

`GRU`利用了数据点地时间顺序，工作原理和`LSTM`类似，但做了一些简化，计算代价更低。

#### 7、利用`循环dropout`降低过拟合

keras的每个循环层都有两个与dropout相关的参数：一个是dropout，他是一个浮点数，指定该层输入单元的dropout比率；另一个是recurrent_dropout，指定循环单元的dropout比率。

#### 8、循环层堆叠

增加网络容量的通常做法是增加每层单元数或增加层数。循环曾堆叠可以构建更加强大的循环网络。在keras中诸葛堆叠循环曾，所有中间层都因该返回完整的输出序列（3D张量），而不是只返回最后一个时间步的输出。这个操作可以通过`return_sequences`来实现

#### 9、双向RNN

双向RNN利用了RNN的顺序敏感性：它包括两个RNN，分别从正序和逆序去处理序列，然后将他们合并在一起。值得注意的是，在一个文本数据集上，逆序处理的效果和正序处理的效果一样好，也就是说：单词顺序对理解语言很重要，但是用哪种顺序并不重要。逆序为机器提供了一种观察任务的全新视角，会捕捉到正序RNN所忽略的一些细节。

```python
model.add(layers.Bidirectional(LSTM(32)))
```

#### 10、一维卷积网络处理序列

卷积运算能从局部输入图块中提取特征，Conv1D接收的输入形状为（samples,time,features）的三维张量。一维卷积网络可以分别处理每个输入序列段，所以他对时间顺序不敏感。同时卷积神经网络在输入时间序列的所有位置寻找模式，呀并不知道所看到的某个模式的时间位置（距开始多长时间，距结束多长时间）。

使用它的技巧是：处理时间序列时，先对他进行卷积，提取特征，之后使用循环神经网络处理序列数据。通常将Conv1D和MaxingPooling1D堆叠咋一起，最后加上一个全局池化操作GlobalMaxPooling1D。如果整体顺序没有那么重要，单独使用一维卷积网络也是可以的。

### 10、深度学习高级操作

#### 1、函数式api

Sequential模型是单输入单输出的线性模型，有时满足不了需要，我们可以使用多输入多输出的Model模型。需要注意的是，严重不平衡的损失贡献会导致模型会针对单个损失最大的任务优化先进性优化，所以我们可以加入损失权重来解决这个问题。

#### 2、共享层权重

层定义一次，但使用多次，每次调用都会相同的权重。

```python
lstm=LSTM(32)
left_output=lstm(left_input)
right_output=lstm(right_input)      
```

#### 3、类标准化

类标准化可以让机器学习模型看到的不同的样本之间批次更加相似，有助于模型的学习与对新数据的泛化。把数据缩放成高斯分布，是数据输入的常用处理手段。批标准化让数据适应性地将数据标准化，有助于梯度上升，在keras中是`BatchNormialization`，该层会接受一个axis参数，表示它对哪一个轴进行表转化，默认是-1，即输入张量的最后一个轴。

#### 4、深度可分离卷积

深度可分离卷积具有更轻便，速度更快的优点，即SeparableConv2D。该层会对输入的每个通道分别执行空间卷积，然后通过逐点卷积（1  x  1点积）将输出通道混合，相当于把空间特征学习和通道特征学习分开。如果输入中的空间位置高度相关，但不同通道之间相对独立，这种方法是很有用的

==深度可分离卷积 = 每个通道先自己卷积（空间提取） + 通道之间再融合（通道提取）==

### 11、深度学习概况

深度学习是机器学习的分支之一。他的模型是一长串的几何函数一个接一个地作用在数据上。这些运算被组织成模块，叫做层。深度学习模型通常是层的堆叠。这些层由权重初始化，权重是在训练过程中需要学习的参数，模型的只是保存在他的权重中，学习的该过程就是为这些权重找到恰当的值。

在深度学习中，一切都是几何空间中的点。首先将模型输入和目标向量化，即将其转换为初始输入向量空间和目标向量空间。深度学习模型的每一层都对通过他的数据做一个简单变换。模型的层链可以被分解为一系列的简单的几何变换。这个复杂变换试图将输入空间映射到目标空间，每次映射一个点。这个变换由层的权重参数化。权重根据当前的表现进行迭代更新。这个几何变换有一个关键性质，就是它必须是可微的，这样我们才能通过梯度下降来学习其参数。直观上看，==这意味着从输入到输出的几何变形必须是平滑且连续的==。

深度的神奇之处在于，它将意义转换为向量，转换为几何空间，然后逐步学习将一个空间映射到另一个空间的复杂几何变换，你需要的只是维度足够大的空间去捕捉原始数据的关系。

### 12、机器学习通用的工作流程

1. 找到可用的数据，你想要预测什么，是否需要人工标注标签。
2. 找到能正确评估目标成功的方法。对于简单任务，可以使用预测精度，但很多情况下都需要与领域相关的复杂指标
3. 准备用于评估模型的验证过程。我们需要定义训练集、验证集、测试集，验证集和测试集的信息不应该提前泄露。
4. 数据向量化，将数据转换为向量并预处理，如标准化。
5. 通过调节参数和添加正则化来逐步改善模型结构，你需要先让模型过拟合，然后再添加正则化或者减小网络尺寸。
6. 调节超参数时要小心验证集过拟合，即超参数可能会过于针对验证集而优化。





# 9、pytorch

## 1、张量运算

[参考该链接](#1、张量)

------

## 2、杂谈

### 1、自动求导

```python
import torch
x=torch.arange(10,dtype=torch.float,requires_grad =True)
y=x*x
u=y.detach()
z=u*x
z.sum().backward()
x.grad
```

自动求导时，只能对标量进行该操作，因此需要进行`sum`。 PyTorch 默认在第一次 `.backward()` 后就**释放了计算图**以节省内存。需要注意的是，只有在反向传播之后，才会使用下面的代码，避免梯度累加。如果张量 `a` 没有参与到当前的计算图中，也就没有被自动微分系统记录下来，它的 `.grad` 不会被更新，因此 **不需要清除梯度**。

```
x.grad.zero_()
```

#### 🪼反向传播

反向传播本质上是链式法则（链式求导），从一个**最终输出**反推所有中间变量的梯度。如果输出是**多个值**（比如一个向量），那么：

- 反向传播不止一个方向，不知道你想对哪个输出做梯度；
- 或者你需要提供梯度权重（即 `.backward(gradient=...)`）告诉它怎么组合多个方向。

```
z.sum().backward(retain_graph=True)
```

#### 🪼**叶子张量（Leaf Tensor）**

1. 是 `requires_grad=True`；

2. 不是通过其他 Tensor 运算得来的（也就是说，没有父节点）；

   对于中间张量来说，pytorch并不会保存他的梯度，除非加上下面代码

   ```
   b.retain_grad()  # 👈 必须加这句才能访问 b.grad
   ```


你提出的这个问题，正是神经网络设计中一个非常经典且关键的“对称性陷阱”，理解它对后续学习非常重要。

------

### 2. **对称性**

你有两个“隐藏单元”，它们结构完全一样（同样的输入，权重参数完全相同），这就是所谓的“对称性”。
 换句话说，隐藏单元1和隐藏单元2从初始化开始是一模一样的。

- **前向传播阶段**
   因为参数完全相同，两个隐藏单元输出完全一样的激活值。
- **反向传播阶段**
   梯度也是一样的，更新后的权重依然完全相同。
   这就像你有两个隐藏单元，但它们一直“绑在一起”，变成了**一个单元**的行为。

- 这就导致网络实际表现就像只有一个隐藏单元，失去了多单元带来的表达能力（也就是学不到更复杂的函数）。

------

#### 1.**梯度不会打破对称**

因为更新是基于相同的梯度和相同的参数执行的，导致两个隐藏单元权重更新完全同步。

==解决方法==

- **随机初始化参数**：这是最常用也是最关键的策略。
   让每个隐藏单元从不同的参数开始，保证它们的激活和梯度不同。
- **暂退法（Dropout）等正则化方法**：在训练过程中随机屏蔽部分单元，可以打破这种对称，帮助学习更丰富的表示。

------

#### 2.总结（务实视角）

> **对称性陷阱**：当多个神经元从相同参数和数据开始时，它们的行为和更新是完全一致的，导致模型能力被严重限制。
>  **解决方法**：避免所有参数完全相同，通常采用随机初始化；训练中还可以借助 Dropout 等技术打破同步。

“协变量偏移”（Covariate Shift）是机器学习和深度学习中一个非常重要但容易被忽视的问题。理解它能帮助你更好地设计训练流程和模型提升泛化能力。

------

### 3、协变量偏移

协变量偏移指的是：

> **训练数据的输入分布和测试数据的输入分布不一致。**

也就是说，模型在训练时见到的数据特征分布 和在测试或实际应用时遇到的特征分布不一样。

------

#### 为什么会出现协变量偏移？

- 现实中数据采集条件不同（传感器、时间、环境变化等）
- 训练集采样不充分或有偏差
- 数据预处理差异

------

#### 协变量偏移的后果？

- 训练好的模型在测试时性能下降，表现出“泛化差”
- 训练时学到的特征分布和实际应用场景不匹配

------

#### 常见的缓解方法

1. **数据归一化和标准化**
    保证训练和测试数据在同一尺度，减少分布差异。
2. **数据增强（Data Augmentation）**
    通过人为制造多样的训练样本，让模型更健壮。
3. **领域自适应（Domain Adaptation）**
    使用技术让模型适应测试数据的分布，比如对抗训练调整特征分布。
4. **重加权样本（Importance Weighting）**
    训练时根据测试数据分布调整样本权重，缩小分布差异。

------

#### 结合神经网络训练的视角

- 在深度学习中，**内部协变量偏移（Internal Covariate Shift）**更为关注——即训练过程中每一层输入分布不断变化，影响训练稳定性。
- 这也是 **批量归一化（Batch Normalization）** 等技术出现的根本原因，帮助缓解训练过程中的协变量偏移。

------

#### 你可以这样理解

> 协变量偏移就像是在学开车，你在白天（训练数据）练习，到了晚上（测试数据）遇到黑暗和雨天，视线变差，表现自然下滑。

------

“内部协变量偏移”（Internal Covariate Shift，简称ICS）是深度学习训练中非常核心但也颇具争议的概念。理解它有助于你更好地掌握训练稳定性和加速收敛的关键技术，比如批量归一化（Batch Normalization）。下面我帮你务实、清晰地解析这个概念。

### 4、“内部协变量偏移”？

- **定义**：在神经网络训练过程中，随着参数不断更新，**每一层输入的分布也在不断变化**，这就是“内部协变量偏移”。
- 换句话说，层与层之间的数据分布“不断漂移”，给训练带来了额外的不稳定性。

- 当某一层的输入分布频繁变化时，下游层需要不断适应新的输入分布，导致训练过程变得**缓慢且不稳定**。
- 梯度更新容易震荡，收敛速度减慢。

------

#### 这个问题的来源

- 每次参数更新，前面层的输出会改变，从而改变后面层的输入分布。
- 网络越深，这种累积的输入分布变化越严重。

------

#### 批量归一化（BatchNorm）如何解决？

- BatchNorm 通过对每一层的输入在**小批量样本内**做归一化（均值0，方差1），**固定输入的分布**，减少了输入的漂移。
- 这样后续层的输入分布变得稳定，网络训练更快、更稳健。

------

#### 总结

| 关键点         | 说明                             |
| -------------- | -------------------------------- |
| 内部协变量偏移 | 网络训练时每层输入分布不断变化   |
| 影响           | 训练不稳定，收敛慢               |
| 解决办法       | 批量归一化，固定输入分布         |
| 争议           | 机制细节仍有学术讨论，但效果显著 |

------

###  5、概念偏移

我们也可能会遇到*概念偏移*（concept shift）： 当标签的定义发生变化时，就会出现这种问题。 这听起来很奇怪——一只猫就是一只猫，不是吗？ 然而，其他类别会随着不同时间的用法而发生变化。 精神疾病的诊断标准、所谓的时髦、以及工作头衔等等，都是概念偏移的日常映射。

### 6、非平稳分布

- 非平稳分布指的是**数据的分布随时间（或环境）发生变化**，即数据生成的统计特性不是固定的。
- 换句话说，数据不满足独立同分布（i.i.d.）的假设。

------

#### 🦄举个简单的例子

- 一个电商平台的用户购买行为随着季节、促销活动不断变化，导致用户的点击率、购买偏好等数据分布不断变。
- 你训练模型时用的是过去半年数据，但测试时用户行为发生了改变，模型效果自然下降。

------

#### 非平稳分布带来的挑战

- **模型泛化能力受限**：模型基于旧数据学到的规律对新数据失效。
- **训练过程不稳定**：模型参数可能在训练中难以收敛。
- **预测性能波动**：线上模型效果随时间波动大，难以保证稳定。

------

#### 非平稳分布的类型（上面提到了一些）

1. **协变量偏移（Covariate Shift）**
    输入分布变化，条件分布不变。
2. **标签偏移（Label Shift）**
    标签分布变化，条件输入分布不变。
3. **概念漂移（Concept Drift）**
    条件分布本身发生变化，即 P(Y∣X)P(Y|X) 变化。

非平稳分布通常表现为上述一种或多种变化的组合。

------

#### 应对非平稳分布的策略

- **在线学习与增量学习**
   模型不断用最新数据微调，适应分布变化。
- **领域自适应与迁移学习**
   设计模型适应新环境，减少分布差异影响。
- **检测与报警机制**
   监测数据分布的漂移，及时触发模型更新。
- **多模型融合**
   结合多个针对不同时间段训练的模型，提高鲁棒性。

------

#### 你可以这样理解

> 非平稳分布就像河流的水流方向和速度时刻变化，行船者（模型）需要不断调整航向，才能顺利到达彼岸。

------

### 7、可学习参数

```
print(net[2].state_dict())
```

这样，可以打印该层的可学习参数权重

```
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
```

这样，可以得到指定层的偏置，类型和数据，要记住：参数是复合的对象，包含值、梯度和额外信息， 除了值之外，我们还可以访问每个参数的梯度。

```
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
```

可以直接得到全部的参数值

```auto
import torch
from torch import nn

"""延后初始化"""
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
# print(net[0].weight)  # 尚未初始化
print(net)

X = torch.rand(2, 20)
net(X)
print(net)
```

延后初始化，也就是定义时，不给出输入维度，让系统根据真实的输入shape去判断。定义层大小时即便唯独不匹配，也不会报错，只有在forward（前向传播）并且需要处理数据时，如果维度不匹配会直接报错（维度匹配是根本）



### 8、保存模型

为什么 **`lambda x: x \* 2`** 这类匿名函数不适合直接放进 `nn.Sequential` 并用 **`torch.save(model)`** 保存整个模型？

| 关键点                                     | 解释                                                         |
| ------------------------------------------ | ------------------------------------------------------------ |
| **PyTorch 保存完整模型依赖 Python pickle** | `torch.save(model)` 实际是把整个模型对象 **pickle** 到磁盘。pickle 记录的并不是“源代码”，而是“在运行时如何找到同名对象”。 |
| **pickle 需要可重定位的“全路径”**          | 当 pickle 遇到一个函数（或方法）时，它存储它的 **module 名 + 对象名**。加载时会做 `import module` 然后用 `getattr(module, name)` 找到同一个对象。 |
| **lambda 没有全路径**                      | 匿名函数的 `__name__` 是 `<lambda>`，而且它通常定义在 `__main__`（交互环境、脚本作用域）里，没有一个可重导入的模块路径；加载时 `import __main__` 后做 `getattr(__main__, '<lambda>')` 必然失败。 |
| **可能还捕获闭包**                         | lambda 往往捕获外部局部变量；pickle 只能保存值，无法重建闭包逻辑，进一步增加不可靠性。 |

演示：为什么 lambda 会出错

```python
import torch
from torch import nn

model = nn.Sequential(
    nn.Linear(4, 4),
    # 放入匿名函数层（会失败）
    lambda x: x * 2                 
)

torch.save(model, "bad.pth")  # 这一步能跑通
model2 = torch.load("bad.pth") # 这里很可能报 pickle 找不到 <lambda>
```

加载时报类似错误：

```
AttributeError: Can't get attribute '<lambda>' on <module '__main__' ...>
```

------

#### 🏞️正确做法：用显式模块来替代 lambda

```python
class TimesTwo(nn.Module):
    def forward(self, x):
        return x * 2

model = nn.Sequential(
    nn.Linear(4, 4),
    TimesTwo()          # 可序列化，因为 TimesTwo 有真正的模块路径
)

torch.save(model, "good.pth")
model2 = torch.load("good.pth")  # 加载成功
```

- `TimesTwo` 定义在某个可导入的 Python 模块（哪怕是当前脚本）。
- pickle 记录 `"your_module_name.TimesTwo"`，加载时能 `import your_module_name` 并拿到同名类，从而恢复对象。

------

#### ⚽面向未来的务实建议

1. **始终把“层”写成 `nn.Module` 子类**
    既避免 lambda 的序列化难题，又能利用 `register_buffer`、`register_parameter` 等高级特性。

2. **生产环境优先保存 `state_dict`**

   ```python
   torch.save(model.state_dict(), "weights.pth")
   ```

   然后 **显式重建结构** 并 `load_state_dict`，跨版本最稳妥。

3. **如果一定要保存完整模型**（原型、快速 demo）：

   - 保证所有自定义层都在可导入的 Python 文件里；
   - 避免 lambda、局部嵌套函数、动态生成类等无法定位的对象。

只要遵循这些约束，你就能既方便地序列化，也能在日后回溯或部署时少踩坑。祝你项目顺利！

### 12、softmax

Softmax 丢掉了特征的绝对信息，仅保留比例信息；若两个特征方向一致或比例相近，Softmax 输出会相似，哪怕它们本质不同。

原输入在经过softmax之后，丢失了原本的特征信息，因为softmax的输出是各个输入在总输入中占的权重比例，那么即使原输入不一样，也很有可能，在总输入中的比例相近。那么，softmax就会得出:这两个输入特征一致，但是事实上，他们并不相同

### 13、==通用架构==

#### 🎋VGG-Net

<img src="C:\Users\osquer\Desktop\typora图片\image-20250627153151729.png" alt="image-20250627153151729" style="zoom: 80%;" />

- VGG-11使用可复用的卷积块构造网络。不同的VGG模型可通过每个块中卷积层数量和输出通道数量的差异来定义。
- 块的使用导致网络定义的非常简洁。使用块可以有效地设计复杂的网络。
- 在VGG论文中，Simonyan和Ziserman尝试了各种架构。特别是他们发现深层且窄的卷积（即3×3）比较浅层且宽的卷积更有效。

#### 🎋Alex-Net

<img src="C:\Users\osquer\Desktop\typora图片\image-20250627153213340.png" alt="image-20250627153213340" style="zoom:80%;" />

1. AlexNet比相对较小的LeNet5要深得多。AlexNet由八层组成：五个卷积层、两个全连接隐藏层和一个全连接输出层。
2. AlexNet使用ReLU而不是sigmoid作为其激活函数。

#### 🎋NIN-Net

<img src="C:\Users\osquer\Desktop\typora图片\image-20250627153125555.png" alt="image-20250627153125555" style="zoom: 80%;" />

- NiN使用由一个卷积层和多个1×1卷积层组成的块。该块可以在卷积神经网络中使用，以允许更多的每像素非线性。
- NiN去除了容易造成过拟合的全连接层，将它们替换为全局平均汇聚层（即在所有位置上进行求和）。该汇聚层通道数量为所需的输出数量（例如，Fashion-MNIST的输出为10）。
- 移除全连接层可减少过拟合，同时显著减少NiN的参数。
- NiN的设计影响了许多后续卷积神经网络的设计。

#### 🎋Geogle-Net

![image-20250627153045561](C:\Users\osquer\Desktop\typora图片\image-20250627153045561.png)

- Inception块相当于一个有4条路径的子网络。它通过不同窗口形状的卷积层和最大汇聚层来并行抽取信息，并使用1×1卷积层减少每像素级别上的通道维数从而降低模型复杂度。
- GoogLeNet将多个设计精细的Inception块与其他层（卷积层、全连接层）串联起来。其中Inception块的通道数分配之比是在ImageNet数据集上通过大量的实验得来的。
- GoogLeNet和它的后继者们一度是ImageNet上最有效的模型之一：它以较低的计算复杂度提供了类似的测试精度。

#### 🎋Dense-Net

- 在跨层连接上，不同于ResNet中将输入与输出相加，稠密连接网络（DenseNet）在通道维上连结输入与输出。
- DenseNet的主要构建模块是稠密块和过渡层。
- 在构建DenseNet时，我们需要通过添加过渡层来控制网络的维数，从而再次减少通道的数量。

#### 🎋LSTM

![image-20250718113432603](C:\Users\osquer\Desktop\typora图片\image-20250718113432603.png)

- 长短期记忆网络有三种类型的门：输入门、遗忘门和输出门。
- 长短期记忆网络的隐藏层输出包括“隐状态”和“记忆元”。只有隐状态会传递到输出层，而记忆元完全属于内部信息。
- 长短期记忆网络可以缓解梯度消失和梯度爆炸。

#### 🎩网络设计心得

1. 网络的输入层，如果size比较大，可以采用较大得kernel size，具有较大的感受野，通常 **7×7 + padding=3 + stride=2**，

2. 1 x 1的卷积核进行逐点卷积，可以保持图像尺寸不变，因为1X1卷积不会看到周围的像素，只会对像素点做线性变换，类似于mlp的效果

3. 使用AGP（AdaptiveAvgPool2d）收尾。这个层对每个通道**独立地计算全局平均**，即对该通道的所有空间位置求平均值，得到一个数值。所以输出是一个 **“每个通道的整体响应强度”**，压缩了空间信息但保留了通道维度的所有信息。换句话说，空间上的细节被“平均掉”了，但每个通道的整体表现依然保留，没有被丢弃。

4. **全局最大池化（Global Max Pooling）**

   - **全局最大池化**是对每个通道取最大值，表达“该通道中是否有特别强烈的激活”。
   - **特点**：
     - 对突出的局部特征敏感，能突出最显著的激活。
     - 适合用于检测“存在性”问题，比如目标检测中判断某类是否出现。
   - **对比平均池化**：
     - 平均池化更稳健，适合捕捉整体的统计信息，降低噪声影响。
     - 最大池化更“激进”，会放大最强响应，忽略其他部分

   | 任务类型             | 推荐池化方式             | 说明                               |
   | -------------------- | ------------------------ | ---------------------------------- |
   | 分类（多类、细粒度） | 全局平均池化（GAP）      | 抓取整体特征，效果稳定             |
   | 检测、定位           | 全局最大池化             | 强调显著区域，有助于定位激活区域   |
   | 结合使用             | 平均池化 + 最大池化 拼接 | 更丰富特征表达，兼顾整体和局部激活 |





### 16、inception块和残差块的对比

#### **Inception 块 vs 残差块：简明对比**

#### **主要区别**
| **特性**     | **Inception 块**                                  | **残差块**                                   |
| ------------ | ------------------------------------------------- | -------------------------------------------- |
| **结构**     | 并行多分支（1x1、3x3、5x5 卷积 + 池化），通道拼接 | 串行卷积（通常 2 个 3x3 卷积）+ 跨层通路相加 |
| **计算方式** | 多尺度特征提取，拼接输出，通道数增加              | 残差学习（`y = F(x) + x`），通道数可变       |
| **设计目标** | 宽度优先：捕获多尺度特征，优化计算效率            | 深度优先：解决深层网络退化，支持更深网络     |
| **输出形状** | 通道数增加，空间尺寸通常不变                      | 通道数可变，空间尺寸可能减小（步幅 > 1）     |

#### **Inception 块特征**
- **并行分支**：1x1、3x3、5x5 卷积和池化，捕获多尺度特征。
- **1x1 卷积降维**：减少计算量，优化效率。
- **通道拼接**：输出通道数为各分支之和，丰富特征表达。
- **应用**：复杂视觉任务（如目标检测、细粒度分类），如 GoogLeNet。

#### **残差块特征**
- **残差连接**：主路径输出 `F(x)` 与输入 `x` 相加，学习残差。
- **跨层通路**：Identity 或 1x1 卷积，确保形状匹配。
- **支持深层网络**：缓解梯度消失和退化，适合极深网络。
- **应用**：图像分类、检测等深层任务，如 ResNet-18/50。

#### **总结**
- **Inception**：强调宽度，多尺度特征提取，计算复杂但高效。
- **残差块**：强调深度，残差学习支持深层网络，结构简单稳定。
- **结合**：Inception-ResNet 融合两者，兼顾宽度和深度。

### 17、序列模型

- 内插法（在现有观测值之间进行估计）和外推法（对超出已知观测范围进行预测）在实践的难度上差别很大。因此，对于所拥有的序列数据，在训练时始终要尊重其时间顺序，即最好不要基于未来的数据进行训练。
- 序列模型的估计需要专门的统计工具，两种较流行的选择是自回归模型和隐变量自回归模型。
- 对于时间是向前推进的因果模型，正向估计通常比反向估计更容易。
- 对于直到时间步的观测序列，其在时间步的预测输出是“步预测”。随着我们对预测时间值的增加，会造成误差的快速累积和预测质量的极速下降。

### 18、模型语言和数据集

最流行的词看起来很无聊， 这些词通常被称为*停用词*（stop words），因此可以被过滤掉。 尽管如此，它们本身仍然是有意义的，我们仍然会在模型中使用它们。 此外，还有个明显的问题是词频衰减的速度相当地快。 例如，最常用单词的词频对比，第个还不到第个的。 

语言模型是自然语言处理的关键。

- 元语法通过截断相关性，为处理长序列提供了一种实用的模型。
- 长序列存在一个问题：它们很少出现或者从不出现。
- 齐普夫定律支配着单词的分布，这个分布不仅适用于一元语法，还适用于其他元语法。
- 通过拉普拉斯平滑法可以有效地处理结构丰富而频率不足的低频词词组。
- 读取长序列的主要方式是随机采样和顺序分区。在迭代过程中，后者可以保证来自两个相邻的小批量中的子序列在原始序列上也是相邻的。

![image-20250705150508506](C:\Users\osquer\Desktop\typora图片\image-20250705150508506.png)

### 19、困惑度

一个更好的语言模型应该能让我们更准确地预测下一个词元。 因此，它应该允许我们在压缩序列时花费更少的比特。 所以我们可以通过一个序列中所有的个词元的交叉熵损失的平均值来衡量：
$$
\frac{1}{n} \sum_{t=1}^n -\log P(x_t \mid x_{t-1}, \ldots, x_1),
$$
其中由P语言模型给出， xt是在时间步从该序列中观察到的实际词元。 这使得不同长度的文档的性能具有了可比性。 由于历史原因，自然语言处理的科学家更喜欢使用一个叫做*困惑度*（perplexity）的量。
$$
\exp\left(-\frac{1}{n} \sum_{t=1}^n \log P(x_t \mid x_{t-1}, \ldots, x_1)\right).
$$

------

#### 🎯 困惑度是什么？一句话定义：

> **困惑度 = 模型对下一个词“不确定”的程度。数值越小越好。**

------

#### 🔍 更准确地说：

困惑度（Perplexity）是语言模型对一个句子的“平均不确定性”的度量，具体公式如下：
$$
Perplexity=exp⁡(−1T∑t=1Tlog⁡P(wt∣w1,...,wt−1))= \exp\left( -\frac{1}{T} \sum_{t=1}^T \log P(w_t \mid w_1, ..., w_{t-1}) \right)
$$
其中：

- TT 是序列长度；
- P(wt∣w1,...,wt−1)P(w_t \mid w_1,...,w_{t-1}) 是模型预测当前词 wtw_t 的概率。

> 换句话说，它就是交叉熵损失的**指数形式**。

------

#### 🔢 用例子理解困惑度

假设词表中有 10000 个词元。

#### ✅ 最理想的模型：

- 每次都把**正确的下一个词**预测概率为 1，其他为 0；
- 那么 log(1) = 0，困惑度 = exp(0) = **1**。

> ✅ 完全没有“困惑”——它完全确定答案。

------

#### ❌ 最差的模型：

- 每次都把正确词元的概率预测为 0；
- log(0) 是负无穷，对数损失无穷大 → 困惑度 = **正无穷**。

> ❌ 永远预测错，彻底“迷惑”。

------

#### ⚠️ 基线模型（完全随机）：

- 假设词表大小为 V = 10000；
- 每个词的概率是 1/10000；
- 那么困惑度 = 10000。

> 📌 这个时候模型“平均每次”都在从 10000 个词中等概率地瞎选 —— 它的困惑度就等于词表大小。

------

#### 📦 所以你说的这句话是？

> 困惑度的最佳理解是：“下一个词元的**实际选择数**的调和平均数”。

这句话可以这样理解：

- 假设你在预测句子：“I want to eat ___.”

  - 模型分配了一个概率分布：比如：

    ```
    pizza: 0.6
    rice:  0.3
    book:  0.1
    ```

  - 那么实际上它只“有效使用”了这几个词元（也就是只关注了3个词，其他忽略不计）；

  - 困惑度会反映这个分布的集中程度：如果概率越集中，困惑度就越小；

  - 如果模型把概率分得很平均（比如每个词都 0.01），就说明它“什么都不确定”，困惑度高。

> 所以：**困惑度越低，代表模型越确定“我知道下一个词是哪个”**。

------

#### 📊 总结一下：

| 情况     | 描述             | 困惑度                     |
| -------- | ---------------- | -------------------------- |
| 理想情况 | 模型预测100%正确 | 1                          |
| 完全随机 | 词表均匀分布     | 词表大小 V                 |
| 完全错误 | 总是错           | 正无穷                     |
| 实际模型 | 介于两者之间     | 一般在 20~500+，取决于任务 |

------

#### 💡 实用建议

- 训练语言模型时，如果你在监控困惑度，看到它下降就说明模型在“学会预测下一个词”。
- 若困惑度长期停留在词表大小附近，说明模型还没学到有用的信息。
- 可以把困惑度当作一个“可解释的准确率指标”。

### 20、隐状态

**隐藏层**是网络的空间维度——处理“这一次输入”要经过哪些算子。 **隐状态**是网络的时间维度——上一次处理结果如何影响“下一次输入”。RNN 的参数矩阵（例如用于计算隐藏状态的权重）在每次训练结束后才会更新。而“承载先前状态”的是隐藏状态本身（h_t），不是这个矩阵。

- **隐状态（hidden state）** 是模型在每个时间步计算出的中间变量，它承载了对序列历史信息的记忆，**用于前向传播时传递信息**。
- 它本身**不直接作为参数**，**不存储梯度**，所以不会被直接更新，也不会参与梯度下降。
- 但是，隐状态是通过权重矩阵（比如你代码里的 `w_hh` 和 `w_xh`）和输入计算出来的，权重矩阵是**模型的可训练参数**，在反向传播时会计算梯度并更新。
- 换句话说，隐状态是“动态生成”的中间结果，权重矩阵才是“静态”的、训练时需要优化的东西。
- 你可以理解为隐状态通过权重矩阵间接反映了模型的“记忆”，权重矩阵的调整影响隐状态的变化和模型的表现。

这也是RNN结构的核心：状态的传递保证了时间维度上的信息积累，而权重矩阵的优化保证了模型性能的提升。

### 21、 `prefix` 预热

定义预测函数来生成`prefix`之后的新字符， 其中的`prefix`是一个用户提供的包含多个字符的字符串。 在循环遍历`prefix`中的开始字符时， 我们不断地将隐状态传递到下一个时间步，但是不生成任何输出。 这被称为*预热*（warm-up）期， 因为在此期间模型会自我更新（例如，更新隐状态）， 但不会进行预测。 预热期结束后，隐状态的值通常比刚开始的初始值更适合预测， 从而预测字符并输出它们。

RNN 是有**记忆能力**的，隐状态 `H` 是对历史输入的“总结”。如果你一开始就随机初始化状态（`H=0`），预测是盲目的。用 prefix 走一遍，把 `H` 更新成“上下文相关”的，预测才有语义意义。

### 22、隐状态和模型参数

别把 **“隐状态 H”** 和 **“模型参数 W”** 混在一起——它们是两条完全不同的链：

| 概念                        | 作用                       | 生命周期                        | 是否 `detach()`                  | 是否跨 epoch 传递                                            |
| --------------------------- | -------------------------- | ------------------------------- | -------------------------------- | ------------------------------------------------------------ |
| **模型参数**`W_xh, W_hh, …` | 网络的可训练权重           | **整个训练过程** 都存在         | **不**；反向传播要用它们的梯度   | **会**：优化器更新后直接写回，同一个 `net` 在下一 epoch 继续用更新后的参数 |
| **隐状态**`H_t`             | RNN 当下记忆，用来算下一步 | **在一个批次的前向过程中** 持续 | **要**（顺序分区时）让梯度别跨批 | **通常不会**：epoch 之间重新置零                             |

------

#### 训练一轮（epoch）时发生什么？

1. **前向计算**

   - 用当前权重 WW 与隐状态 HH 计算输出。

2. **反向传播**

   - 计算损失对各层权重的梯度 ∇W\nabla W（此时 `H.detach()` 只影响梯度**链条长度**，不影响 ∇W\nabla W 的数值正确性）。

3. **优化器步进**

   ```python
   optimizer.step()    # W ← W - lr * ∇W
   optimizer.zero_grad()
   ```

   - **权重 W 被真正修改并保存在 `net` 内部**。

4. **下一个批次 / epoch**

   - 用更新后的 WW 继续前向，模型能力逐步提升。
   - **隐状态 H** 在跨批(顺序分区)可延续其数值、跨 epoch 一般重置为 0；它不是参数，不参与 optimizer 更新。

------

#### 一目了然的小循环伪码

```python
for epoch in range(num_epochs):
    state = net.begin_state(batch_size, device)  # 隐状态通常每轮重置
    for X, Y in train_iter:                      # 遍历所有 batch
        # ---- forward ----
        y_hat, state = net(X, state)             # state.detach() 已在 net 内做
        loss = criterion(y_hat, Y)

        # ---- backward & update ----
        loss.backward()                          # 计算 ∇W
        torch.nn.utils.clip_grad_norm_(net.params, theta)  # (可选)梯度裁剪
        optimizer.step()                         # ← 真正写回 W
        optimizer.zero_grad()
```

- `optimizer.step()` **写回修改后的参数 W**，所以 **权重自动随 epoch 累积效果**。
- `state` 只是这一批的记忆，“借壳”来帮助前向计算；`detach()` 只影响它的梯度链，不影响 `W`。

------

#### 核心记忆点

> **`detach()` 只是让 \*梯度\* 不再往更早的时间步传，
>  它不会阻止 \*权重\* 在 optimizer.step() 时被永久更新，
>  下一 epoch 加载的仍然是“上轮更新后”的权重。**

这样就能既控制梯度爆炸，又让参数持续学习。

### 23、把 **“梯度链”** 和 **“权重张量”** 区分开来

| 概念               | 作用                            | 生命周期                      | 与 `detach()` 的关系                        |
| ------------------ | ------------------------------- | ----------------------------- | ------------------------------------------- |
| **梯度计算图**     | 记录前向算子，用来反向传播误差  | 仅存在于**一次前向‑反向周期** | `detach()` **剪断**它，使反向传播停在剪断点 |
| **权重张量** (`W`) | 网络可训练参数，存储在 `net` 里 | 整个训练过程中持续存在        | `detach()` **不会**影响它的存储或更新       |

------

#### 1 为什么要 `detach()`？

```python
state = state.detach()
```

- 目的是**断开梯度反向传播链条**，让接下来的 `loss.backward()` 只在 **当前小批量的时间步** 内展开。
- **只是对梯度的“图结构”动刀**；`state`（数值）还在，下一批采用这同一份 `H` 值继续前向。

------

#### 2 权重何时更新？

```python
loss.backward()   # 计算 dL/dW
optimizer.step()  # W ← W - lr * dL/dW
```

- `optimizer.step()` **就地修改** `W` 的 `.data`，新的数值立即写回。
- 这一步**与梯度链条是否被 detach 毫无冲突**——它只看 `W.grad` 里的数值。

------

#### 3 跨 epoch 的连贯性

- **epoch 结束**：我们通常 `state = None` 或重新置 0，但 **`net` 仍然保留更新后的权重**。
- **下一 epoch 开始**：重用 **同一个 `net` 对象**，它的权重已经是“上一轮学习后”的新值，于是继续收敛。

```text
Epoch 1
  forward → backward → optimizer.step()  (W₁ → W₂)
Epoch 2
  forward uses W₂
```

------

#### 4 一句核心对照

| 操作               | 影响                                            |
| ------------------ | ----------------------------------------------- |
| `detach()`         | **剪断梯度传播路径**，限制反向传播深度          |
| `optimizer.step()` | **更新权重数值**，持久写入 `net`，跨 epoch 保留 |

两者操作对象完全不同，因此“`detach()` 不会阻止权重更新”——权重依旧在每个 batch 后被写回；而“权重更新”也不会破坏你对梯度链深度的控制。

------

#### ⌛ 口诀

> **梯度链用 `detach` 割，
>  权重值由 `step` 改，
>  割链不割权，权改值长存。**
>
> 

### 24、向前传播与反向传播

==关键点：**计算图是“边建边连”的**，而不是“只在 backward 那一刻才存在”==

1. **每一次前向运算**（矩阵乘、加法、`tanh` …）
    PyTorch 都会即时在内存里创建一个小节点，串进**当前这一条链**。
2. **如果你连续调用多次前向**，并且这些运算之间 **有张量依赖**，
    那么这几条小链就会**拼成一条更长的链**。
   - 在顺序分区的 RNN 里：
     - **Batch 1** 把隐状态 `H₁_end` 传给 **Batch 2** 作输入；
     - Batch 2 的图就会把 **Batch 1 的整条链也挂进来**，
        于是整段梯度路径 = Batch₁+Batch₂ 的总长度。
3. **直到你调用 `loss.backward()`**，PyTorch 才沿着这“一整条链”回溯并释放图。
   - 如果这条链横跨了几个 batch，就变得又长又占显存，还可能梯度爆炸。

------

#### 所以要 `detach()` 做什么？

```python
state = state.detach()
```

- **把 `state` 的“计算历史”切断**：
  - `state.data` 的数值保留给下一个 batch 当输入；
  - 但它不再指向之前那一大串算子节点。
- 下一个 batch 的前向会**从这根新的“断点”重新开始建图**，
   于是 **反向传播只需要翻过当前 batch 的 `num_steps` 步**，
   既省显存，也避免跨批次爆炸。

> 可以把它想成：**“把已有纸带剪断，新的记录从空白处开始写”**。纸带内容（数值）还在，但旧纸带上的墨迹（梯度可追溯链）被封存，不再连下去。

------

#### 如果不 `detach()` 会怎样？

- Batch 1 图 + Batch 2 图 + …
   一直拖到你调用 `backward()` 的时刻才整体回溯。
- 对 RNN，可能是 **几十上百步**，显存直线上升；
   还可能出现梯度爆炸/消失。

------

#### 核心结论

| 误区                                     | 正解                                                         |
| ---------------------------------------- | ------------------------------------------------------------ |
| “计算图只在 backward 时才有，所以不用管” | ❌ 前向时就**即时构建**，会跨 batch 延长                      |
| “`detach()` 会丢失数值”                  | ❌ 只切梯度链，**数值不变**                                   |
| “剪断后就学不到长依赖”                   | ✅ 只能学到 `num_steps` 步内的依赖；这是**截断 BPTT 的设计权衡** |

通过 `detach()`，我们把**一次反向传播的跨度**控制在可以承受的范围，而权重依然在每个 batch 后被 `optimizer.step()` 持续更新，这就是截断 BPTT 的意义所在。

### 25、BPTT

------

#### 🔁 什么是 BPTT（Backpropagation Through Time）？

BPTT = **时间上的反向传播**

是 RNN（循环神经网络）中训练模型时的核心算法，它是普通的反向传播（Backpropagation，BP）在时间维度上的“扩展”。

------

#### 🧠 直觉类比

普通神经网络是“深层前馈”：

```
x → L1 → L2 → L3 → y
     ↑    ↑    ↑
   grad grad grad
```

RNN 则是“时间展开”，你可以理解为一个网络复制了多份，展开在时间轴上：

```
x₁ → h₁ → y₁  
     ↓  
x₂ → h₂ → y₂  
     ↓  
x₃ → h₃ → y₃  
... 时间轴 ...
```

每个时间步之间的连接是通过**隐藏状态** `hₜ` 实现的。

------

#### ✅ BPTT 步骤

当我们训练 RNN 时，损失是整个序列的：

```python
loss = loss₁ + loss₂ + loss₃ + ...
```

#### BPTT 做的是：

1. 正向传播：逐步计算每个时间步的输出、隐藏状态
2. 反向传播：从最后一个时间步 **向前反传每个时间步的梯度**
   - `∂L/∂W`, `∂L/∂hₜ`, `∂L/∂xₜ`，等等
3. 参数更新：基于累积的梯度更新模型

------

#### 💣 为什么 BPTT 有挑战？

- 计算图太长：比如一个长度 100 的序列，反向传播要展开 100 步
- 容易导致 **梯度爆炸** 或 **梯度消失**
- 显存压力极大

------

#### ✂ 截断 BPTT（Truncated BPTT）

为了解决上述问题，实际训练中我们会限制时间步长度，比如：

```python
num_steps = 35
```

每次只在这 35 步内计算反向传播，**其余的历史用 `detach()` 剪断**，这就叫：

> Truncated BPTT（截断时间上的反向传播）

------

#### ✍️ 总结一句话：

> **BPTT 就是 RNN 训练时在时间维度上的反向传播**，
>  为了避免计算/内存/梯度爆炸问题，常常使用“截断 BPTT”，每 `num_steps` 步就 `detach()` 一次，控制梯度链长度。



### 26、模型训练流程

| 模块       | 关键词                              | 解释                                 |
| ---------- | ----------------------------------- | ------------------------------------ |
| 前向传播   | `model(input)` → `output`           | 将输入数据传入神经网络，得到预测输出 |
| 损失函数   | `loss(output, label)`               | 用于评估预测结果与真实标签之间的误差 |
| 反向传播   | `loss.backward()`                   | 自动计算损失对每个参数的“梯度”       |
| 梯度清零   | `optimizer.zero_grad()`             | 每次更新前都要清除旧梯度，避免累加   |
| 梯度剪裁   | `torch.nn.utils.clip_grad_norm_()`  | 限制梯度过大，防止梯度爆炸           |
| 参数更新   | `optimizer.step()`                  | 按梯度方向更新模型的参数             |
| 隐状态管理 | RNN中如`state.detach()`或重新初始化 | 控制隐藏状态的传递与更新方式         |



### 27、RNN中的常用属性

| 属性名           | 说明                                             |
| ---------------- | ------------------------------------------------ |
| `input_size`     | 输入特征的维度（每个时间步的输入大小）           |
| `hidden_size`    | 隐藏状态的特征维度（隐藏层神经元数量）           |
| `num_layers`     | 堆叠的RNN层数                                    |
| `nonlinearity`   | 非线性激活函数类型，`'tanh'` 或 `'relu'`         |
| `bias`           | 是否使用偏置，布尔值                             |
| `batch_first`    | 输入输出张量的第一个维度是否为batch size，布尔值 |
| `dropout`        | 除最后一层外，其他层的dropout概率                |
| `bidirectional`  | 是否是双向RNN，布尔值                            |
| `weight_ih_l[k]` | 第k层输入到隐藏层的权重（参数张量）              |
| `weight_hh_l[k]` | 第k层隐藏到隐藏层的权重（参数张量）              |
| `bias_ih_l[k]`   | 第k层输入到隐藏层的偏置                          |
| `bias_hh_l[k]`   | 第k层隐藏到隐藏层的偏置                          |

### 28、`tanh` 和 `sigmoid`对比

#### ✅ tanh：用于候选记忆元

$$
\tilde{C}_t
$$



- **值域：**[−1,1]
- **作用：** 能表示正向和负向的激活，适合用来“编码复杂的信息”。
- **原因：** 候选记忆元是**新信息的载体**，需要有能力表达“增强”与“抑制”（正负信号），因此选择了 `tanh`。

👉 **总结一句话：**
 `候选记忆元` 不是门，而是“内容”，需要表达正负信息，选 `tanh` 比较合理。

------

#### ✅ sigmoid：用于门控（输入门、遗忘门、输出门）

- **值域：** [0,1]
- **作用：** 表示“通过程度”（0代表不通过，1代表完全通过）
- **原因：** 门控结构本质上是“滤波器”或者“掩码”，在做“选择性通行”，因此需要的是“比例控制”。

👉 **总结一句话：**
 门控层负责控制信息“通过多少”，使用 sigmoid 实现“0~1之间的比例控制”是最自然的选择。

### 29、`LSTM ` 和  `GRU`

在普通的 RNN 中，隐藏状态的更新是直接的，这样会造成梯度爆炸或消失，难以学习长期依赖。为了解决这个问题，LSTM 和 GRU 引入了**门控机制**，让网络“有选择地”保留旧信息或接纳新信息。

因此，引入了一个“**候选值**”，就是“我准备更新的内容”，但是否真正写入状态（隐状态或记忆元），要看门控的控制。

------

#### 🧠 二、LSTM 中的“候选记忆元参数”详解

LSTM 中有两个状态：

- hth_t：隐状态
- ctc_t：**记忆单元（记忆元）**

更新公式如下：
$$
\begin{align*} f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad\text{(遗忘门)} \\ i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad\text{(输入门)} \\ \tilde{c}_t &= \tanh(W_c \cdot [h_{t-1}, x_t] + b_c) \quad\text{(**候选记忆元**)} \\ c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \quad\text{(更新记忆元)} \\ o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad\text{(输出门)} \\ h_t &= o_t \odot \tanh(c_t) \quad\text{(更新隐状态)} \end{align*}
$$
🔑 **解释**：

- $$
  \tilde{c}_t是候选记忆元：是当前时刻想要写入的内容。
  $$

- 但是否真的写入，要乘上 it（输入门），这是个介于0到1之间的权重。

- 所以，“候选记忆元”提供**新内容的来源**，但真正“记不记”要看门。

#### ✅ 作用：

- **防止信息盲目进入记忆单元**；
- 提高对信息更新的控制能力；
- 保证长期依赖信息不会轻易丢失。

------

#### 🌀 三、GRU 中的“候选隐状态参数”详解

GRU 没有显式的记忆元，只更新隐状态 hth_t，它的结构更紧凑：
$$
\begin{align*} z_t &= \sigma(W_z \cdot [h_{t-1}, x_t] + b_z) \quad\text{(更新门)} \\ r_t &= \sigma(W_r \cdot [h_{t-1}, x_t] + b_r) \quad\text{(重置门)} \\ \tilde{h}_t &= \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h) \quad\text{(**候选隐状态**)} \\ h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \quad\text{(更新隐状态)} \end{align*}
$$
🔑 **解释**：

- $$
  \tilde{h}_t
  $$

  是候选隐状态：当前输入产生的新信息；

- zt控制它与旧隐状态h t-1的比例。

#### ✅ 作用：

- $$
  \tilde{h}_t
  $$

   是模型“希望”作为当前状态的内容；

- 但最终是否使用它由 ztz_t（更新门）决定。

------

#### ✅ 四、总结：为什么要有“候选状态”？

| 类型 | 候选参数         | 作用               | 是否最终使用由谁决定 |
| ---- | ---------------- | ------------------ | -------------------- |
| LSTM | ct（候选记忆元） | 提供新记忆内容     | 输入门 iti_t         |
| GRU  | ht（候选隐状态） | 提供新隐藏状态内容 | 更新门 ztz_t         |

------

### 30、==端到端== 和 ==序列到序列==

当然可以，以下是简洁对比：

------

#### ✅ 端到端学习（End-to-End）

- **定义**：输入直接映射到输出，中间无手工特征。
- **特点**：训练一个整体模型完成全部任务。
- **示例**：语音 → 文本、图像 → 标签。

------

#### ✅ 序列到序列学习（Seq2Seq）

- **定义**：输入输出都是**序列**（长度可变）。
- **结构**：编码器 + 解码器。
- **示例**：机器翻译、文本摘要。

### 31、初识注意力模型

------

✅ 是的，现代的主流网络（尤其是注意力机制下的模型）**不再只依赖编码器“最后一个时刻”的隐状态**来传递信息，而是：

> **传递编码器的每一个时刻的隐状态**（也叫“中间状态序列”或“序列特征”），并让解码器通过“注意力”机制，从中**动态选取重要部分**，生成每个时刻所需的“上下文”向量。

------

#### ❓如果只传最后一个时刻的状态会怎样？

📌 这种做法在最早的 Seq2Seq 模型（如 2014 年 Sutskever 提出的）中是这样做的：

```python
# 编码器输出为一个最后的状态
encoder_final_state = encoder(X)[1]

# 解码器以该状态作为初始状态开始生成序列
decoder_state_0 = encoder_final_state
```

🔍 这种方式的 **缺点** 很明显：

1. **信息压缩瓶颈（信息 bottleneck）**
   - 编码器必须把整个输入序列的信息压缩进一个固定大小的向量中（最后一个隐状态）。
   - 对于长句子或复杂结构的信息，这种压缩是不够的，**会导致丢失上下文细节**。
   - 比如在翻译长句时，解码器根本“记不住”句首的信息。
2. **对长序列性能下降**
   - 随着输入句子变长，最后一个状态包含的信息变得越来越“稀释”，性能显著下降。
3. **不能动态选择关注点**
   - 解码器只能依赖那一个状态，**无法灵活决定“我现在要关注输入序列的哪一部分”**。

------

#### ✅ 为什么注意力机制更好？

✅ 具体做法：

- 编码器输出整个序列的隐状态（shape: `[seq_len, batch_size, hidden_dim]`）
- 解码器每个时间步都通过注意力，从这个序列中“挑选”最相关的信息，构造上下文向量
- 这个上下文再和上一个时间步的解码状态、预测结果组合，输入下一步

✅ 好处：

| 特性       | 传统 Seq2Seq | 注意力机制               |
| ---------- | ------------ | ------------------------ |
| 信息来源   | 最后一个状态 | 所有编码状态             |
| 动态性     | 无           | 每个时间步动态决定注意点 |
| 适应长序列 | 差           | 强                       |
| 翻译质量   | 下降明显     | 提升明显                 |
| 可解释性   | 差           | 高（能看到关注哪些词）   |

------

#### 📌 小结对比图（你可以记住这个关键演变）

```
早期模型：
[ h1 → h2 → h3 → h4 ] → h4 → 解码器

注意力模型：
[ h1, h2, h3, h4 ] → Attention → 每一步动态上下文 → 解码器
```

------

🚀 最后总结一句：

> **“只传最后一个状态”是一种压缩式思维，而“传所有状态 + 注意力”是一种信息解耦 + 动态聚合的更现代策略。**

你也可以把注意力理解为“让解码器拥有搜索输入上下文的能力”，不再死记硬背压缩结果。

------

### 32、🦄比较卷积神经网络、循环神经网络和自注意力这几个架构的计算复杂性、顺序操作和最大路径长度。

#### 1. **计算复杂性**
- **CNN**：
  $$
  O(l \cdot n \cdot k \cdot d \cdot c)
  $$
  
  
  - 卷积操作，线性依赖序列长度 \( n \)，卷积核大小 \( k \)，层数 \( l \)，隐藏维度 \( d \)，通道数 \( c \)。
- **RNN**：
  $$
  O(n \cdot d^2)
  $$
  
  - 逐时间步计算，复杂度随 \( n \) 和隐藏维度 \( d \) 增加，注意力机制加剧 \( n^2 \) 项。
- **自注意力**：
  $$
  O(n^2 \cdot d + n \cdot d^2)
  $$
  
  - 点积 \( QK^T \) 和加权 
    $$
    \text{softmax}(QK^T)V
    $$
    主导，复杂度随 \( n^2 \) 增长。



#### 2. **顺序操作**
- **CNN**：无，卷积并行处理所有位置，适合 GPU 加速。
- **RNN**：有，时间步顺序计算，限制并行性。
- **自注意力**：无，矩阵操作并行计算所有位置，高效。

#### 3. **最大路径长度**
- **CNN**：\( O(l) \)，感受野随层数 \( l \) 线性增加。
- **RNN**：\( O(n) \)，依赖序列长度 \( n \)，长序列易丢失信息。
- **自注意力**：\( O(1) \)，直接建模任意位置依赖，适合长序列。

#### 4. **实现要点**
- **CNN**：
  - 卷积核滑动提取局部特征，池化降维，全连接输出。
  - 例：`nn.Conv1d(num_hiddens, num_hiddens, kernel_size=3)`。
- **RNN**（如 LSTM）：
  - 逐时间步更新隐藏状态（`h_n, c_n`），结合注意力（如 `AdditiveAttention`）。
  - 例：`nn.LSTM(embed_size + num_hiddens, num_hiddens, num_layers)`。
- **自注意力**（如 MultiHeadAttention）：
  - 点积计算注意力权重，多头并行捕捉特征，线性变换输出。
  - 例：`d2l.MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, dropout)`。

#### 5. **上下文总结**
- **CNN**：高效并行，适合局部特征提取，长距离依赖需深层。
- **RNN**：顺序处理，适合短序列，注意力机制提升性能但仍受限于 \( O(n) \) 路径。
- **自注意力**：并行高效，路径长度 \( O(1) \)，适合长序列，但计算成本高（\( O(n^2 \cdot d) \)）。

---

### 33、模型的基本属性

- `children()` = 直接子模块（外层结构）。
- `modules()` = 所有模块（递归）。



## 3、优化算法详解

### 1、综述

优化和深度学习的目标是根本不同的，优化主要关注的是最小化目标，而后者则关注在给定有限有数据量的情况下寻找合适的模型。训练误差与泛化误差通常不同，由于优化算法的目标通常是基于训练数据集的损失函数，因此优化的目标是减少训练误差。但是，深度学习的目标是减少泛化误差，除了使用优化算法来减少训练误差之外，我们还需要注意过拟合。

深度学习中，大多数目标函数都很复杂，没有解析解只有数值解，因此我们必须使用数值优化算法。然而却存在着不小的挑战，比如局部最小值、鞍点、梯度消失。

---

==<  局部最小值  >== 对于任何目标函数，如果在某一点处值都小于其他点处的值，那么该点就是局部最小值点，对应的值就是局部最小值。倘若该最小值是整个域中目标函数的最小值，那么这一值就被称为全局最小值。

当优化问题的数值接近局部最优值时，随着目标函数解的梯度接近或变为零，最终迭代获得的数值解仅使目标函数局部最优，而不是全局最优。只有一定程度的噪声可能会使参数跳出局部最小值，同样，理论上也有可能跳出全局最小值。然而，现实中，深度学习几乎没有**唯一且准确**的全局最小值，最小值往往不是点，而是一片**低损平原**，容错性很好，换言之，只要模型处于这一片区域的某个点，性能就已经足够优秀了。

---

==<  鞍点  >==  函数的所有梯度消失但既不是全局最小值也不是局部最小值的任何位置，如下所示。此时，该点周围的点既有大数值点，又有小数值点。

![../_images/output_optimization-intro_70d214_66_0.svg](https://zh.d2l.ai/_images/output_optimization-intro_70d214_66_0.svg)

对于高维问题，至少部分特征值为负的可能性相当高，这就使得鞍点比局部最小值更有可能出现。高维特征图如下所示。

![../_images/output_optimization-intro_70d214_81_0.svg](https://zh.d2l.ai/_images/output_optimization-intro_70d214_81_0.svg)

---

==<  梯度消失  >==  多个很小的值连乘时出现的梯度很小，近似于无的现象，导致模型无法更新。

---

### 2、梯度下降

==<  本质  >==  

1. **单维梯度下降**

![image-20250725170403684](C:\Users\osquer\Desktop\typora图片\image-20250725170403684.png)

2. **多维梯度下降**

![   ](C:\Users\osquer\Desktop\typora图片\image-20250725170636818.png)

==<  学习率 （learning rate） >== 决定目标函数能否收敛到局部最小值，以及何时收敛到最小值。选择“恰到好处”的学习率是很棘手的。 如果我们把它选得太小，就没有什么进展；如果太大，得到的解就会振荡，甚至可能发散。所以学习率的选取出现了两个方向：预先确定学习率 或 完全不必选择学习率。同时，预处理有助于调节比例。

### 3、随机梯度下降

相比于梯度下降，随机梯度每次下降引入了噪声，每次处理一个样本。也就是说，即使我们接近最小值，我们仍然会受到随机噪声注入的梯度的不确定性的影响，并且这一现象并不会随着时间得到改善，所以我们需要改变学习率。但是，如果我们一开始设置的学习率过小，就不会得到有意义的进展，如果学习率过大，模型将陷入震荡的状态。所以一个好的解决方案是：模型的学习率能够动态减小。下方是设置学习率的基本策略。
$$
\begin{split}\begin{aligned}
    \eta(t) & = \eta_i \text{ if } t_i \leq t \leq t_{i+1}  && \text{分段常数} \\
    \eta(t) & = \eta_0 \cdot e^{-\lambda t} && \text{指数衰减} \\
    \eta(t) & = \eta_0 \cdot (\beta t + 1)^{-\alpha} && \text{多项式衰减}
\end{aligned}\end{split}
$$

### 4、小批量随机梯度下降

相比于前面的梯度更新方法，小批量随机梯度下降可以充分利用GPU或者CPU的多线程能力，同时对一个 batch 里面的数据同时计算，又同时兼具随机梯度下降的优点。
$$
\mathbf{g}_{t, t-1} = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w}_{t-1}) = \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} \mathbf{h}_{i, t-1}.
$$

| 符号      | 含义                                                       |
| --------- | ---------------------------------------------------------- |
| g ~t,t−1~ | 第 tt 次迭代中，对上一轮权重 wt−1的梯度估计                |
| B~t~      | 第 t 个 mini-batch 的样本集合                              |
| f~(xi,w)~ | 损失函数，例如 MSE、交叉熵等                               |
| w~t−1~    | 第 t−1次迭代时的模型参数（如权重）                         |
| $∂/∂w$    | 对参数 w 求导                                              |
| h~i,t−1~  | 第 t−1次迭代中，第 i个样本对应的梯度项：∇~w~f(x~i~,w~t−1~) |



### 5、动量法

==<  泄露平均值  >==  是一种**带有衰减权重的滑动平均**，它不像普通平均那样平均所有历史值，而是对新值赋予更多权重、对旧值逐渐“遗忘”。

我们用泄露平均值取代梯度计算,其中$ \beta ∈(0,1)$​ 。

$ \begin{split}\begin{aligned}
\mathbf{v}_t &\leftarrow \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}, \\
\mathbf{x}_t &\leftarrow \mathbf{x}_{t-1} - \eta_t \mathbf{v}_t.
\end{aligned}\end{split}$

这将有效地把瞬时梯度替换为多个“过去”梯度的平均值，v 被称为动量，它累积了过去的维度，我们可以递归地将v~t~进行扩展。$ \begin{aligned}
\mathbf{v}_t = \beta^2 \mathbf{v}_{t-2} + \beta \mathbf{g}_{t-1, t-2} + \mathbf{g}_{t, t-1}
= \ldots, = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}.
\end{aligned}$

其中，较大的 $\beta$ 相当于长期平均值，而较小的 $ \beta $只是相对于梯度法略有修正。新的梯度替换不再指向特定实例下降最陡的方向，而是指向过去梯度加权平均值的方向。由于对过去的数据进行了指数降权，有效梯度数为 $ \frac{1}{1-\beta}$​

### 6、$AdaGrad$  算法

==<  稀疏特征  >==  某些特征出现的频率比较小，并且只有当这些不常见的特征出现时，与其相关的参数才会得到有意义的更新，最终出现常见特征相当迅速地收敛到最佳值，而对于不常见的特征，我们仍缺乏足够地观测以确定其最佳值。换言之，学习率要么对于常见特征而言降低太慢，要么对于不常见特征而言降低太快。

解决此问题的一个方法是记录我们看到特定特征的次数，然后将其用作调整学习率。 即我们可以使用大小为 $ \eta_i = \frac{\eta_0}{\sqrt{s(i, t) + c}}$ 的学习率。对于本节的`AdaGrad` 算法来说，它使用了 $ s(i, t+1) = s(i, t) + \left(\partial_i f(\mathbf{x})\right)^2$  来调整学习率。 它会随梯度的大小自动变化。通常对应于较大梯度的坐标会显著缩小，而其他梯度较小的坐标则会得到更平滑的处理。我们使用变量s~t~ 去累加过去的梯度方差。具体公式如下：

$\begin{split}\begin{aligned}
    \mathbf{g}_t & = \partial_{\mathbf{w}} l(y_t, f(\mathbf{x}_t, \mathbf{w})), \\
    \mathbf{s}_t & = \mathbf{s}_{t-1} + \mathbf{g}_t^2, \\
    \mathbf{w}_t & = \mathbf{w}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \cdot \mathbf{g}_t.
\end{aligned}\end{split}$

其中 $ \epsilon$ 为一个为维持数值稳定性而添加的常数，确保我们不会除以零。就像在动量法中我们需要跟踪一个辅助变量一样，在AdaGrad算法中，我们允许每个坐标有单独的学习率。需要注意的是在 s~t~ 中累加平方梯度意味着s~t~ 基本上以线性速率额增长。具体代码如下：

```python
def init_adagrad_states(feature_dim):
    s_w = torch.zeros((feature_dim, 1))
    s_b = torch.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] += torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()
```

### 7、$RMSProp$ 算法

 `Adagrad` 算法的明显缺点是，它把梯度的平方类加成状态矢量s ~t~ ，由于缺乏规范化，没有约束力，s~t~ 持续增长，几乎上实在算法收敛时线性增长。解决的方法是：$ \mathbf{s}_t \leftarrow \gamma \mathbf{s}_{t-1} + (1-\gamma) \mathbf{g}_t^2$，同时保持其他部分不变就成为了 `RMSProp` 算法。具体方程如下：
$$
\begin{split}\begin{aligned}
    \mathbf{s}_t & \leftarrow \gamma \mathbf{s}_{t-1} + (1 - \gamma) \mathbf{g}_t^2, \\
    \mathbf{x}_t & \leftarrow \mathbf{x}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t.
\end{aligned}\end{split}
$$
具体代码如下：

```python
def init_rmsprop_states(feature_dim):
    s_w = torch.zeros((feature_dim, 1))
    s_b = torch.zeros(1)
    return (s_w, s_b)

def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] = gamma * s + (1 - gamma) * torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()
```

- RMSProp算法与Adagrad算法非常相似，因为两者都使用梯度的平方来缩放系数。
- RMSProp算法与动量法都使用泄漏平均值。但是，RMSProp算法使用该技术来调整按系数顺序的预处理器。
- 在实验中，学习率需要由实验者调度。
- 系数决定了在调整每坐标比例时历史记录的时长。

### 8、$Adadelta$ 算法

$Adadelta$ 是 $AdaGrad$  的另一种变体，主要区别在于前者减少了学习率适应坐标的数量。此外，广义上的 $AdaGrad$ 算法没有学习率，因为它使用了变量量本身作为未来变化的校准。简而言之，Adadelta使用两个状态变量，s~t~用于存储梯度二阶导数的泄露平均值，$ \Delta\mathbf{x}_t$用于存储模型本身中参数变化二阶导数的泄露平均值。具体方程如下：
$$
\begin{aligned}
    \mathbf{s}_t & = \rho \mathbf{s}_{t-1} + (1 - \rho) \mathbf{g}_t^2.
\end{aligned}
$$

$$
\begin{aligned}
    \Delta \mathbf{x}_t & = \rho \Delta\mathbf{x}_{t-1} + (1 - \rho) {\mathbf{g}_t'}^2,
\end{aligned}
$$

$$
\begin{split}\begin{aligned}
    \mathbf{g}_t' & = \frac{\sqrt{\Delta\mathbf{x}_{t-1} + \epsilon}}{\sqrt{{\mathbf{s}_t + \epsilon}}} \odot \mathbf{g}_t, \\
\end{aligned}\end{split}
$$

$$
\begin{split}\begin{aligned}
    \mathbf{x}_t  & = \mathbf{x}_{t-1} - \mathbf{g}_t'. \\
\end{aligned}\end{split}
$$

我们使用重新缩放的梯度g~t~ ^,^ 执行更新。.其中$\Delta \mathbf{x}_{t-1}$是重新缩放梯度的平方的泄漏平均值。我们将$\Delta \mathbf{x}_{0}$初始化为0，然后在每个步骤中使用 g~t~^,^更新它，

代码实现如下：

```python
%matplotlib inline
import torch
from d2l import torch as d2l


def init_adadelta_states(feature_dim):
    s_w, s_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
    delta_w, delta_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        with torch.no_grad():
            # In-placeupdatesvia[:]
            s[:] = rho * s + (1 - rho) * torch.square(p.grad)
            g = (torch.sqrt(delta + eps) / torch.sqrt(s + eps)) * p.grad
            p[:] -= g
            delta[:] = rho * delta + (1 - rho) * g * g
        p.grad.data.zero_()
```

### 9、$Adam$算法

$Adam$  算法汇总了前面的算法的优点。

- 随机梯度下降——在解决优化问题时比梯度下降更有效。
- 小批量随机梯度下降——可以充分利用GPU或者CPU的并行处理数据能力，更高效。
- 动量法——添加了一种机制，用于汇总过去梯度的历史以加速收敛。
- AdaGrad  算法——我们通过对每个坐标缩放实现高效计算的预处理器。
- RMSProp 算法——我们通过学习率的调整来分离每个坐标的缩放。

---

方程如下：

![image-20250725204620553](C:\Users\osquer\Desktop\typora图片\image-20250725204620553.png)

具体代码如下：

```python
%matplotlib inline
import torch
from d2l import torch as d2l


def init_adam_states(feature_dim):
    v_w, v_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
    s_w, s_b = torch.zeros((feature_dim, 1)), torch.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = beta2 * s + (1 - beta2) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1
```

- Adam算法将许多优化算法的功能结合到了相当强大的更新规则中。
- Adam算法在RMSProp算法基础上创建的，还在小批量的随机梯度上使用EWMA。
- 在估计动量和二次矩时，Adam算法使用偏差校正来调整缓慢的启动速度。

### 10、$Yogi$​算法

Adam算法也存在一些问题： 即使在凸环境下，当的二次矩估计值爆炸时，它可能无法收敛。作者建议在正式训练前，**用一个大批量数据**，先运行一次（或几次）梯度计算，得到更“可靠”的初始梯度统计量，再作为动量/方差估计的初始值。

![image-20250725205417840](C:\Users\osquer\Desktop\typora图片\image-20250725205417840.png)

```python
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = s + (1 - beta2) * torch.sign(
                torch.square(p.grad) - s) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

- 对于具有显著差异的梯度，我们可能会遇到收敛性问题。我们可以通过使用更大的小批量或者切换到改进的估计值来修正它们。Yogi提供了这样的替代方案。

### 11、学习率调度器

#### 1、==综述==

1. 首先，学习率的大小很重要。如果它太大，优化就会发散；如果它太小，训练就会需要过长时间，或者我们最终只能得到次优的结果。
2. 其次，衰减速率同样很重要。如果学习率持续过高，我们可能最终会在最小值附近弹跳，从而无法达到最优解。
3. 另一个同样重要的方面是初始化。这既涉及参数最初的设置方式（，又关系到它们最初的演变方式。这被戏称为*预热*（warmup），即我们最初开始向着解决方案迈进的速度有多快。一开始的大步可能没有好处，特别是因为最初的参数集是随机的。最初的更新方向可能也是毫无意义的。
4. 虽然我们不可能涵盖所有类型的学习率调度器，但我们会尝试在下面简要概述常用的策略：多项式衰减和分段常数表。 此外，余弦学习率调度在实践中的一些问题上运行效果很好。 在某些问题上，最好在使用较高的学习率之前预热优化器。

---

#### 2、==下面是各种调度器，用于得到合适的学习率==

1. **单因子调度器**

   ![image-20250725211612794](C:\Users\osquer\Desktop\typora图片\image-20250725211612794.png)

```python
class FactorScheduler:
    def __init__(self, factor=1, stop_factor_lr=1e-7, base_lr=0.1):
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.base_lr = base_lr

    def __call__(self, num_update):
        self.base_lr = max(self.stop_factor_lr, self.base_lr * self.factor)
        return self.base_lr

scheduler = FactorScheduler(factor=0.9, stop_factor_lr=1e-2, base_lr=2.0)
d2l.plot(torch.arange(50), [scheduler(t) for t in range(50)])
```

![image-20250725211655241](C:\Users\osquer\Desktop\typora图片\image-20250725211655241.png)

2. **多因子调度器**

<img src="C:\Users\osquer\Desktop\typora图片\image-20250725212913226.png" alt="image-20250725212913226" style="zoom:80%;" />

调度器并不知道「你正在第几个 epoch」，它只是每次你调用 scheduler.step()，就内部计数 +1。一切靠你在训练循环中按正确时机调用它。

```python
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
scheduler = lr_scheduler.MultiStepLR(trainer, milestones=[15, 30], gamma=0.5)

def get_lr(trainer, scheduler):
    lr = scheduler.get_last_lr()[0]
    trainer.step()
    scheduler.step()
    return lr

d2l.plot(torch.arange(num_epochs), [get_lr(trainer, scheduler)
                                  for t in range(num_epochs)])
```

![../_images/output_lr-scheduler_1dfeb6_98_0.svg](https://zh.d2l.ai/_images/output_lr-scheduler_1dfeb6_98_0.svg)

3. **余弦调度器**（Cosine Annealing Scheduler）

学习率从初始值逐步 **沿余弦曲线下降**，最终接近 0。

```
初始 lr ↓          ⎯⎯⎯⎯⎯
                ／        ＼
              ／            ＼
            ／                ＼
末尾 lr →                 ⎯⎯⎯⎯⎯

```

![image-20250725213350658](C:\Users\osquer\Desktop\typora图片\image-20250725213350658.png)



```python
class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0,
               warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                self.base_lr_orig - self.final_lr) * (1 + math.cos(
                math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr

scheduler = CosineScheduler(max_update=20, base_lr=0.3, final_lr=0.01)
d2l.plot(torch.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

4. **预热**

预热（Warm-up）指的是在训练初期，逐步从较小的学习率增加到设定的初始学习率。它不是直接使用目标学习率，而是像这样慢慢升上去：

```python
学习率曲线：
         /‾‾‾‾‾‾‾‾‾‾‾‾
        /
       /
______/

```

它的整体目标是：

- **前若干步先预热**：学习率从很小线性增长到 base_lr。
- **中后期逐渐减小学习率**：使用余弦函数平滑地从 base_lr 减到 final_lr。

---

```python
class CosineScheduler:
    def __init__(self, 
                 max_update,         # 总的训练轮数（或步数）
                 base_lr=0.01,       # 初始学习率（预热后达到的）
                 final_lr=0,         # 最终学习率（训练结束后达到的）
                 warmup_steps=0,     # 预热的轮数
                 warmup_begin_lr=0): # 预热开始时的学习率
```

举例：

- 如果 `warmup_steps=5`，`base_lr=0.3`，`final_lr=0.01`，`max_update=20`：
  - 前5个 epoch：线性预热从 0 → 0.3
  - 第6~20个 epoch：从 0.3 通过余弦退火平滑地减小至 0.01

---



1️⃣ 计算预热阶段学习率（线性增长）

```python
def get_warmup_lr(self, epoch):
    increase = (self.base_lr_orig - self.warmup_begin_lr) * float(epoch) / float(self.warmup_steps)
    return self.warmup_begin_lr + increase
```

这段代码的含义是：

- 第 `epoch` 步的学习率 = 初始值 + 线性增长的比例
- 比如 warmup 从 0 到 0.3，共 5 步，那么第 3 步的学习率大约是 `0 + (0.3 - 0) * 3/5 = 0.18`

------



2️⃣ 正式退火阶段（余弦函数控制学习率下降）

```python
if epoch <= self.max_update:
    self.base_lr = self.final_lr + (
        self.base_lr_orig - self.final_lr) * 
        (1 + math.cos(math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
```



------

✅ 优点总结

| 特性               | 说明                                                |
| ------------------ | --------------------------------------------------- |
| 🚀 Warm-up          | 初始训练不稳定，避免过大梯度引起发散                |
| 🌊 Cosine Annealing | 平滑收敛，避免震荡，适合 fine-tuning                |
| 🔧 参数灵活         | 可设置最大/最小学习率、warmup长度等                 |
| 🎯 精度提升         | 实际应用中，常用于 ResNet、Transformer 等大模型训练 |

## 4、计算性能

### 1、编译器和解释器

==<  命令式编程  >==  命令式编程是一种按照**指令一步步改变程序状态**的编程风格，程序员告诉计算机 **怎么做**、按什么步骤执行。

```python
total = 0
for i in range(10):
    total += i
print(total)

```

==<  符号式编程  >==  符号式编程强调用**符号表示抽象问题**，程序结构就是数据结构，程序**可以操作自身的代码结构**，更侧重“表达”和“重写”而不是执行指令。因为符号式编程是将代码**作为表达式结构**来处理，编译器可以“看到”整个表达式的结构，所以就有空间做**很多自动优化**。

```python
(define (square x) (* x x))
(square 5)

```

命令式（解释型）编程和符号式编程的区别如下：

- 命令式编程更容易使用。在Python中，命令式编程的大部分代码都是简单易懂的。命令式编程也更容易调试，这是因为无论是获取和打印所有的中间变量值，或者使用Python的内置调试工具都更加简单；
- 符号式编程运行效率更高，更易于移植。符号式编程更容易在编译期间优化代码，同时还能够将程序移植到与Python无关的格式中，从而允许程序在非Python环境中运行，避免了任何潜在的与Python解释器相关的性能问题。



```python
net = torch.jit.script(net)
net(x)

```

上方的代码可以将命令式模型转换为符号式模式。

==<  解释器  >==  程序运行时，**逐行翻译+立即执行**源码，边读边干

==<  编译器  >==  在程序运行前，**一次性把源码翻译成机器码/字节码**，之后直接运行已编译好的结果。

### 2、异步计算

`pytorch` 的GPU操作默认是异步执行的 ，把数据送入GPU之后，python会立即收回控制权，此时可能数据还没有算完。

广义上说，PyTorch有一个用于与用户直接交互的前端（例如通过Python），还有一个由系统用来执行计算的后端。PyTorch 的 Python 层**只负责构建计算任务并发出指令，真正的计算任务是由底层的 C++/CUDA 后端完成的，并且默认是异步执行的**。当你执行 `z = x + y` 这样的操作时，Python 并没有真正去“计算”加法；它只是构造了一个“加法任务”并传给 C++ 层，后者再决定怎么交给 CUDA 执行；一直到你真正需要这个结果（比如 `print(z)`），**PyTorch 才同步等待后端完成任务**。（==“前后端解耦”==）

### 3、硬件基础

<u>计算机由以下关键部件组成：</u>

- 一个处理器（也被称为CPU），它除了能够运行操作系统和许多其他功能之外，还能够执行给定的程序。它通常由个或更多个核心组成；
- 内存（随机访问存储，RAM）用于存储和检索计算结果，如权重向量和激活参数，以及训练数据；
- 一个或多个以太网连接，速度从1GB/s到100GB/s不等。在高端服务器上可能用到更高级的互连；
- 高速扩展总线（PCIe）用于系统连接一个或多个GPU。服务器最多有个加速卡，通常以更高级的拓扑方式连接，而桌面系统则有个或个加速卡，具体取决于用户的预算和电源负载的大小；
- 持久性存储设备，如磁盘驱动器、固态驱动器，在许多情况下使用高速扩展总线连接。它为系统需要的训练数据和中间检查点需要的存储提供了足够的传输速度。



==<  缓存  >==

- **一级缓存**是应对高内存带宽要求的第一道防线。一级缓存很小（常见的大小可能是32-64KB），内容通常分为数据和指令。当数据在一级缓存中被找到时，其访问速度非常快，如果没有在那里找到，搜索将沿着缓存层次结构向下寻找。
- **二级缓存**是下一站。根据架构设计和处理器大小的不同，它们可能是独占的也可能是共享的。即它们可能只能由给定的核心访问，或者在多个核心之间共享。二级缓存比一级缓存大（通常每个核心256-512KB），而速度也更慢。此外，我们首先需要检查以确定数据不在一级缓存中，才会访问二级缓存中的内容，这会增加少量的额外延迟。
- **三级缓存**在多个核之间共享，并且可以非常大。AMD的EPYC 3服务器的CPU在多个芯片上拥有高达256MB的高速缓存。更常见的数字在4-8MB范围内。



==<  深度学习的互联方式  >==  ![image-20250727184633244](C:\Users\osquer\Desktop\typora图片\image-20250727184633244.png)

## 5、自然语言处理

### 1、跳元模型和连续词袋模型

 **跳元模型（Skip-gram）** 和 **连续词袋模型（CBOW, Continuous Bag of Words）** 两者都是 **Word2Vec** 框架下的两种词向量训练方法，核心目标都是：

> 根据词在上下文的分布规律学习词的稠密向量表示，使得语义相近的词在向量空间中距离更近。

------

#### 2️⃣ 核心思想

| 模型          | 核心任务                          |
| ------------- | --------------------------------- |
| **CBOW**      | 用 **上下文词** 来预测 **中心词** |
| **Skip-gram** | 用 **中心词** 来预测 **上下文词** |

------

#### 3️⃣ 输入输出结构

**CBOW**

- **输入**：上下文中的多个词（窗口大小 `m`）
- **输出**：窗口中间的那个中心词
- **例子**（窗口大小=2）：
   句子 `The quick brown fox jumps`
   预测 `"brown"` 的输入是 `["The", "quick", "fox", "jumps"]`

**Skip-gram**

- **输入**：一个中心词
- **输出**：它的上下文词
- **例子**（窗口大小=2）：
   输入 `"brown"` → 输出 `["The", "quick", "fox", "jumps"]`

------

#### 4️⃣ 模型结构（训练过程）

CBOW 过程

1. 对输入的上下文词做 **词向量查表** → 得到多组词向量
2. 求平均（或加和） → 得到上下文向量
3. 经过全连接层（或 softmax）预测中心词
4. 用交叉熵损失更新词向量参数

Skip-gram 过程

1. 对输入的中心词做 **词向量查表**
2. 将中心词向量送入模型，分别预测每个上下文词
3. 多个输出词的损失求和
4. 反向传播更新参数

------

#### 5️⃣ 对比总结

| 对比维度         | CBOW                   | Skip-gram              |
| ---------------- | ---------------------- | ---------------------- |
| **预测方向**     | 上下文 → 中心词        | 中心词 → 上下文        |
| **适合数据规模** | 大语料，低频词训练不佳 | 小语料也能表现好       |
| **训练速度**     | 快（一次预测一个词）   | 慢（一次预测多个词）   |
| **低频词效果**   | 较差                   | 更好                   |
| **优点**         | 训练快、对高频词效果好 | 能更好捕捉罕见词的语义 |
| **缺点**         | 低频词语义学习差       | 训练时间长             |



### 2、负采样和分层 `softmax`

好，那我帮你系统整理一下 **负采样（Negative Sampling）** 和 **层序 Softmax（Hierarchical Softmax）**，这两个是 Word2Vec 里常用的 **softmax 加速技巧**。

------

#### 1️⃣ 背景问题：为什么要加速？

在 Word2Vec（Skip-gram 或 CBOW）里，输出层通常是一个 **softmax 分类器**：

问题：

- 词表大小 VV 可以是几十万甚至上百万
- 每次计算梯度都要遍历整个词表 → **非常慢**

于是就有了 **近似 softmax** 的两种主流方法：

- 负采样（Negative Sampling）
- 层序 softmax（Hierarchical Softmax）

------

#### 2️⃣ 负采样（Negative Sampling）

💡 核心思想

不去更新整个词表的参数，而是：

1. 更新**正样本**（目标词）的参数
2. 再随机挑选 k 个**负样本**（非目标词）更新参数

这样每次只计算 k+1个词，而不是 V 个。



------

#### 3️⃣ 层序 Softmax（Hierarchical Softmax）

💡 核心思想

用一棵 **霍夫曼树（Huffman Tree）** 代替扁平化的 softmax 分类器。

- 词表中每个词是叶子节点

- 每个内部节点是一个二分类器

- 词的概率 = 从根到该词叶子的路径上所有二分类器概率的乘积

  

------

#### 4️⃣ 对比总结

| 特性             | 负采样                | 层序 Softmax         |
| ---------------- | --------------------- | -------------------- |
| **时间复杂度**   | O(k)                  | O(log⁡V)              |
| **实现难度**     | 简单                  | 较复杂               |
| **低频词效果**   | 好                    | 一般                 |
| **适合场景**     | 词表大、需要稀疏更新  | 词表特别大（百万+）  |
| **是否全局更新** | 否（只更新 k+1 个词） | 否（只更新路径节点） |

---

#### 5️⃣ 小记忆口诀

> - **负采样**：扔掉大部分词，只和少数“假词”比
> - **层序 Softmax**：走一条树路径，省得遍历全词表



## 6、数学基础

### 1、张量

==张量可以存储数据，并在gpu上面进行计算，并支持自动微分。==

#### 📦 1. 张量创建

```python
import torch

# 零 / 一
torch.zeros(2, 3)                 # 2x3 全零张量
torch.ones((4, 5))               # 4x5 全一张量

# 随机
torch.rand(3, 3)                 # 均匀分布[0,1)
torch.randn(3, 3)                # 标准正态分布 N(0,1)

# 常数/指定值
torch.full((2, 2), 7.5)          # 全为7.5的张量

# 从列表创建
torch.tensor([[1, 2], [3, 4]])   # 显式创建

# 类似 numpy 的 linspace/eye
torch.linspace(0, 1, steps=5)    # 均分 0~1 的 5 个数
torch.eye(3)                     # 单位矩阵
```

------

#### 🧮 2. 基础数学运算

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# 加减乘除
a + b
a - b
a * b
a / b

# 指数、平方根、log
torch.exp(a)
torch.sqrt(b)
torch.log(b)

# 点积、矩阵乘法
torch.dot(a, b)                    # 向量点积
A = torch.randn(2, 3)
B = torch.randn(3, 4)
torch.matmul(A, B)                # 矩阵乘法
```

------

#### 🔄 3. 维度变换操作

```python
x = torch.randn(2, 3)

x.view(3, 2)                      # 改变 shape（连续张量）
x.reshape(3, 2)                   # 改变 shape（更通用）

x.T                               # 转置（2D）
x.permute(1, 0)                   # 更通用的维度重排

x.unsqueeze(0)                    # 增加维度 [2,3] → [1,2,3]
x.unsqueeze(-1)                   # 增加最后一维

x.squeeze()                       # 去掉所有维度为1的轴
```

------

#### 🧱 4. 拼接与分割

```python
a = torch.randn(2, 3)
b = torch.randn(2, 3)

torch.cat((a, b), dim=0)          # 沿 dim=0 拼接 → [4,3]
torch.stack([a, b], dim=0)        # 添加新维度 → [2,2,3]

# 拆分
x = torch.randn(4, 6)
torch.chunk(x, 2, dim=0)          # 沿 dim=0 分成2块
torch.split(x, 3, dim=1)          # 每块3列
```

------

#### 🎯 5. 索引 & 条件操作

```python
x = torch.tensor([[1, 2], [3, 4]])

x[0, 1]                            # 获取指定位置
x[:, 1]                            # 获取第2列
x[x > 2]                           # 条件筛选

torch.where(x > 2, x, torch.zeros_like(x))  # 条件选择
```

------

#### 🔁 6. 广播（Broadcasting）

```python
x = torch.tensor([[1.0], [2.0]])    # shape [2,1]
y = torch.tensor([10.0, 20.0])      # shape [2]

# 会自动扩展为 [2,2]
x * y                              # 广播乘法
```

------

#### 🧠 7. 聚合运算

```python
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

x.sum()                            # 所有元素求和
x.mean(dim=0)                      # 每列均值
x.max(dim=1)                       # 每行最大值和索引
x.argmin()                         # 最小值索引
```

------

#### 🧷 8. 数据类型与设备管理

```python
x.float()                          # 转 float32
x.double()                         # 转 float64
x.int()                            # 转 int32

x.to("cuda")                       # 送到 GPU
x.to(torch.device("cpu"))         # 回到 CPU
```

------

#### 📐 9. 维度命名（高级特性）

```python
# 需要启用
torch.set_default_tensor_type('torch.FloatTensor')

x = torch.randn((2, 3), names=('N', 'C'))
x = x.rename(None)                 # 移除名字
```

------

#### 🔒 10. 原地操作（以 _ 结尾）

```python
x = torch.tensor([1.0, 2.0])
x.add_(3.0)                        # 原地加法 x = x + 3
x.zero_()                          # 原地归零
```

------

#### ✅ 总结建议

| 类型     | 推荐操作                          |
| -------- | --------------------------------- |
| 创建张量 | `torch.tensor / rand / zeros`     |
| 运算     | `+ - * /`, `matmul`, `dot`        |
| 维度变换 | `reshape`, `permute`, `unsqueeze` |
| 广播     | 自动完成（理解右对齐即可）        |
| GPU 支持 | `.to("cuda")` 或 `device="cuda"`  |

### 2、广播机制

1. 通过适当复制元素来扩展一个或两个数组，以便在转换之后，两个张量具有相同的形状；
2. 对生成的数组执行按元素操作。

```python
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
```

---

### 3、线性代数

```python
A.sum(axis=1)
A.sum(axis=1,keepdims=True)   # 保持维度
A.cumsum(axis=1)  #沿着该轴计算累计和，并且不改变张量shape
A * B  # 逐元素乘法,即对应位置相乘再相加
A @ B  # 元素点积
torch.mv(A,B)  # 矩阵-向量积
torch.mm(A,B)  # 矩阵乘法，适用于二维以下的矩阵乘法
```

![image-20250811212504070](C:\Users\osquer\Desktop\typora图片\image-20250811212504070.png)

---



线性代数中最有用的一些运算符是*范数*（norm）。 非正式地说，向量的*范数*是表示一个向量有多大。 这里考虑的*大小*（size）概念不涉及维度，而是分量的大小。

![image-20250811213436334](C:\Users\osquer\Desktop\typora图片\image-20250811213436334.png)

 $\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.$ 为 L1 范数 ，$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2}$ 为L2范数，

![image-20250811213612962](C:\Users\osquer\Desktop\typora图片\image-20250811213612962.png) 

**范数和目标**

在深度学习中，我们经常试图解决优化问题： *最大化*分配给观测数据的概率; *最小化*预测和真实观测之间的距离。 用向量表示物品（如单词、产品或新闻文章），以便最小化相似项目之间的距离，最大化不同项目之间的距离。 目标，或许是深度学习算法最重要的组成部分（除了数据），通常被表达为范数



## 8、模型度量方法（损失函数）

==*损失函数*（loss function）能够量化目标的*实际*值与*预测*值之间的差距。==



### 1. 均方误差 (MSE) 损失
- **使用场景**：主要用于回归任务，目标是预测连续值，例如房价预测或温度预测。
- **选择原因**：MSE因其平方操作对较大误差施加更大惩罚，适用于需要特别避免大偏差的情景。此外，其可微性支持PyTorch中的梯度优化。
- **特征**：
  - 测量预测值与实际值平均平方差。
  - 由于二次性质，对异常值敏感。
  - 在PyTorch中通过 `torch.nn.MSELoss()` 实现。
  - 公式：  
    $$
    \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
    $$

  ```
  criterion = nn.MSELoss()
  loss = criterion(y_pred, y)
  ```
  
  

### 2. 均绝对误差 (MAE) 损失
- **使用场景**：适用于稳健性要求较高的回归任务，例如金融预测中对异常值的容忍。
- **选择原因**：MAE对所有误差的惩罚线性相关，较少受极端值影响，适合数据分布不均匀或存在噪声的情况。
- **特征**：
  - 计算预测值与实际值绝对差的平均值。
  - 对异常值鲁棒性强，但梯度信息较少。
  - 在PyTorch中通过 `torch.nn.L1Loss()` 实现。
  - 公式：  
    $$
    \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
    $$

  ```
  criterion = nn.L1Loss()
  loss = criterion(y_pred, y)
  ```
  
  

### 3. 交叉熵损失 (Cross-Entropy Loss)
- **使用场景**：广泛用于分类任务，包括二分类（sigmoid激活）和多分类（softmax激活），如图像分类或自然语言处理。
- **选择原因**：交叉熵损失直接优化分类问题的对数似然，适用于概率输出模型，能有效区分正确类别与错误类别。
- **特征**：
  - 结合softmax或sigmoid与负对数似然，衡量预测概率分布与真实分布的差异。
  - 对置信度低的预测施加较大惩罚。
  - 在PyTorch中通过 `torch.nn.CrossEntropyLoss()` 实现（包含softmax）。
  - 公式（二分类简化）：  
    $$
    \text{CE} = - \frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]
    $$

```
criterion = nn.CrossEntropyLoss()
loss = criterion(y_pred, y)
```



### 4. 二元交叉熵损失 (Binary Cross-Entropy Loss)
- **使用场景**：专用于二分类任务，例如垃圾邮件检测或疾病诊断。
- **选择原因**：针对二值输出优化，适用于sigmoid激活后的概率预测，计算效率高。
- **特征**：
  
  - 针对单个二元分类器，评估正类和负类的对数损失。
  - 需与 `torch.nn.Sigmoid()` 结合使用，或直接用 `torch.nn.BCELoss()`。
  - 公式：  
    $$
    \text{BCE} = - \frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]
    $$

```
criterion = nn.BCELoss()
loss = criterion(y_pred, y)
```



### 5. 枢纽损失 (Huber Loss)
- **使用场景**：在回归任务中，需平衡MSE对异常值的敏感性和MAE的鲁棒性，例如机器人控制或时间序列预测。
- **选择原因**：Huber损失在误差较小时接近MSE，在误差较大时接近MAE，兼顾稳定性与梯度平滑。
- **特征**：
  
  - 引入阈值参数（默认1），控制线性与二次区域的切换。
  - 对异常值较鲁棒，梯度平滑。
  - 在PyTorch中通过 `torch.nn.HuberLoss()` 实现。
  - 公式：  
    $$
    \text{Huber}(x) = 
    \begin{cases} 
    \frac{1}{2} x^2 & \text{if } |x| \leq \delta \\
    \delta |x| - \frac{1}{2} \delta^2 & \text{if } |x| > \delta 
    \end{cases}
    $$
    

### 总结与选择指导
- **回归任务**：MSE适合数据平滑，MAE或Huber适合异常值较多；Huber为折中选择。
- **分类任务**：交叉熵适用于平衡数据集，二元交叉熵专为二分类，Focal Loss针对不平衡数据。
- **特点对比**：MSE和交叉熵对梯度敏感，MAE和Huber更鲁棒，Focal Loss强调困难样本。
- 在PyTorch中，损失函数通过 `torch.nn` 模块调用，需根据任务需求选择并与模型输出格式匹配（例如，`CrossEntropyLoss` 需原始logits）

---

## 9、参数初始化

参数初始化是深度学习模型训练的重要步骤，直接影响收敛速度和最终性能。以下是对 PyTorch 中常用参数初始化方法（以 `net[0].weight.data.normal_(0, 0.01)` 和 `net[0].bias.data.fill_(0)` 为例）的总结，包括相关方法、应用场景及原因，保持专业、清晰的语言结构。

### 1. 常用参数初始化方法
#### a. 正态分布初始化 (`normal_`)
- **方法**: `tensor.normal_(mean, std)` 或 `tensor.data.normal_(mean, std)`
  - 例如: `net[0].weight.data.normal_(0, 0.01)` 将 `weight` 初始化为均值为 0、标准差为 0.01 的正态分布。
- **应用场景**:
  - 适用于大多数全连接层和卷积层的权重初始化，尤其在深层网络中。
  - 常见于激活函数为 ReLU 或其变种的网络（如 ResNet、Transformer），或需要小初始权重以避免梯度问题。
- **原因**:
  - 小标准差（如 0.01）防止初始权重过大，减少梯度爆炸或消失风险。
  - 均值 0 确保对称性，便于梯度传播。
  - Xavier/Glorot 初始化（`nn.init.xavier_normal_`）是其改进版，基于输入输出维度自适应标准差。

#### b. 常数填充 (`fill_`)
- **方法**: `tensor.fill_(value)` 或 `tensor.data.fill_(value)`
  - 例如: `net[0].bias.data.fill_(0)` 将偏置初始化为 0。
- **应用场景**:
  - 偏置项（bias）的标准初始化，特别是在 ReLU 激活前。
  - 某些情况下用于特定层（如批量归一化层的偏置），但需谨慎。
- **原因**:
  - 偏置初始化为 0 不会影响网络对称性，且与 ReLU（输出非负）兼容。
  - 避免初始输出偏移，保持梯度流动平稳。

#### c. 均匀分布初始化 (`uniform_`)
- **方法**: `tensor.uniform_(low, high)` 或 `tensor.data.uniform_(low, high)`
  - 例如: `nn.init.uniform_(tensor, a=0, b=0.01)`
- **应用场景**:
  - 适用于权重初始化，特别是在浅层网络或 sigmoid/tanh 激活函数的场景。
  - Xavier 均匀初始化 (`nn.init.xavier_uniform_`) 基于输入输出维度调整范围。
- **原因**:
  - 均匀分布提供多样性，防止权重过于集中。
  - 范围需根据激活函数选择（如 [-0.1, 0.1] 避免饱和）。

#### d. Xavier/Glorot 初始化
- **方法**: `nn.init.xavier_normal_(tensor)` 或 `nn.init.xavier_uniform_(tensor)`
- **应用场景**:
  - 适用于深层网络，尤其是使用 sigmoid 或 tanh 激活的模型。
  - 常用于 CNN 和 RNN 的权重初始化。
- **原因**:
  - 根据输入维度 \(n_{\text{in}}\) 和输出维度 \(n_{\text{out}}\)，标准差为 \(\sqrt{\frac{2}{n_{\text{in}} + n_{\text{out}}}}\)，平衡前向和反向信号。

#### e. He 初始化
- **方法**: `nn.init.kaiming_normal_(tensor, mode='fan_out', nonlinearity='relu')`
- **应用场景**:
  - 专为 ReLU 及其变种激活函数设计的网络（如现代 CNN，如 VGG、ResNet）。
- **原因**:
  - 标准差为 \(\sqrt{\frac{2}{n_{\text{in}}}}\)，适配 ReLU 的单侧激活特性，防止梯度消失。

#### f. 预训练初始化
- **方法**: 加载预训练模型权重，例如 `model.load_state_dict(torch.load('pretrained.pth'))`
- **应用场景**:
  - 迁移学习场景，如使用 ImageNet 预训练的 ResNet 模型。
- **原因**:
  - 利用预训练权重加速收敛，改善小数据集性能。

### 2. 应用场景总结
- **浅层网络**: 均匀初始化或小正态初始化（std=0.01）即可，简单有效。
- **深层网络**:
  - ReLU 激活: He 初始化（`kaiming_normal_`）最佳。
  - Sigmoid/Tanh 激活: Xavier 初始化（`xavier_normal_` 或 `xavier_uniform_`）更合适。
- **偏置**: 通常初始化为 0，除非特定需求（如批量归一化层的偏置可能初始化为小值）。
- **迁移学习**: 优先使用预训练权重，微调特定层。
- **特殊情况**: 某些模型（如 LSTM）可能需要正交初始化 (`nn.init.orthogonal_`) 以保持梯度流动。

### 3. 注意事项
- **激活函数匹配**: 初始化方法需与激活函数匹配，否则可能导致梯度问题（例如，ReLU 需 He 初始化，sigmoid 需 Xavier）。
- **数据尺度**: 初始化范围应与输入数据尺度一致，避免过大或过小。
- **in-place 操作**: 方法如 `normal_`, `fill_` 直接修改 `data`，需谨慎使用。
- **PyTorch 集成**: 使用 `nn.Module` 的参数时，可通过 `nn.init` 模块或直接操作 `.weight.data` 和 `.bias.data`。

### 结论
参数初始化如 `net[0].weight.data.normal_(0, 0.01)` 和 `net[0].bias.data.fill_(0)` 是标准做法，分别适配权重的小随机初始化和偏置的零初始化。选择方法需根据网络深度、激活函数和任务需求，Xavier 和 He 初始化为深层网络的优选，预训练初始化则适用于迁移学习。合理初始化可显著提升训练效率和模型性能。

## 10、经典网络层

### 1、线性神经网络

![image-20250813104626628](C:\Users\osquer\Desktop\typora图片\image-20250813104626628.png)

由于模型重点在发生计算的地方，所以通常我们在计算层数时不考虑输入层。 我们可以将线性回归模型视为仅由单个人工神经元组成的神经网络，或称为单层神经网络。对于线性回归，每个输入都与每个输出（在本例中只有一个输出）相连， 我们将这种变换（ [图3.1.2](https://zh.d2l.ai/chapter_linear-networks/linear-regression.html#fig-single-neuron)中的输出层） 称为*全连接层*（fully-connected layer）或称为*稠密层*（dense layer）。 

当面对更多的特征而样本不足时，线性模型往往会过拟合。 相反，当给出更多样本而不是特征，通常线性模型不会过拟合。 不幸的是，线性模型泛化的可靠性是有代价的。 简单地说，线性模型没有考虑到特征之间的交互作用。 对于每个特征，线性模型必须指定正的或负的权重，而忽略其他特征。

### 2、Dropout层 

#### 1、基本概况（[返回原连接](#2、dropout 丢弃法)）

- **训练阶段**: Dropout 以概率 \( p \)（例如 0.5）随机将神经网络中某些神经元的输出置为 0，同时将保留神经元的输出乘以$ \frac{1}{1-p}$（例如 2），以保持期望输出不变。每次前向传播，丢弃的神经元集合是随机的。

- **测试阶段**: Dropout 被禁用，使用完整模型，不进行丢弃或缩放。

- **目标**: 通过随机丢弃，模拟多个子网络的训练，增强模型泛化能力，防止过拟合。

-  应用场景：

  - 适用于深层网络，尤其是卷积神经网络（CNN）和循环神经网络（RNN），如 ResNet、LSTM。
  - 特别适合数据量不足或模型过参数化的情况。

- **代码示例**:

  ```python
  import torch
  import torch.nn as nn
  
  model = nn.Sequential(
      nn.Linear(10, 5),
      nn.Dropout(p=0.5),  # 50% dropout 率
      nn.ReLU(),
      nn.Linear(5, 2)
  )
  ```

#### 2. 您的疑问解析
- **问题核心**: 您提到“当丢弃一些神经元后，此时的隐藏层与权重相乘，但下一次丢弃时，隐藏层与之前不同，那学习的权重不就效果没那么好了吗？”
  - **观察**: 每次训练迭代中，Dropout 随机丢弃不同神经元，导致隐藏层输出的激活值（即与权重相乘的输入）在不同批次或迭代中变化。这可能让人疑惑：如果权重是基于当前隐藏层学习的，下次隐藏层变化后，权重是否仍适用？
  - **潜在担忧**: 您可能担心权重学习变得不稳定，因为它们是为特定丢弃模式优化的，而每次丢弃模式不同。

#### 3. 为什么权重学习不会“效果变差”
- **Dropout 的正则化效应**:
  - Dropout 迫使网络在训练时学习更鲁棒的权重。每次丢弃不同神经元，模型必须依靠剩余神经元的组合来完成任务。这鼓励权重分布更均匀，不依赖特定神经元，类似于集成学习（ensemble learning）的效果。
  - 例如，假设隐藏层有 10 个神经元，\( p = 0.5 \)，每次可能丢弃 5 个。网络学会了多种子网络的权重配置，最终权重适应了多种丢弃模式。

- **权重更新的适应性**:
  - 权重通过梯度下降优化，基于整个训练过程的平均梯度更新。Dropout 引入的随机性被平均化，权重逐渐调整为适合各种丢弃配置。
  - 数学上，Dropout 的期望输出保持不变（\(\mathbb{E}[h_{\text{dropout}}] = h\)，其中 \( h \) 是原始激活），这确保权重学习的目标一致。

- **隐藏层变化的意义**:
  - 隐藏层输出变化是 Dropout 设计的意图。每次不同的丢弃模式相当于训练一个新的子网络，权重通过多次迭代学习这些子网络的共同模式。
  - 这种变化不会使权重“失效”，而是使权重更具泛化性，因为它们必须适应随机缺失的输入。

#### 4. 为什么不会影响学习效果
- **鲁棒性提升**:
  - 如果权重过于依赖特定神经元，Dropout 的随机丢弃会惩罚这种依赖，迫使网络学习更分散的表示。这提高了模型对噪声或数据变化的鲁棒性。
  - 例如，在垃圾邮件分类中，如果权重过于依赖“尼日利亚”一词，丢弃后网络必须依靠其他特征（如“西联汇款”），最终学习两者的组合效应。

- **测试时的完整性**:
  - 测试时关闭 Dropout，使用所有神经元，权重已通过训练适应了多种子网络的平均行为。缩放因子 \(\frac{1}{1-p}\) 确保训练和测试的期望输出一致，避免偏差。

- **梯度平均效应**:
  - 反向传播的梯度是基于当前批次的损失计算的。Dropout 随机性被多次迭代平滑，权重更新反映了整体趋势，而非单一丢弃模式。

#### 5. 澄清困惑
- **“学习权重效果没那么好”**:
  - 短期看，特定迭代的权重可能不完美匹配当前丢弃模式，但长期看，权重通过多次随机丢弃学习到稳健解。
  - 这是 Dropout 的设计目标：牺牲短期一致性，换取长期泛化能力。
- **“隐藏层与之前不同”**:
  - 不同是正常的，Dropout 利用这种变化迫使网络分散学习，而不是依赖固定结构。

#### 6. 直观类比
- 想象一个足球队：训练时随机让部分球员休息（Dropout），迫使其他球员学会协作；比赛时（测试）所有球员上场，团队已适应多种配置，表现更稳定。权重就像球员的技能，随机休息让技能更全面。

#### 7. 实践验证
- **实验建议**: 尝试一个简单网络，比较有/无 Dropout 的训练过程。观察权重变化和测试准确率，验证随机性如何提升泛化。

#### 结论
Dropout 随机丢弃神经元导致隐藏层变化，但不会使权重学习效果变差。相反，这种变化通过正则化迫使权重适应多种子网络，增强泛化性。测试时关闭 Dropout 利用完整模型，权重已优化为稳健解。理解这一机制有助于优化 Dropout 率 \( p \) 和网络设计。



### 3、卷积层

#### 1、**卷积层的特点：**

1. *平移不变性*（translation invariance）：不管检测对象出现在图像中的哪个位置，神经网络的前面几层应该对相同的图像区域具有相似的反应，即为“平移不变性”。
2. *局部性*（locality）：神经网络的前面几层应该只探索输入图像中的局部区域，而不过度在意图像中相隔较远区域的关系，这就是“局部性”原则。最终，可以聚合这些局部特征，以在整个图像级别进行预测。

卷积层会将输入和核矩阵进行交叉相关，奖赏偏移后得到输出，其中核矩阵和偏移和可学习参数，核矩阵的大小为超参数。

#### 2、特征映射

- **定义**:
  - 卷积层的输出通常被称为特征映射，因为它将输入数据（例如图像）通过卷积操作转换为下一层的空间维度表示。每个特征映射对应于一个卷积核（filter），提取输入中的特定特征（如边缘、纹理）。
- **作为转换器**:
  - 特征映射可以将输入的空间信息（宽度、高度）和通道信息映射到新的表示，保留局部模式，同时减少全局冗余。

#### 3、感受野
- **定义**:
  - 在 CNN 的某一层中，任意输出元素 \( y[i, j] \) 的感受野是指在前向传播期间，==所有可能影响其计算的输入元素集合==。这些输入元素通常来自所有先前层，覆盖一个局部的空间区域。
  - 感受野的大小由卷积核尺寸、层数、步幅和池化操作共同决定。
- **直观理解**:
  - 感受野描述了网络中某一神经元“看到”的输入区域。例如，在第一层卷积，感受野等于滤波器大小；在深层，感受野扩展到输入的更大区域。
- **示例**:
  - 对于一个$  3 \times 3  $滤波器，第一层的输出元素受输入中 $  3 \times 3  $ 区域影响。
  - 如果第二层再次应用$  3 \times 3  $ 滤波器，感受野扩展到$  5 \times 5  $（因为每个输出元素依赖前一层 $  3 \times 3  $的输出，而前一层输出又依赖输入$  3 \times 3  $。

#### 4、卷积权重

卷积核与输入的乘积大小代表了这块区域的匹配程度，但需要注意的是卷积中的“匹配”不是靠大小，而是靠方向，乘积结果只有在方向匹配的情况下，数值才有意义

#### 5、卷积输入和输出的联系

$$
W=(W+Padding*2-kernel Size+strides)//strides
$$

当 $ Padding*2-kernel Size+1=0$​ 时，输入和输出的图像大小一致。

#### 6、多输入和多输出通道



![image-20250815175816473](C:\Users\osquer\Desktop\typora图片\image-20250815175816473.png)

* 输出通道数是卷积层的超参数
* 每个输入通道有独立的二维卷积核，所有通道结果相加得到一个输出通道结果
* 每个输出通道有独立的三维卷积核： $Cin \times kernel\ size \times kernel \ size$​
* 此外， $  1 \times 1  $​ 卷积可以在不改变空间特征的情况下，  变换输出通道数。

### 4、汇聚层（池化层）

在处理多通道输入数据时，汇聚层在每个输入通道上单独运算，而不是像卷积层一样在通道上对输入进行汇总。 这意味着汇聚层的输出通道数与输入通道数相同。
==注意点==:对于池化层，如果不显著设定stride，那么stride就会和池化窗口的size相等，对于卷积层，如果不显著设定stride，那么stride的size会默认为1。

- 对于给定输入元素，最大汇聚层会输出该窗口内的最大值，平均汇聚层会输出该窗口内的平均值。
- 汇聚层的主要优点之一是减轻卷积层对位置的过度敏感。
- 我们可以指定汇聚层的填充和步幅。
- 使用最大汇聚层以及大于1的步幅，可减少空间维度（如高度和宽度）。
- 汇聚层的输出通道数与输入通道数相同。
- 没有可学习的参数



### 5、BatchNorm层（批量规范化层）

BatchNorm 是一种线性变换（加权平移），加在非线性激活函数前能更有效地调节输出的分布，使激活函数更“活跃”。在卷积网络中，`BatchNorm2d` 对每个输出通道的所有空间位置（跨 batch 和空间维度）进行统一的归一化，然后再用通道独立的缩放和平移参数恢复表达能力。

批量规范化应用于单个可选层（也可以应用到所有层），其原理如下：在每次训练迭代中，我们首先规范化输入，即通过减去其均值并除以其标准差，其中两者均基于当前小批量处理。 接下来，我们应用比例系数和比例偏移。 正是由于这个基于*批量*统计的*标准化*，才有了*批量规范化*的名称。

请注意，如果我们尝试使用大小为1的小批量应用批量规范化，我们将无法学到任何东西。 这是因为在减去均值之后，每个隐藏单元将为0。 所以，只有使用足够大的小批量，批量规范化这种方法才是有效且稳定的。 请注意，在应用批量规范化时，批量大小的选择可能比没有批量规范化时更重要。

特点：

* 具有可学习参数 $ \ gama 和 \ beita$​
* 作用在：
  * 全连接层和卷积层输出上，激活函数前
  * 全连接层和卷积层输入上
* 对于全连接层，作用在特征维
* 对于卷积层，作用在通道维

---



==总结：==

- `mean` 和 `var` 是**当前小批量**的统计值，仅用于训练时的归一化。
- `moving_mean` 和 `moving_var` 是**长期统计值**，通过指数移动平均累积，专门用于推理模式，确保归一化不依赖于批量大小。
- **训练模式**：使用 `mean` 和 `var` 归一化，并更新 `moving_mean` 和 `moving_var`。
- **推理模式**：直接使用 `moving_mean` 和 `moving_var` 进行归一化。



---

**使用 `Y = gamma * X_hat + beta` 进行缩放和平移**

归一化后的输出计算公式为：

```python
Y = gamma * X_hat + beta
```

其中，`X_hat` 是归一化后的张量（均值接近0，方差接近1）。



**设计初衷**

- **归一化的局限性**：
  - 批量归一化将输入标准化为均值0、方差1（即 `X_hat = (X - mean) / torch.sqrt(var + eps)`）。虽然这有助于稳定训练（通过减少内部协变量偏移），但它限制了网络的表达能力。
  - 如果直接输出 `X_hat`，网络无法学习到适合任务的分布（例如，可能需要非零均值或非单位方差）。
- **引入 `gamma` 和 `beta`**：
  - `gamma`：缩放参数，允许网络调整归一化后输出的方差。
  - `beta`：偏移参数，允许网络调整归一化后输出的均值。
  - 通过公式 `Y = gamma * X_hat + beta`，网络可以学习任意均值和方差的输出分布，恢复或增强模型的表达能力。
- **为何需要缩放和平移**：
  - **灵活性**：`gamma` 和 `beta` 是可学习的参数（通常通过 `nn.Parameter` 定义），通过梯度下降优化，使网络能够根据任务需要调整输出分布。
  - **保持归一化的优点**：归一化（`X_hat`）确保输入到后续层的分布稳定，减少训练中的梯度爆炸或消失问题。
  - **兼容性**：如果网络认为归一化不必要（例如，`gamma=1`, `beta=0`），可以学到接近原始输入的分布，相当于“禁用”归一化的效果。
- **实际意义**：
  - 在卷积层中，每个通道有自己的 `gamma` 和 `beta`，允许不同特征图学习不同的缩放和偏移。
  - 这种设计使批量归一化既能标准化输入（提高训练稳定性），又能保留网络的表达能力（通过学习适当的分布）。





## 11、缓解模型过拟合

### 1、基本概念

训练集：直接参与训练的数据集

验证集：验证训练结果好坏，以便进行后续调整的数据集

测试集：只用一次，用于评估模型好坏

### 2、过拟合和欠拟合

![image-20250813184850091](C:\Users\osquer\Desktop\typora图片\image-20250813184850091.png)

训练误差是模型在训练数据上的预测误差。如果模型在训练过程中无法显著减少这一误差，说明模型可能缺乏足够的复杂性或容量（capacity），无法很好地拟合训练数据中的模式（patterns）。这被称为表达能力不足。泛化误差是训练误差与验证误差的差值，表示模型从训练数据到未见过数据的适应能力。如果泛化误差很小，说明训练误差和验证误差接近，模型对训练数据的拟合程度与对验证数据的表现差异不大。

泛化误差小通常表明模型没有严重过拟合（overfitting），因此可以通过增加模型复杂度（例如增加层数或参数）来进一步降低训练误差，而不会显著损害泛化性能。

泛化误差小的含义: 泛化误差小表明模型没有严重记住训练数据的噪声或特殊模式，而是较为“诚实”地反映了数据的总体趋势。这通常发生在模型过于简单时，限制了其过拟合能力。

“泛化误差很小”: 您可能疑惑为何不直接优化验证误差。泛化误差小只是说明模型未过拟合，但高误差表明模型未达到潜力。

### 3、解决手段

#### 1、欠拟合

可以增加模型层数。

#### 2、过拟合（以下为具体手段）

权重衰减（Weight Decay）和丢弃法（Dropout）是深度学习中常用的正则化技术，旨在防止过拟合（overfitting）并提高模型的泛化能力。以下是对两者应用场景及特点的详细总结，保持专业、清晰的语言结构。

#### **1. 权重衰减 (Weight Decay)**

- 权重衰减通过在损失函数中添加 L2 正则化项，惩罚较大的权重值。其形式为在原始损失 \( L \) 上增加 $\frac{\lambda}{2} \| \theta \|_2^2$，其中 $\theta$是模型参数，$\lambda$ 是正则化强度（超参数）。
- 在 PyTorch 中，通常通过优化器（如 `torch.optim.SGD`）的 `weight_decay` 参数实现。

##### b. 应用场景

- 适用于深层神经网络，尤其是全连接层较多的模型（如 MLP、Transformer）。
- 常用于数据量有限或特征维度较高的情况，以防止模型过于依赖特定权重。

##### c. 实现

- **代码示例**:
  ```python
  import torch
  import torch.nn as nn
  import torch.optim as optim
  
  model = nn.Linear(10, 2)
  optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)  # weight_decay 启用 L2 正则化
  ```
- 优化器在每次更新时自动将权重衰减项加入梯度。

##### d. 特点

- **机制**: 通过缩小权重范数，限制模型复杂度，减少过拟合风险。
- **优点**:
  - 简单高效，易于与梯度下降结合。
  - 对平滑解（robust solution）有正向影响。
- **缺点**:
  - 对所有参数施加相同惩罚，可能不适合需要大权重的层（如批量归一化）。
  - 超参数 $\lambda$ 需要调优。

#### 2、dropout 丢弃法

[参考该连接 ](#2、Dropout层 )

## 12、图像增广

应用图像增广的原因是，随机改变训练样本可以减少模型对某些属性的依赖，从而提高模型的泛化能力。 例如，我们可以以不同的方式裁剪图像，使感兴趣的对象出现在不同的位置，减少模型对于对象出现位置的依赖。 我们还可以调整亮度、颜色等因素来降低模型对颜色的敏感度。优先使用`torchvision.transforms.v2` 

多种图像变换方法仅针对训练集，防止过拟合；测试集不做变换。



### 1. **`ColorJitter` 参数概览**

`ColorJitter` 的构造函数如下：
```python
torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
```
- **`brightness`（亮度）**：
  - 类型：浮点数或元组 `(min, max)`。
  - 作用：随机调整图像的亮度。亮度因子从 `[max(0, 1 - brightness), 1 + brightness]` 中随机采样，图像像素值乘以该因子。
  - **常用值**：`0.1` 到 `0.5`，过大的值可能导致图像过亮或过暗，影响模型训练。
- **`contrast`（对比度）**：
  - 类型：浮点数或元组 `(min, max)`。
  - 作用：随机调整图像的对比度。对比度因子从 `[max(0, 1 - contrast), 1 + contrast]` 中随机采样，图像像素值根据公式 `(pixel - mean) * factor + mean` 调整。
  - **常用值**：`0.1` 到 `0.5`，过大的值可能导致图像细节丢失。
- **`saturation`（饱和度）**：
  - 类型：浮点数或元组 `(min, max)`。
  - 作用：随机调整图像的饱和度。饱和度因子从 `[max(0, 1 - saturation), 1 + saturation]` 中随机采样，影响图像的颜色强度。
  - **常用值**：`0.1` 到 `0.5`，过大的值可能使图像颜色过于鲜艳或褪色。
- **`hue`（色调）**：
  - 类型：浮点数或元组 `(min, max)`。
  - 作用：随机调整图像的色调，色调因子从 `[-hue, hue]` 或指定的 `(min, max)` 范围内采样，调整图像的颜色偏向（如偏红、偏蓝）
  - **常用值**：`0.05` 到 `0.2`，过大的色调变化可能导致图像颜色失真。

#### **使用场景及常用参数**
- **`ColorJitter` 常用参数**：

- - 通用任务：`brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1`
  - 轻量增强：`brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05`
  - 强增强：`brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2`
  - 特定属性：仅设置需要的参数（如 `brightness=0.5, contrast=0.5`）。
- **建议**：
  - 使用 `ToImage` 或 `tv_tensors.Image` 显式标记图像数据，避免 transforms 误判。
  - 根据任务需求调整参数，验证增强效果。
  - 结合其他变换（如 `RandomResizedCrop`、`Normalize`）构建数据增强流水线。

### 2、其他变换方法

[参考API文档](https://docs.pytorch.ac.cn/vision/stable/transforms.html#geometry)

## 13、迁移学习

*迁移学习*（transfer learning）将从*源数据集*学到的知识迁移到*目标数据集*。 ImageNet数据集上训练的模型可能会提取更通用的图像特征，这有助于识别边缘、纹理、形状和对象组合。 

```
pretrained_net = torchvision.models.resnet18(pretrained=True)
```

在使用预训练模型之前，必须对图像进行预处理（使用正确的分辨率/插值调整大小、应用推理转换、重新缩放值等）。没有标准的方法可以做到这一点，因为它取决于给定模型的训练方式。它可能因型号系列、变体甚至重量版本而异。使用正确的预处理方法至关重要，否则可能会导致准确性下降或输出不正确。

每个预训练模型的推理转换的所有必要信息都在其权重文档中提供。为了简化推理，TorchVision 将必要的预处理转换捆绑到每个模型权重中。这些可以通过 `weight.transforms` 属性访问

```
import torchvision.models as models
import torch.nn as nn

# 加载预训练 ResNet-50
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)

# 替换全连接层
num_classes = 10  # 新任务的类别数
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 可选：冻结卷积层以加速训练
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad = True

# 定义预处理（保持 ImageNet 归一化）
preprocess = weights.transforms()


```

---



## 14、图像检测技术

### 1、锚框

* 生成锚框 $(s_1, r_1), (s_1, r_2), \ldots, (s_1, r_m), (s_2, r_1), (s_3, r_1), \ldots, (s_n, r_1).$​,这种组合可以有效减少锚框数量

* 计算交并比，用于衡量锚框和边界框之间的相似性 $J(\mathcal{A},\mathcal{B}) = \frac{\left|\mathcal{A} \cap \mathcal{B}\right|}{\left| \mathcal{A} \cup \mathcal{B}\right|}.$​ 

* 将真实边界框分配给锚框(此时分配的只是标签索引)

  在训练集中，我们将每个锚框视为一个训练样本。 为了训练目标检测模型，我们需要每个锚框的*类别*（class）和*偏移量*（offset）标签，其中前者是与锚框相关的对象的类别，后者是真实边界框相对于锚框的偏移量。 在预测时，我们为每个图像生成多个锚框，预测所有锚框的类别和偏移量，根据预测的偏移量调整它们的位置以获得预测的边界框，最后只输出符合特定条件的预测边界框。

  目标检测训练集带有*真实边界框*的位置及其包围物体类别的标签。 要标记任何生成的锚框。

* 有了带有边界框信息的锚框之后，计算类别和偏移量 $\left( \frac{ \frac{x_b - x_a}{w_a} - \mu_x }{\sigma_x},
  \frac{ \frac{y_b - y_a}{h_a} - \mu_y }{\sigma_y},
  \frac{ \log \frac{w_b}{w_a} - \mu_w }{\sigma_w},
  \frac{ \log \frac{h_b}{h_a} - \mu_h }{\sigma_h}\right),$​  如果一个锚框没有被分配真实边界框，我们只需将锚框的类别标记为*背景*（background）。 背景类别的锚框通常被称为*负类*锚框，其余的被称为*正类*锚框。

* 为锚框分配真实标签

* 使用非极大值抑制预测边界框，因为可能出现多个锚框很相似的情况，此时，可以过滤掉重叠的边界框预测，减少冗余检测，保证效率

==多尺度锚框==

既然每张特征图上都有个不同的空间位置，那么相同空间位置可以看作含有个单元。感受野的定义，特征图在相同空间位置的个单元在输入图像上的感受野相同： 它们表征了同一感受野内的输入图像信息。 因此，我们可以将特征图在同一空间位置的个单元变换为使用此空间位置生成的个锚框类别和偏移量。 本质上，我们用输入图像在某个感受野区域内的信息，来预测输入图像上与该区域位置相近的锚框类别和偏移量。

### 2、语义分割

像素标注和预测区域是像素级别的，并且在处理图像时，不再采用直接缩放的方式，因为这样很可能会导致图像像素和标签不匹配，即使它们一一对应，但原图像也发生了改变，训练也就没意义了。

*图像分割*将图像划分为若干组成区域，这类问题的方法通常利用图像中像素之间的相关性。它在训练时不需要有关图像像素的标签信息，在预测时也无法保证分割出的区域具有我们希望得到的语义。以 [图13.9.1](https://zh.d2l.ai/chapter_computer-vision/semantic-segmentation-and-dataset.html#fig-segmentation)中的图像作为输入，图像分割可能会将狗分为两个区域：一个覆盖以黑色为主的嘴和眼睛，另一个覆盖以黄色为主的其余部分身体。

- *实例分割*也叫*同时检测并分割*（simultaneous detection and segmentation），它研究如何识别图像中各个目标实例的像素级区域。与语义分割不同，实例分割不仅需要区分语义，还要区分不同的目标实例。例如，如果图像中有两条狗，则实例分割需要区分像素属于的两条狗中的哪一条。

### 3、转置卷积

可以增加上采样中间特征图的空间维度，用于逆转下采样导致的空间尺寸减小。

![image-20250819110830115](C:\Users\osquer\Desktop\typora图片\image-20250819110830115.png)

* 在转置卷积中，填充被应用于的输出（常规卷积将填充应用于输入）。 例如，当将高和宽两侧的填充数指定为1时，转置卷积的输出中将删除第一和最后的行与列。
* 步幅被指定为中间结果（输出），而不是输入。 

### 4、全卷积神经网络

*全卷积网络*（fully convolutional network，FCN）采用卷积神经网络实现了从图像像素到像素类别的变换，但又与卷积神经网络不同，全卷积网络将中间层特征图的高和宽变换回输入图像的尺寸，输出的类别预测与输入图像在像素级别上具有一一对应关系：通道维的输出即该位置对应像素的类别预测。

### 5、样式迁移

 首先，我们初始化合成图像，例如将其初始化为内容图像。 该合成图像是风格迁移过程中唯一需要更新的变量，即风格迁移所需迭代的模型参数。 然后，我们选择一个预训练的卷积神经网络来抽取图像的特征，其中的模型参数在训练中无须更新。 这个深度卷积神经网络凭借多个层逐级抽取图像的特征，我们可以选择其中某些层的输出作为内容特征或风格特征。

![../_images/neural-style.svg](https://zh.d2l.ai/_images/neural-style.svg)

 风格迁移常用的损失函数由3部分组成：

1. *内容损失*使合成图像与内容图像在内容特征上接近；
2. *风格损失*使合成图像与风格图像在风格特征上接近；
3. *全变分损失*则有助于减少合成图像中的噪点。
