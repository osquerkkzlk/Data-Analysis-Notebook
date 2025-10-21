# python中的算法与leetcode题目

## * ——值得n刷的题

1. 53（最大子数组和问题），真的难

## *  ——感悟

1. range函数的参数复杂度要尽可能低，这样方便理解
2.  到起点的那些值赋值就ok了

## 1. Python 算法

### 1、 for _ in range(n)

重点在于_作为占位符仅用于计数，而range(n)则表示迭代n次。

==进阶==：

```python
mylist=[content for _ in range(n)]
#content会重复n次放入mylist中

mylist2=[[content for _ in range(n)]for _ in range(m)]
#content会先重复n次放入一个列表中，然后该列表再重复m次放入mylist2中
```

### 2、startswith 方法

用于检查字符串是否以指定的前缀开始。如果字符串以该前缀开始，则返回`True`；否则返回`False`

### 3、动态规划DP问题😍

==方法上的选取：DFS>记忆化搜索>dp==

动态规划的的四个解题步骤是：

- **定义子问题-*

  > ==子问题是和原问题相似，但规模较小的问题。==

<u>动态规划的核心要点实际上就是通过求这一堆子问题的解，来求出原问题的解。</u>这要求子问题需要具备两个性质：1）原问题要能由子问题表示。2）一个子问题的解要能通过其他子问题的解求出，即*f*(*k*) 可以由 *f*(*k*−1) 和 *f*(*k*−2) 求出，这个性质就是教科书中所说的“最优子结构”。

- **写出子问题的递推关系**

  ✔️尤其要注意边界条件和逻辑递推关系

- **确定 DP 数组的计算顺序**

  一般而言，计算顺序是自底而上，使用DP数组的循环方法。

  > ==DP 数组也可以叫”子问题数组”，因为 DP 数组中的每一个元素都对应一个子问题。==

- **空间优化（可选）**

1. 确定dp数组及下标定义
2. 确定递推公式
3. dp数组如何进行初始化
4. 确定遍历顺序
5. 举例推导dp数组

#### 1.子数组或者子串问题

可以参考如下代码（求连续子数组的最大和），pre是前一个子数组中的和，那么问题的关键点就是判断前一个子数组和的正负，如果是负数就另起灶台，如果是非负数就把当前元素加到子数组上面，并实时与ans比较找到最大值

```
pre=max(pre,0)+lst[i]
ans=max(ans,pre)
```

✔️例题——给你一个整数数组 `nums` ，请你找出数组中乘积最大的非空连续 子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。**（力扣152）**

```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        fmin,fmax,ans=nums[0],nums[0],nums[0]
        for i in range(1,len(nums)):
            fmax,fmin=max(fmin*nums[i],fmax*nums[i],nums[i]),\
                      min(fmin*nums[i],fmax*nums[i],nums[i])
            ans=max(ans,fmax)
        return ans
```

点评：本代码把fmax和fmin放在了一起写，这样就有效避免了先求fmax而对后求fmin产生的干扰。其次代用fmax和fmin来替代pre，这样避免了对pre和当前元素正负的判断，妙极了。

------

✔️例题——给你一个下标从 **0** 开始的整数数组 `nums` ，它表示一个班级中所有学生在一次考试中的成绩。老师想选出一部分同学组成一个 **非空** 小组，且这个小组的 **实力值** 最大，如果这个小组里的学生下标为 `i0`, `i1`, `i2`, ... , `ik` ，那么这个小组的实力值定义为 `nums[i0] * nums[i1] * nums[i2] * ... * nums[ik]` 。请你返回老师创建的小组能得到的最大实力值为多少。**(力扣2708)**

```python
class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        fmin,fmax=nums[0],nums[0]
        for num in nums[1:]:
            fmax,fmin=max(fmax*num,num,fmin*num,fmax),\
                      min(fmax*num,num,fmin*num,fmin)
        return fmax
```

点评：本代码仍然采用了fmax和fmin的写法，并且有了重大是收获——对于子数组而言，fmax不能把之前的fmax直接拿过来，如果这样就不连续了。对于子序列而言，则可以把之前的fmax直接拿过来使用，因为子序列本身就是不连续的。并且，如果定义fmax和fmin的时候直接赋值1，在子序列问题中很容易无中生有，答案可能会出现1，所以给他们赋值`nums[0]`即可。

------



### 4、回文数|回文字符串

#### 1.反转数字

对于数字x而言：我们可以先==进行判断哪些数字一定不是回文数==——负数、末尾为0的数字（0其实也是回文数）。x/10相当于取出数字的末尾数字，我们对末尾数字进行翻转，直到反转后的结果不小于原来的数字，此时，如果x为偶数，则两个数字应该相等，必为回文数；如果x为奇数，则反转后的数字/10与x比较，若相等，则也为回文数。

#### 2.找到最长回文子串（==动态规划问题==）

本题的关键点在于二维数组的构建，并通过该数组来判断某一个子串是否是回文子串，同时加上if判断该子串是否长度最小

```python
  dp = [[False] * n for _ in range(n)] 
```

==序列切片==，值得注意的是,下标从i开始，长度为length，则直接用list[i:i+length]，这是因为切片时，最后一个元素不计入（i+length）

代码如下，可供复习,但不完整：（两边夹进方法）

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        length = 1 # 最长回文子串的长度
        start = 0  # 最长回文子串的起始位置
         # dp[j][i]表示子串s[j:i]是否为回文串 
        for i in range(n):
以i为终点，往回枚举起点j
            for j in range(i, -1, -1):
                if i == j:
                    dp[j][i] = True    # 一个字符，一定为回文串
                elif i == j + 1:
                    dp[j][i] = (s[i] == s[j])  # 两个字符，取决于二者是否相等
                else: 
                    dp[j][i] = (s[i] == s[j]) and dp[j+1][i-1]  
```

### 5、背包问题

#### 1.0-1背包

关键点在于递推公式的选取上。其中，dp[i][j]表示从下标为i的物品里面任意取，放入容量为j的背包中，其背包的总价值。

1. i物品不放入背包

   ​	则dp[i][j]与dp[ i-1 ][ j ] 相等，因为我此时没有把i物品放入背包，所以他和i-1物品的价值相同

2. i物品放入背包

   ​	那么dp[i][j]=dp[i - 1] [j - weight[i]] + value[i] （物品i的价值）

3. 以下为模板，其中 num表示物品的重量，nums表示对应的价值，target表示最大容量

![image-20241209150322141](C:\Users\osquer\Desktop\typora图片\image-20241209150322141.png)

### 6. 滑动窗口

滑动窗口需要满足单调性，当右端点元素进入窗口时，窗口元素和是不能减少的。本题 nums 包含负数，当负数进入窗口时，窗口左端点反而要向左移动，导致算法复杂度不是线性的。

==**滑动窗口使用前提：**==

1. 连续子数组。
2. 有单调性。本题元素均为正数，这意味着只要某个子数组满足题目要求，在该子数组内的更短的子数组同样也满足题目要求。

✔️例题：给你一个整数数组 `nums` 和一个整数 `k` ，请你返回子数组内所有元素的乘积严格小于 `k` 的连续子数组的数目。下面是输入条件

- `1 <= nums.length <= 3 * 104`
- `1 <= nums[i] <= 1000`
- `0 <= k <= 106`

```
class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        if k<=1 :return 0
        left,ans,prod=0,0,1
        for right,x in enumerate(nums):
            prod*=x
            while prod>=k:
                prod//=nums[left]
                left+=1
            ans+=right-left+1
        return ans
```

解析：prod神来之笔，跳脱出思维定式，之前我一直把思维停留在了滑窗用加减来做或者用累积函数来做，那么既然我们有了每个right对应的x值，直接x*prod不就好了吗。另一个非常重要的电就是：==每次解题前，尽可能地分析题目，先找出特殊情况（也即容易分析的极端情况）==，这种情况一般是与题目条件有关的。

✔️例题：一个数组的 **分数** 定义为数组之和 **乘以** 数组的长度。比方说，`[1, 2, 3, 4, 5]` 的分数为 `(1 + 2 + 3 + 4 + 5) * 5 = 75` 。给你一个正整数数组 `nums` 和一个整数 `k` ，请你返回 `nums` 中分数 **严格小于** `k` 的 **非空整数子数组数目**。**子数组** 是数组中的一个连续元素序列。

- 我的代码：

```
class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        left,ans=0,0
        for right,x in enumerate(nums):
            while sum(array:=nums[left:right+1])*len(array)>=k:
                left+=1
            ans+=right-left+1
        return ans
```

- 改进后的代码

```
class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        left,ans,s=0,0,0
        for right,x in enumerate(nums):
            s+=x
            while s*(right-left+1)>=k:
                s-=nums[left]
                left+=1
            ans+=right-left+1
        return ans
```

**题解**：通过对比，我们不难看出，之前的代码存在着大量的重复计算（求和）。以下是超时原因分析—**对于大规模输入数据**：当输入的 `nums` 数组长度 `n` 很大时，比如在一些算法竞赛平台或者大规模数据处理场景下， 的时间复杂度执行效率就会变得很低，很容易超出时间限制导致超时。**重复计算问题**：在判断子数组是否满足条件的过程中，每次内层循环调整 `left` 后，都会重新计算当前子数组的和（通过 `sum` 函数），存在大量重复计算。例如，当 `left` 从位置 `i` 移动到 `i + 1` 时，其实新的子数组相比之前只是去掉了最左边的一个元素，==但是代码并没有利用之前已经计算好的子数组和信息，而是重新对整个新子数组进行求和计算，浪费了大量计算资源和时间。==

### 7. 二分法

破题的关键主要是因为**对区间的定义没有想清楚，区间的定义就是不变量**。要在二分查找的过程中，保持不变量，就是==在while寻找中每一次边界的处理都要坚持根据区间的定义==来操作，这就是**循环不变量**规则。写二分法，区间的定义一般为两种，左闭右闭即[left, right]，或者左闭右开即[left, right)。

- ==左闭右闭区间写法==

  区间的定义这就决定了二分法的代码应该如何写，**因为定义target在[left, right]区间，所以有如下两点：**

  - while (left <= right) 要使用 <= ，因为left == right是有意义的，所以使用 <=
  - if (nums[middle] > target) right 要赋值为 middle - 1，因为当前这个nums[middle]一定不是target，那么接下来要查找的左区间结束下标位置就是 middle - 1

- ==左闭右开区间写法==

  如果说定义 target 是在一个在左闭右开的区间里，也就是[left, right) ，那么二分法的边界处理方式则截然不同。有如下两点：

  - while (left < right)，这里使用 < ,因为left == right在区间[left, right)是没有意义的
  - if (nums[middle] > target) right 更新为 middle，因为当前nums[middle]不等于target，去左区间继续寻找，而寻找区间是左闭右开区间，所以right更新为middle，即：下一个查询区间不会去比较nums[middle]

✔️例题：给你一个按照非递减顺序排列的整数数组 `nums`，和一个目标值 `target`。请你找出给定目标值在数组中的开始位置和结束位置。如果数组中不存在目标值 `target`，返回 `[-1, -1]`。

```
class Solution:
    def way(self, nums: List[int], target: int) -> List[int]:
        left,right=0,len(nums)-1
        while(left<=right):
            mid=(left+right)//2
            if nums[mid]>=target:right=mid-1
            if nums[mid]<target: left=mid+1
        return left
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        start=self.way(nums,target)
        if start==len(nums) or nums[start]!=target:
            return [-1,-1]
        end=self.way(nums,target+1)-1
        return [start,end]
```

本代码的巧妙之处是通过求target和target+1的下标直接确定了范围，非常奈斯。==同时，我们需要注意采用二分法初步得到的left只是大于等于target的第一个值，所以我们还要进行判断left是否越界，left是否对应target，这一道理在其他二分查找题型中同样适用==

✔️例题：给你一个整数数组 `nums` 和一个正整数 `threshold` ，你需要选择一个正整数作为除数，然后将数组里每个数都除以它，并对除法结果求和。请你找出能够使上述结果小于等于阈值 `threshold` 的除数中 **最小** 的那个。每个数除以除数后都向上取整，比方说 7/3 = 3 ， 10/2 = 5 。题目保证一定有解。==（求最小）==

```python
from bisect import bisect_left
class Solution:
    def smallestDivisor(self, nums: List[int], threshold: int) -> int:
        rule=lambda x: sum((num-1)//x for num in nums)<=threshold-len(nums)
        return bisect_left(range(max(nums)),True,1,key=rule)
```

bisect_left本质上就是建立在二分查找上的

### 8、分组循环

外层循环负责数组遍历之前的准备工作，记录数组开始位置以及记录数组长度，内层循环负责遍历数组，找出改组在哪里结束。分组循环适合场景：数组被划分成若干组，每一组的处理逻辑都是相同的。

### 9、二进制数字转换成十进制数字，超绝思路

每读取链表的一个节点值，可以认为读到的节点值是当前二进制数的最低位；当读到下一个节点值的时候，需要将已经读到的结果乘以2，再将新读到的节点值当作当前二进制数的最低位；如此进行下去，直到读到了链表的末尾。

```python
class Solution:
    def getDecimalValue(self, head: ListNode) -> int:
        cur = head
        ans = 0
        while cur:
            ans = ans * 2 + cur.val
            cur = cur.next
        return ans

```

### 10、哈希集合

由于直接判断节点值是否在 nums 中，需要遍历 nums，时间复杂度为 O(n)。通过把 nums 中的元素加到一个哈希集合中，然后判断节点值是否在哈希集合中，这样可以做到每次判断时间复杂度为 O(1)。

```python
class Solution:
    def modifiedList(self, nums: List[int], head: Optional[ListNode]) -> Optional[ListNode]:
        st = set(nums)#set是一个哈希集合
        cur = dummy = ListNode(next=head)
        while cur.next:
            if cur.next.val in st:
                cur.next = cur.next.next  # 删除
            else:
                cur = cur.next  # 向后移动
        return dummy.next
```

- 普通数组
  - 当需要频繁通过索引访问元素时，数组是一个很好的选择，例如在处理矩阵、图像等数据时。
  - 当元素的顺序很重要，并且需要保持插入顺序时，数组更合适。
- 哈希集合
  - 当需要快速判断一个元素是否存在于集合中时，哈希集合的效率更高，例如在去重、查找重复元素等场景中。
  - 当需要处理大量的数据，并且对元素的顺序没有要求时，哈希集合可以提供更高效的插入、删除和查找操作。

### 11、链表

#### 1、链表反转

```python
cur=head
tail=None
while cur:
    temp=cur.next
    cur.next=tail
    tail=cur
    cur=temp
```

值得注意的是，反转链表之后，链表的结构会发生改变，不会复原哦，除非再次反转链表

#### 2、链表中间节点

```
fast=head
slow=head
while fast and fast.next:
    fast=fast.next.next
    slow=slow.next
return slow
```

值得注意的是，`slow`表示中间节点（靠前），`slow.next`则表示题目要求的中间节点（靠后）

## 2、新的语法知识点

### 1、operational

`Optional`类型，它实际上是一个类型别名，等同于`Union[type, None]`。这允许开发者在函数注解中指定一个参数或返回值可以是特定的类型或者`None`。从Python 3.10开始，`Optional`类型被标记为过时，并且推荐直接使用`Union`类型来替代`Optional`。，`Union[str, None]`直接表示`name`可以是`str`类型或者`None`。

### 2、and is not语法

**`or` 是为了提前返回正确结果，不用额外搜索另一边，提高效率。**

**`and` 可能会导致错误，因为它需要两侧都返回非空值，而目标节点只会出现在一侧。**

在Python中，`is` 是一个内置的运算符，用于比较两个对象的标识（即它们是否为同一个对象）。当你使用 `is` 来比较两个变量时，你实际上是在检查它们是否引用了内存中的同一个对象。and可以作为c++中的&&来使用，or可以作为c++中的||来使用，is可以作为c++中的来使用.==真的很方便！==

```
class Solution:
    def rob(self, nums: [int]) -> int:
        def my_rob(nums):
            cur, pre = 0, 0
            for num in nums:
                cur, pre = max(pre + num, cur), cur
            return cur
        return max(my_rob(nums[:-1]),my_rob(nums[1:])) if len(nums) != 1 else nums[0]

```

#### 1. is与==的判别

在Python中，`is` 关键字用于比较两个对象的**身份**，即它们是否为同一个对象，而不是比较它们的**值**。这与使用 `==` 比较运算符不同，后者用于比较两个对象的值是否相等。

==延伸，用于判断二叉树的叶子节点==

```
root.left is root.right(叶子节点的特征)
```



- `is`：用于检查两个变量引用是否指向同一个对象。
- `==`：用于检查两个对象的值是否相等。

**1. 示例**

```python
# 基本用法
a = [1, 2, 3]
b = [1, 2, 3]
c = a

# 检查值是否相等
print(a == b)  # 输出 True，因为两个列表包含相同的元素
print(a == c)  # 输出 True，因为 a 和 c 引用同一个列表

# 检查身份是否相同
print(a is b)  # 输出 False，因为 a 和 b 是不同的对象
print(a is c)  # 输出 True，因为 a 和 c 引用同一个对象
```

**2. 特点和注意事项**

1. **对象身份**：`is` 关键字检查的是对象的身份，即它们是否为同一个对象。如果两个变量引用同一个对象，`is` 将返回 `True`。

2. **内存地址**：在底层，`is` 比较的是对象的内存地址。如果两个变量指向同一个内存地址，`is` 返回 `True`。

3. **不可变对象**：对于不可变对象（如整数、浮点数、字符串和元组），Python 可能会重用相同的对象以节省内存。因此，即使两个变量看起来是分别创建的，它们也可能引用同一个对象。

4. **可变对象**：对于可变对象（如列表、字典和集合），除非明确地让它们引用同一个对象，否则即使它们的值相同，`is` 也会返回 `False`。

5. **单例对象**：Python 中的一些对象是单例，例如 `None`、`True` 和 `False`。这些对象在程序中是唯一的，所以 `is` 可以用来检查一个变量是否为 `None`。

**3. 实际应用**

1. **检查 None**：`is None` 是检查变量是否为 `None` 的推荐方式，因为它直接比较对象的身份。

```python
x = None
if x is None:
    print("x is None")
```

2. **比较内置类型**：由于 ==Python 可能会重用不可变对象==，使用 `is` 来比较两个看似独立的不可变对象可能是有用的。

```python
a = 10
b = 10
print(a is b)  # 通常输出 True，因为 Python 可能会重用相同的整数对象
```

3. **避免使用 `==` 比较不可变对象**：在比较不可变对象时，如果关心的是对象的身份而不是值，应使用 `is` 而不是 `==`。

------



### 3、pow和math.pow()函数

**`pow()`** 的时间复杂度均为 *O*(log*a*) ；而 **`math.pow()`** 始终调用 C 库的 `pow()` 函数，其执行浮点取幂，时间复杂度为 *O*(1) 。如果使用了math.pow( )函数，在计算int类型的整数时，我们可能需要用到int( ) 进行强制转换

### 4、二叉搜索树

二叉树里每个节点都是一个爸爸，每个爸爸有两个儿子，而二叉“搜索”树就是要满足一个额外的条件：所有左儿子的数字都比爸爸数字小，所有右儿子的数字都比爸爸数字大。其实就是**二分查找**所抽象出来的。

案例如下：

![屏幕快照 2020-07-03 下午12.04.44.png](https://pic.leetcode-cn.com/0219df381cfbd02130b76c0af1d149b6013283d934195c7bc6feab4372b794bd-%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202020-07-03%20%E4%B8%8B%E5%8D%8812.04.44.png)

### 5、*号的一些常见用途

1. 解包操作符，用于将序列中的元素解包为独立的参数或变量。

```
def func(a, b, c):
    print(a, b, c)

args = [1, 2, 3]
func(*args)  # 这等同于 func(1, 2, 3)
```

### 6、@cache装饰器

在Python中，`@cache` 是一个装饰器，用于缓存函数的返回值。这个装饰器通常用于那些计算成本高昂的函数，以避免重复计算相同的结果。

```
@cache
def expensive_function(param):
    # 这里是一些计算成本高昂的操作
    return result
```

当`expensive_function`被调用时，如果参数`param`相同，那么函数的结果会被缓存起来，下一次调用时直接返回缓存的结果，而不是重新计算。

### 7、enumerate函数（适用于可迭代对象）

`enumerate` 是一个内置函数，它用于将一个可遍历的数据对象（如列表、元组或字符串）组合为一个索引序列，同时列出数据和数据下标。这在循环遍历列表时尤其有用，因为它允许你同时获得元素的索引和值。

```
my_list = ['apple', 'banana', 'cherry']
for index, item in enumerate(my_list):
    print(index, item)
```

同时会输出：

```
0 apple
1 banana
2 cherry
```

### 8、ASCII码+str容器(chr||ord)

#### 1.大小写字母转换

1. ==大写字母+32为小写字母==

大写字母 'A' 的ASCII码值是65，小写字母 'a' 的ASCII码值是97。这里我们可以利用chr（）函数和ord（）函数进行处理代码点（该字符的数值表示）

- **ord() 函数**

`ord()` 函数用于获取一个字符（字符是字符串中的单个元素）的ASCII（或Unicode）代码点（即该字符的数值表示）。这个函数接受一个字符作为参数，并返回对应的整数。

```
ascii_value = ord('A')                   #值为65
```

- **chr() 函数**

`chr()` 函数是 `ord()` 函数的逆操作，它用于将一个整数（ASCII或Unicode代码点）转换为对应的字符。这个函数接受一个整数作为参数，并返回该整数对应的字符。

```
char = chr(65)                           #值为'A'
```

#### 2使用内置函数

- **str.lower()**

`str.lower()` 方法用于将字符串中的所有大写字母转换为小写字母。

- **str.upper()**

`str.upper()` 方法用于将字符串中的所有小写字母转换为大写字母。

- **str.capitalize()**

`str.capitalize()` 方法用于将字符串的第一个字母转换为大写，其余字母转换为小写。

- **str.swapcase()**

`str.swapcase()` 方法用于将字符串中的大写字母转换为小写字母，小写字母转换为大写字母。

- **str.title()**

`str.title()` 方法用于将字符串中每个单词的首字母转换为大写，其余字母转换为小写。



### 9、defaultdict方法

`defaultdict` 是 Python 标准库 `collections` 模块中的一个类，它继承自内置的 `dict` 类。`defaultdict` 的特殊之处在于，它允许你为字典提供一个默认的工厂函数，这个工厂函数会在你尝试访问不存在的键时被调用，并返回一个默认值。

使用 `defaultdict` 可以避免在访问字典时出现 `KeyError` 异常，因为它会自动为不存在的键创建一个默认值。这对于某些需要频繁添加元素到字典的场景非常有用，尤其是当你需要一个键对应的值是一个列表或集合时。

```
from collections import defaultdict

# 创建一个默认值为 int 的 defaultdict，int() 默认值为 0
dd_int = defaultdict(int)
dd_int['a'] += 1  # 自动创建键 'a' 并设置其值为 1
print(dd_int['a'])  # 输出: 1
print(dd_int['b'])  # 输出: 0，因为 'b' 不存在，使用默认值 int()

# 创建一个默认值为 list 的 defaultdict
dd_list = defaultdict(list)
dd_list['a'].append(1)  # 自动创建键 'a' 并添加元素 1
print(dd_list['a'])  # 输出: [1]
print(dd_list['b'])  # 输出: [], 因为 'b' 不存在，使用默认值 list()
```

### 10、join方法

`join` 是 Python 中字符串（`str`）类的一个方法，它用于将序列中的元素连接成一个新的字符串。这个方法接受一个可迭代对象（如列表、元组等）作为参数，并返回一个由这些元素连接而成的字符串。`join` 方法通常用于将字符串列表中的元素合并成一个单一的字符串。

```
# 连接字符串列表
words = ["Hello", "World"]
result = " ".join(words)  # 使用空格作为分隔符，同样也可以使用其它字符作为分隔符
print(result)  # 输出: Hello World
```

#### 连接非字符串元素

如果可迭代对象中的元素不是字符串，`join` 方法会抛出 `TypeError`。为了连接非字符串元素，你需要先将它们转换为字符串：

```
# 连接整数列表
numbers = [1, 2, 3]
result = ",".join(str(num) for num in numbers)  # 先将整数转换为字符串
在编写代码中还是很实用的
print(result)  # 输出: 1,2,3
```

### 11、filter函数（可用于过滤）

`filter` 是 Python 中的一个内置函数，它用于过滤序列，过滤掉不符合条件的元素，返回一个迭代器对象，可以使用 `list()` 函数转化为列表。`filter` 函数接受两个参数：第一个参数是一个函数，用于测试所有元素是否符合条件；第二个参数是一个序列（如列表、元组等）。

- #### 过滤掉列表中的偶数，只保留奇数

```
numbers = [1, 2, 3, 4, 5, 6]
odd_numbers = list(filter(lambda x: x % 2 != 0, numbers))
print(odd_numbers)  # 输出: [1, 3, 5]
```

- #### 过滤掉字符串列表中的空字符串

```
words = ["hello", "", "world", " ", "python"]
non_empty_words = list(filter(None, words))
print(non_empty_words)  # 输出: ['hello', 'world', 'python']
```

- #### 注意事项

- `filter` 函数返回的是一个迭代器，如果你需要一个列表，可以使用 `list()` 函数进行转换。
- `filter` 函数不会修改原始序列，它返回一个新的迭代器。
- 如果你提供的测试函数总是返回 `True`，那么 `filter` 函数将返回原始序列的一个副本。
- 如果测试函数可能抛出异常，那么 `filter` 函数也会抛出异常。

### 12、generator迭代器

是一种使用简单且内存高效的迭代器。生成器允许你逐个产生值，而不是一次性将所有值存储在内存中。这使得生成器非常适合处理大数据集或无限序列。**使用 `yield` 关键字：** 当你在一个函数中使用 `yield` 时，Python会将该函数转换为一个生成器。每次调用生成器函数时，它会在上次 `yield` 表达式停止的地方继续执行。

```
def count_up_to(max):
    count = 1
    while count <= max:
        yield count
        count += 1

counter = count_up_to(5)
for number in counter:
    print(number)  # 输出: 1, 2, 3, 4, 5
```

### 13、Counter方法

`collections.Counter` 是 Python 标准库 `collections` 模块中的一个子类，它是一个字典子类，用于计数可哈希对象。它是一个集合，其中的元素存储为字典的键，而它们的计数存储为字典的值。`Counter` 通常用于快速、方便地进行计数操作，比如统计列表中元素的出现次数。

```
from collections import Counter

# 统计元素出现的次数
c = Counter(['red', 'blue', 'red', 'green', 'blue', 'blue'])
print(c)  # 输出: Counter({'blue': 3, 'red': 2, 'green': 1})
```

### 14、items方法

在Python中，`items()` 是一个用于从字典对象中获取其键值对的方法。它返回一个包含所有键值对的视图对象，可以用于迭代。从Python 3.7开始，`dict.items()` 返回的是字典视图对象（dictionary view object），它提供了一个动态视图，反映字典键值对的变化。

- #### 基本用法


```python
# 创建一个简单的字典
my_dict = {'a': 1, 'b': 2, 'c': 3}

# 使用 items() 方法获取字典的键值对
for key, value in my_dict.items():
    print(key, value)
# 输出:
# a 1
# b 2
# c 3
```

- #### 返回类型

`items()` 方法返回的是一个字典视图对象，它支持迭代，但不是列表，所以它不会复制字典的内容，而是提供一个实时的视图。

- #### 转换为列表


如果你需要一个列表形式的键值对，可以使用 `list()` 函数将视图对象转换为列表：

```python
# 将字典的键值对转换为列表
items_list = list(my_dict.items())
print(items_list)  # 输出: [('a', 1), ('b', 2), ('c', 3)]
```

- #### 字典推导式

`items()` 方法经常与字典推导式一起使用，用于创建新的字典或修改现有的字典：

```python
# 使用字典推导式和 items() 创建一个新的字典
new_dict = {k: v * 2 for k, v in my_dict.items()}
print(new_dict)  # 输出: {'a': 2, 'b': 4, 'c': 6}
```

- #### 在 `collections` 模块中的使用

`items()` 方法也与 `collections` 模块中的一些类如 `defaultdict` 和 `Counter` 一起使用：

```python
from collections import Counter

# 使用 Counter 并获取其键值对
counter = Counter(['a', 'b', 'a', 'c', 'b', 'c'])
for item in counter.items():
    print(item)
# 输出:
# ('a', 2)
# ('b', 2)
# ('c', 2)
```

### 15、 range特点

```
range(i,i+mxaxlen)或者range(i+maxlen,i,-1)		都表示序列切片长度为maxlen
```

### 16、map函数

在Python中，`map()`函数是一个非常有用的内置函数，它允许你对可迭代对象（如列表、元组、集合等）中的每个元素应用一个给定的函数，并返回一个迭代器，该迭代器生成应用函数后的结果。==实际上，map函数为对列表等对象中的每一个元素应用传入的函数，并返回一个map对象，我们可以使用list等使结果转换为常见的形式。一个字，妙！==

```
map(function, iterable, ...)
```

示例：

```
numbers = [1, 2, 3, 4, 5]
squared = map(lambda x: x**2, numbers)
print(list(squared))  # 输出: [1, 4, 9, 16, 25]
```

### 17、set容器

```
words = set(wordDict)
```

- **去重**：如果 `wordDict` 中有重复的单词，转换成集合后会自动去除重复项。
- ==**查找效率**==：集合在 Python 中是基于哈希表实现的，这意味着检查一个元素是否存在于集合中（即查找操作）的平均时间复杂度是 O(1)，这比列表的查找效率（平均 O(n)）要高得多。
- **内存使用**：虽然集合可能会比列表使用更多的内存，但考虑到查找效率的提升，这是一个合理的权衡。

### 18、下划线与数字


​	在编程中，数字 `1_000_000_007` 是一个整数，它使用了Python中的下划线分隔符来提高数字的可读性。这种分隔符在数字的任意两个数字之间，可以是任意数量的下划线，用于分隔数字，使得大数字更容易阅读和理解。

例如，`1_000_000_007` 实际上是 `1000000007`，这是一个十位数，读作“十亿零七”。

### 19、模运算

```python
MOD = 1_000_000_007

// 加
(a + b) % MOD

// 减
(a - b + MOD) % MOD

// 把任意整数 a 取模到 [0,MOD-1] 中，无论 a 是正是负
(a % MOD + MOD) % MOD

// 乘（注意使用 64 位整数）
a * b % MOD

// 多个数相乘，要步步取模，防止溢出
a * b % MOD * c % MOD

// 除（MOD 是质数且 b 不是 MOD 的倍数）
a * qpow(b, MOD - 2, MOD) % MOD

```

### 20、bisect包（相当重要的）

#### 1、 bisect_left

`bisect_left`的主要作用是在一个已排序的序列中查找给定元素可以插入的最左位置（索引），以维持序列的有序性。也就是说，它找到的位置是在该位置插入目标元素后，序列依然保持有序的最左边的那个索引值。

==函数语法==

`bisect_left(a, x, lo=0, hi=None)`，各参数含义如下：

- `a`：表示要操作的已排序的序列（通常是列表等可变序列类型）。
- `x`：要插入的目标元素，也就是查找该元素在序列中合适插入位置的那个元素。
- `lo`（可选参数，默认值为 0）：指定查找范围的起始索引，即从序列的这个索引位置开始查找。
- `hi`（可选参数，默认值为`None`，若为`None`则表示序列 `a` 的长度，也就是查找范围到序列末尾结束）：用于指定查找范围的结束索引（不包含该索引对应的元素）。

==示例==

- ```python
  import bisect
  
  a = [1, 3, 3, 6, 8, 9]
  x = 3
  result = bisect.bisect_left(a, x)
  print(result)
  ```

### 21.列表技巧

1. 对列表进行扩充，可以直接相乘

```
lst=lst*2
```

2. 循环中的临时变量会额外增加系统开销，比较耗时，如下所示：

```
代码一：
# 否则，k>=2
for i in range(2*length):
    pre=max(pre,0)+(arr*2)[i]
    ans=max(pre,ans)
return max(ans,0,ans+(k-2)*sum(arr)) % MOD

代码二：
# 否则，k>=2
cur=arr*2
for i in range(2*length):
    pre=max(pre,0)+cur[i]
    ans=max(pre,ans)
return max(ans,0,ans+(k-2)*sum(arr)) % MOD
```

### 22、sorted方法

在 Python 中，`sorted()` 是一个内置函数，用于对可迭代对象进行排序并返回一个新的已排序的列表，原可迭代对象不会被修改。

#### 1、基本语法

- ```
  sorted(iterable, key=None, reverse=False)
  ```

  - `iterable`：表示要排序的可迭代对象，如列表、元组、字符串、字典等。
  - `key`：是一个可选参数，用于指定一个函数，该函数将作用于可迭代对象的每个元素上，并根据该函数的返回值进行排序。
  - `reverse`：也是一个可选参数，默认值为 `False`，表示按升序排序；如果设置为 `True`，则按降序排序。

==特殊点——字典的排序操作==

```python
my_dict = {'apple': 3, 'banana': 1, 'cherry': 4, 'date': 1}
sorted_dict_keys = sorted(my_dict.keys())
sorted_dict_values = sorted(my_dict.values())
sorted_dict_items = sorted(my_dict.items())
print(sorted_dict_keys)  
print(sorted_dict_values)  
print(sorted_dict_items)  
```

对字典的键进行排序时，`sorted(my_dict.keys())` 会返回一个按照键的顺序排序后的新列表。对字典的值进行排序时，`sorted(my_dict.values())` 会返回一个按照值的顺序排序后的新列表。对字典的项进行排序时，`sorted(my_dict.items())` 会返回一个按照键的顺序排序后的新列表，每个元素是一个键值对组成的元组。

#### 2、使用 key 参数进行自定义排序

它允许你指定一个函数，该函数将作用于可迭代对象的每个元素上，并根据该函数的返回值进行排序。举例：按照元素长度进行排序

```
words = ["apple", "banana", "cherry", "date", "elderberry"]
sorted_words = sorted(words, key=len)
print(sorted_words)  
```

- 上述代码中，`key=len` 表示根据每个单词的长度进行排序，返回一个新的按照单词长度升序排列的列表 `['date', 'apple', 'banana', 'cherry', 'elderberry']`。

#### 3、使用 reverse 参数进行降序排序

```
nums = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
sorted_nums_desc = sorted(nums, reverse=True)
print(sorted_nums_desc)  
```

上述代码中，`reverse=True` 使得 `sorted()` 函数对列表 `nums` 进行降序排序，返回一个新的已排序的列表 `[9, 6, 5, 5, 5, 4, 3, 3, 2, 1, 1]`。

### 23. 数组处理

#### 1. 删除元素

数字组删除元素只能通过数据覆盖来实现

```python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        k=0
        for x in nums:
            if x!=val:
                nums[k]=x
                k+=1
        return k
```

### 24.海象运算符

海象运算符，即 `:=`，是 Python 3.8 引入的一个新特性，它的主要作用是在表达式内部进行赋值并返回赋值后的结果，使得代码更加简洁和高效。

#### 1、基本语法和使用场景

- **语法**：`:=` 将赋值和表达式求值结合在一起，它可以在需要使用变量的同时进行赋值，而不需要先单独进行赋值操作再使用变量。其基本形式为 `(变量 := 表达式)`，其中括号在某些情况下可以省略，但为了清晰起见，通常建议加上括号。
- **使用场景**：主要用于简化代码，特别是在需要在条件判断、循环条件、列表推导式等中使用变量并同时进行赋值的情况，避免了先赋值再使用的繁琐步骤，使代码更加紧凑和易读。

例子：

```
my_list = [1, 2, 3, 4, 5]
for index, element in enumerate(my_list):
    if (len := len(my_list)) > index + 1:
        print(index, element)
    else:
        break
```

### 25、生成器表达式

**语法形式 生成器表达式的语法形式与列表推导式很接近，它使用圆括号 `()` 来包裹表达式，基本结构是 `(表达式 for 变量 in 可迭代对象 [if 条件])`。**例如：`(x**2 for x in range(10))` 就是一个生成器表达式，它会生成 `0` 到 `9` 每个数的平方所构成的序列（但不会立即生成所有结果，而是按需生成）。

 而列表推导式使用方括号 `[]`，如 `[x**2 for x in range(10)]` 会立即创建一个包含 `0` 到 `9` 每个数平方的列表。

**按需生成数据**：生成器表达式并不会像列表推导式那样一次性计算并存储所有结果，而是在迭代过程中，每次需要一个值时才去计算并返回那个值，这使得它在处理大量数据或者无限序列时非常节省内存空间。例如，假设有一个非常大的数字范围，要获取其中每个数的平方，如果使用列表推导式生成列表，可能会占用大量内存，但使用生成器表达式则只会在每次迭代获取元素时计算相应的平方值，内存占用量随迭代逐步增加，而不是一开始就开辟很大空间来存储所有结果。 -

**迭代器协议**：生成器表达式实现了迭代器协议，意味着它可以使用 `next()` 函数来逐个获取元素，或者在 `for` 循环等可迭代的上下文中使用。例如：

```python
python gen_expr = (x**2 for x in range(5)) print(next(gen_expr)) print(next(gen_expr)) for num in gen_expr:    
print(num) ``` 在上述代码中，首先通过 `next()` 
```

 **内存占用**：列表推导式会一次性创建整个列表，将所有元素都存储在内存中，适合数据量较小且需要多次随机访问元素的情况；而生成器表达式是按需生成元素，更适合处理大量数据或者只需遍历一次元素的场景，能有效节省内存。 

**可迭代次数**：列表推导式生成的列表可以多次迭代，每次都能获取到完整的元素列表；生成器表达式对应的生成器一般只能迭代一次，因为它是按需生成数据，第一次迭代完后就不会再重新生成数据了（除非重新创建生成器表达式对象）。

### 26、next语句

在Python中，`next()` 是一个内置函数，主要用于迭代器（包括生成器，因为生成器也是一种特殊的迭代器）的操作.

**功能**：用于获取迭代器的下一个元素。当你创建了一个迭代器对象（比如通过生成器表达式、包含 `yield` 语句的函数等方式生成的迭代器）后，使用 `next()` 函数可以按照顺序逐个获取其中的元素。 

**语法**：`next(iterator[, default])`，其中 `iterator` 是要操作的迭代器对象，是必须要传入的参数；而 `default` 是一个可选参数，当迭代器耗尽（即已经迭代完所有元素，没有下一个元素了）时，如果指定了 `default` 值，就会返回这个默认值，而不会抛出 `StopIteration` 异常，若不指定 `default` 值，迭代器耗尽时则会抛出 `StopIteration` 异常。 

**迭代器**：当不确定迭代器是否还有下一个元素时，可以结合 `next()` 函数的默认值参数来避免因迭代器耗尽而抛出异常，使代码更加健壮，比如在读取一个不确定长度的可迭代对象中的元素时，可以通过指定默认值来应对元素已经全部读取完的情况。 总之，`next()` 函数在Python中是操作迭代器的一个重要工具，通过它能够灵活地控制迭代的过程并获取相应的元素。 

### 27、yield语句

基本概念与作用 - “yield”语句用在函数中，它使得这个函数变成了一个生成器函数。与普通函数不同的是，普通函数执行到 `return` 语句时会返回结果并结束函数执行，而生成器函数在执行到 `yield` 语句时，会暂停函数的执行，并返回 `yield` 后面表达式的值，同时记住函数执行的当前状态（包括局部变量的值、执行位置等信息），以便下次从这个状态继续执行。 

语法形式与示例 - 语法形式上，就是在函数体中使用 `yield` 关键字，后面跟着要返回的值（可以是各种数据类型）。 ### 使用方式与效果 - 当调用生成器函数时，并不会像普通函数那样立即执行函数体中的所有代码，而是返回一个生成器对象。例如： ```python gen = my_generator() ``` 这里 `my_generator()` 返回的就是一个生成器对象 `gen`。 - 可以通过多种方式来获取这个生成器对象产生的值，比如使用 `next()` 函数（前面介绍过）或者在 `for` 循环中使用它（因为生成器实现了迭代器协议，可在迭代的上下文中使用）。

```python
python gen = my_generator() 
print(next(gen))  
print(next(gen))  
for num in gen:  
	print(num)
```

 在上述代码中，首先通过两次调用 `next()` 函数分别获取了生成器产生的前两个值 `1` 和 `2`，然后通过 `for` 循环获取剩余的值（这里就是 `3`）。每次执行到 `yield` 语句时，函数暂停，返回相应的值，下次再获取元素时，函数从上次暂停的位置继续执行，直到遇到下一个 `yield` 语句或者函数体结束。 

### 28、del语句

在 Python 中，`del` 是一个重要的语句，用于删除对象。

#### 1.、基本语法与作用

`del` 语句的语法形式较为简单，通常写作 `del <对象名>`，这里的对象名可以是变量、列表中的元素、字典中的键值对、对象的属性等各种 Python 中的对象。其核心作用就是从内存中移除指定的对象或者对象的某个部分，释放相应的内存空间，并且使对应的对象标识符（比如变量名）不再指向原来的对象（如果是删除变量的情况）。

#### 2. 在不同数据结构中的使用示例

##### （1）删除变量

```python
a = 10
print(a)  # 输出 10
del a
# print(a)  # 这行代码会报错，因为变量a已经被删除，不存在了
```

在上述代码中，首先定义了变量 `a` 并赋值为 `10`，然后通过 `del a` 语句将变量 `a` 删除，后续再尝试访问 `a` 就会引发错误，因为它已经不存在于当前的命名空间中了。

##### （2）删除列表元素

```python
my_list = [1, 2, 3, 4]
del my_list[1]  # 删除索引为1的元素（也就是值为2的元素）
print(my_list)  # 输出 [1, 3, 4]
```

这里使用 `del` 语句并指定列表的索引，可以删除列表中对应的元素。需要注意的是，删除元素后，列表的长度会相应减少，后续元素的索引也会发生改变。

##### （3）删除字典键值对

```python
my_dict = {'key1': 'value1', 'key2': 'value2'}
del my_dict['key1']
print(my_dict)  # 输出 {'key2': 'value2'}
```

对于字典，通过 `del` 语句指定要删除的键，可以移除对应的键值对。如果指定的键不存在于字典中，那么会引发 `KeyError` 异常，所以在使用 `del` 删除字典键值对时，要确保键是存在的。

##### （4）删除对象属性（针对自定义类实例）

假设定义了如下简单的类：

```python
class MyClass:
    def __init__(self):
        self.attribute = 10

obj = MyClass()
print(obj.attribute)  # 输出 10
del obj.attribute
# print(obj.attribute)  # 这行代码会报错，因为属性已被删除
```

当创建了类的实例 `obj` 后，通过 `del` 语句可以删除实例的属性。同样，删除后再尝试访问该属性就会导致错误。

#### 3. 注意事项

- **内存管理与引用计数**：Python 的内存管理机制中包含引用计数这一概念，当使用 `del` 删除对象时，实际上是减少了对象的引用计数。当引用计数降为 `0` 时，Python 的垃圾回收机制会回收该对象占用的内存空间。但对于一些存在循环引用的复杂对象结构，单纯依靠引用计数和 `del` 语句可能无法及时回收内存，需要依赖 Python 的垃圾回收器进行额外处理。
- **谨慎使用以避免意外错误**：由于 `del` 语句会直接删除对象，如果在代码中不小心删除了还需要使用的对象，就会导致程序出现错误，比如函数中误删了某个全局变量，或者在循环中过早删除了还在后续逻辑中依赖的列表元素等情况。所以在使用 `del` 时，要清晰地了解对象的使用范围和生命周期，确保删除操作不会影响程序的正常运行。

### 28、math包

#### 1、isqrt和sqrt

isqrt函数会进行开根并向下取整，最后得到整数，而sqrt函数则会得到浮点数，结果较为精确。

### 29、科学技术法

`1e9+7`实际上表示是浮点数的形式，但可以通过int强制转换为整数格式，在取模运算时需要格外注意

### 30、序列切片和直接复制的区别

在 Python 中，`nums[:]` 和 `temp = nums` 有明显的区别，下面从赋值本质、对原列表的影响等方面进行详细分析。 

1. 赋值本质 - **`temp = nums`**    - 这是一个简单的引用赋值操作。`temp` 和 `nums` 会指向内存中的同一个列表对象。也就是说，它们实际上是同一个列表的不同别名，并没有创建新的列表对象。 - **`nums[:]`**    - 这是使用切片操作来创建一个新的列表对象。`nums[:]` 会生成一个 `nums` 列表的副本，这个副本包含了 `nums` 列表中的所有元素，但它是一个独立的对象，在内存中有自己的存储位置。 
2. .对原列表的影响 - **`temp = nums`**    - 由于 `temp` 和 `nums` 指向同一个列表对象，对 `temp` 进行的任何修改（如添加、删除、修改元素等操作）都会直接反映在 `nums` 上，反之亦然。   

### 31、文件相关的操作

#### 1、相对路径与绝对路径

相对路径是指文件相对于当前文件的路径，写法规则1如下

`./`表示当前文件所在的目录

`../`表示当前文件所在的上层目录

### 32、pairwise

首先需要导入itertools模块,`pairwise` 函数接收一个可迭代对象（例如列表、元组、字符串等）作为输入，然后返回一个迭代器，这个迭代器会生成一系列包含相邻元素对的元组。如果输入的可迭代对象的元素数量少于 2 个，那么 `pairwise` 函数返回的迭代器将为空。

```
from itertools import pairwise

# 示例 1：对列表使用 pairwise
numbers = [1, 2, 3, 4, 5]
pairs = pairwise(numbers)
for pair in pairs:
    print(pair)

# 示例 2：对字符串使用 pairwise
string = "abcde"
pairs = pairwise(string)
for pair in pairs:
    print(pair)
```

### 33、gcd函数

在 Python 中，`math` 模块提供了 `gcd` 函数，你可以直接使用它来计算两个数的最大公约数。同时，也可以手动实现欧几里得算法来计算最大公约数。

```
import math

# 使用 math 模块的 gcd 函数
a = 24
b = 36
result = math.gcd(a, b)
print(f"使用 math.gcd 计算 {a} 和 {b} 的最大公约数: {result}")

# 手动实现欧几里得算法
def gcd(x, y):
    while y:
        x, y = y, x % y
    return x

result_manual = gcd(a, b)
print(f"手动实现计算 {a} 和 {b} 的最大公约数: {result_manual}"
```

### 34、random

Python 的 `random` 模块用于生成**随机数**，包括整数、浮点数、序列随机选择等。

------

#### **常用函数**

#### 1️⃣ **随机整数**

```python
import random

num = random.randint(1, 10)  # 生成 [1,10] 之间的随机整数
print(num)
```

**等价于**：

```python
num = random.randrange(1, 11)  # 生成 [1,10] 之间的随机整数（注意 range() 右端开区间）
```

------

#### 2️⃣ **随机浮点数**

```python
num = random.random()  # 生成 [0,1) 之间的随机浮点数
print(num)
```

**指定范围**：

```python
num = random.uniform(1.5, 3.5)  # 生成 [1.5,3.5] 之间的随机浮点数
```

------

#### 3️⃣ **从列表中随机选择**

```python
arr = ["苹果", "香蕉", "橘子"]
fruit = random.choice(arr)  # 从列表中随机选一个
print(fruit)
```

------

#### 4️⃣ **从列表中随机抽取多个元素**

```python
arr = [1, 2, 3, 4, 5]
subset = random.sample(arr, 2)  # 从 arr 中随机取 2 个不同的元素
print(subset)
```

------

#### 5️⃣ **打乱列表**

```python
arr = [1, 2, 3, 4, 5]
random.shuffle(arr)  # 随机打乱列表
print(arr)
```

------

------

#### **总结**

| 方法                    | 作用                           |
| ----------------------- | ------------------------------ |
| `random.randint(a, b)`  | 生成 `[a, b]` 范围的随机整数   |
| `random.random()`       | 生成 `[0, 1)` 的随机浮点数     |
| `random.uniform(a, b)`  | 生成 `[a, b]` 范围的随机浮点数 |
| `random.choice(seq)`    | 从序列中随机选一个             |
| `random.sample(seq, k)` | 从序列中随机选 `k` 个不同元素  |
| `random.shuffle(seq)`   | **原地**打乱列表               |

### 35、nonlocal

第一，两者的功能不同。global关键字修饰变量后标识该变量是全局变量，对该变量进行修改就是修改全局变量，而nonlocal关键字修饰变量后标识该变量是上一级函数中的局部变量，如果上一级函数中不存在该局部变量，nonlocal位置会发生错误（最上层的函数使用nonlocal修饰变量必定会报错）。

第二，两者使用的范围不同。global关键字可以用在任何地方，包括最上层函数中和嵌套函数中，即使之前未定义该变量，global修饰后也可以直接使用，而nonlocal关键字只能用于嵌套函数中，并且外层函数中定义了相应的局部变量，否则会发生错误（见第一）。
————————————————

#### **🔹 `nonlocal` 适用于不可变变量**

**`nonlocal` 只在你需要修改** **外层函数中的** **不可变变量（int, str, tuple 等）** 时才需要。

但 `s` 是一个**可变对象（set）**，Python 允许我们**直接对它进行修改**，**而不需要 `nonlocal`**。

#### **🔹 总结**

| 变量类型                          | 需要 `nonlocal` 吗？ | 示例                        |
| --------------------------------- | -------------------- | --------------------------- |
| **可变对象**（list, set, dict）   | ❌ 不需要             | `s.add(val)` ✅              |
| **不可变对象**（int, str, tuple） | ✅ 需要               | `x += 1` ❌（需 `nonlocal`） |

👉 `s.add(node.val)` **修改的是 `set` 的内容**，而不是重新赋值 `s`，**无需 `nonlocal`**。
 👉 **如果 `s = new_set` 需要 `nonlocal`**，但这种情况并未发生。

### 36、cur、cur[:]、cur[::]的区别

`cur`、`cur[:]` 和 `cur[::]` 的区别主要涉及 **对象引用** 和 **深浅拷贝**。让我们详细分析它们的不同点。

------

#### **1. `cur`（原列表）**

- `cur` 是一个 **引用（reference）**，它指向原始列表对象。
- 如果 `cur` 发生修改，所有指向 `cur` 的变量都会看到变化。

**示例**

```python
cur = [1, 2, 3]
new_cur = cur  # new_cur 也是指向同一个列表

new_cur.append(4)  # 修改 new_cur
print(cur)   # [1, 2, 3, 4]  也受到了影响
```

✅ `cur` 和 `new_cur` 其实指向的是**同一个列表对象**，修改 `new_cur` 也会修改 `cur`。

------

#### **2. `cur[:]`（浅拷贝）**

- `cur[:]` 是 **一个新的列表对象**，但其中的元素仍然是原列表中的对象（浅拷贝）。
- 修改 `cur[:]` 不会影响 `cur`，但如果 `cur` 里面有**可变对象（如列表、字典）**，修改内部元素仍然会影响原列表。

**示例**

```python
cur = [1, 2, 3]
new_cur = cur[:]  # 创建一个新的列表

new_cur.append(4)
print(cur)      # [1, 2, 3]  ✅ 原列表不受影响
print(new_cur)  # [1, 2, 3, 4] ✅ 只修改了新列表
```

**但如果列表包含可变对象（如嵌套列表），浅拷贝仍然会影响原始对象**

```python
cur = [[1, 2], [3, 4]]
new_cur = cur[:]

new_cur[0].append(99)
print(cur)      # [[1, 2, 99], [3, 4]]  ❌ 原列表的内部对象被修改
print(new_cur)  # [[1, 2, 99], [3, 4]]  ❌
```

✅ `cur[:]` 只复制了外层列表，但内部的 `[1, 2]` 仍然是原来的对象，所以修改 `new_cur[0]` 也会影响 `cur`。

------

#### **3. `cur[::]`（等同于 `cur[:]`，浅拷贝）**

- `cur[::]` 也是一个新的列表对象，与 `cur[:]` **完全等价**，因为 `::` 只是切片的完整写法。

- 语法：

  ```
  cur[start:stop:step]
  ```

  - `start`：起始索引（默认 `0`）
  - `stop`：结束索引（默认 `len(cur)`）
  - `step`：步长（默认 `1`）

**示例**

```python
cur = [1, 2, 3]
new_cur = cur[::]  # 等价于 cur[:]

new_cur.append(4)
print(cur)      # [1, 2, 3]  ✅ 原列表不受影响
print(new_cur)  # [1, 2, 3, 4] ✅
```

**使用步长**

```python
cur = [1, 2, 3, 4, 5, 6]
print(cur[::2])  # [1, 3, 5] ✅ 每隔 2 取一个数
print(cur[::-1]) # [6, 5, 4, 3, 2, 1] ✅ 逆序
```

------

#### **总结**

| 方式      | 说明                                      | 是否新建对象 | 是否影响原列表 | 是否深拷贝 |
| --------- | ----------------------------------------- | ------------ | -------------- | ---------- |
| `cur`     | **引用**（指向原列表）                    | ❌ 否         | ✅ 会影响       | ❌ 否       |
| `cur[:]`  | **浅拷贝**（新列表，但内部对象相同）      | ✅ 是         | ❌ 不影响       | ❌ 否       |
| `cur[::]` | **浅拷贝**（等同于 `cur[:]`，可以加步长） | ✅ 是         | ❌ 不影响       | ❌ 否       |

💡 **如果要真正的深拷贝，可以用 `copy.deepcopy(cur)`**：

```python
import copy
cur = [[1, 2], [3, 4]]
new_cur = copy.deepcopy(cur)  # 完全复制，包括内部对象

new_cur[0].append(99)
print(cur)      # [[1, 2], [3, 4]]  ✅ 原列表不变
print(new_cur)  # [[1, 2, 99], [3, 4]] ✅ 只修改了新列表
```

🚀 **结论**

- `cur[:]` **适用于** 复制普通列表（只包含数值、字符串等不可变对象）。
- `copy.deepcopy(cur)` **适用于** 复制包含**可变对象**（如列表、字典）的复杂结构。

### 37、字典序

字典序是一种对字符串或序列进行排序的方法，其规则类似于字典中单词的排列方式。在这种排序中，首先比较序列的第一个元素；如果相同，则比较第二个元素，依此类推，直到找到不同的元素为止。如果一个序列是另一个序列的前缀，则长度较短的序列被视为更小。

### 38、二进制数快速转换为十进制数

在 Python 中，您可以使用内置的 `int()` 函数将二进制字符串直接转换为十进制整数。该函数的第一个参数是要转换的字符串，第二个参数是进制基数，对于二进制数，这个基数是 2。

**示例：**

```python
binary_str = '1010'
decimal_number = int(binary_str, 2)
print(decimal_number)  # 输出：10
```

在上述示例中，二进制字符串 `'1010'` 被转换为十进制整数 `10`。

**注意事项：**

- 确保输入的字符串仅包含有效的二进制字符（即 `'0'` 和 `'1'`）。
- 如果输入的字符串包含非二进制字符，`int()` 函数将引发 `ValueError`。

**示例：**

```python
binary_str = '1021'  # 非法的二进制字符串
try:
    decimal_number = int(binary_str, 2)
    print(decimal_number)
except ValueError:
    print("输入的字符串不是有效的二进制数。")
```

在此示例中，字符串 `'1021'` 包含非二进制字符 `'2'`，因此会引发 `ValueError`，并输出提示信息。

使用 `int()` 函数是将二进制字符串转换为十进制整数的最简单和直接的方法。

### 39、zip

#### zip 函数的基本概念

zip 是 Python 内置函数，用于将多个可迭代对象（比如列表、元组、数组等）的元素按位置配对组合，返回一个迭代器。它的基本语法是：

python

收起自动换行复制

```
zip(*iterables)
```

- *iterables：表示任意数量的可迭代对象。
- 返回值：Python 3 中返回一个 zip 对象（迭代器），需要用 list() 或其他方法转换为具体的数据结构。

#### 简单示例

python

收起自动换行复制

```
a = [1, 2, 3] b = [4, 5, 6] zipped = zip(a, b) print(list(zipped))  # 输出: [(1, 4), (2, 5), (3, 6)]
```

- zip(a, b) 将 a 和 b 的元素按位置配对，生成元组 (a[i], b[i])。

#### 不同长度的情况

如果传入的可迭代对象长度不同，zip 会以最短的长度为准：

python

收起自动换行复制

```
a = [1, 2, 3] b = [4, 5] print(list(zip(a, b)))  # 输出: [(1, 4), (2, 5)]
```

- 第三个元素被忽略，因为 b 只有 2 个元素。

#### 解包（unzipping）

可以用 * 运算符将 zip 的结果解包：

python

收起自动换行复制

```
zipped = list(zip(a, b))  # [(1, 4), (2, 5)] x, y = zip(*zipped) print(x)  # (1, 2) print(y)  # (4, 5)
```



### 40、extend

`extend` 是 Python 列表（`list`）的一个方法，用于将一个可迭代对象（如列表、元组、字符串等）的所有元素逐个追加到当前列表的末尾 …


System: `extend` 方法讲解

`extend` 是 Python 列表（`list`）的一个方法，用于将另一个可迭代对象（如列表、元组、集合、字符串等）的所有元素逐个追加到当前列表的末尾。

#### 语法
```python
list.extend(iterable)
```

- **`list`**: 要扩展的列表。
- **`iterable`**: 任何可迭代对象（例如列表、元组、集合、字符串等）。
- **效果**: 将 `iterable` 中的每个元素逐个添加到 `list` 的末尾，直接修改原列表，不返回新列表。

#### 特点
- **修改原列表**：`extend` 直接在原列表上操作，不创建新列表。
- **逐个追加**：与 `append` 不同，`extend` 将可迭代对象的每个元素单独追加，而不是将整个对象作为一个元素添加。

#### 示例
```python
# 示例 1：扩展列表
a = [1, 2]
a.extend([3, 4])
print(a)  # 输出：[1, 2, 3, 4]

# 示例 2：与 append 的区别
a = [1, 2]
a.append([3, 4])
print(a)  # 输出：[1, 2, [3, 4]]
a = [1, 2]
a.extend([3, 4])
print(a)  # 输出：[1, 2, 3, 4]

# 示例 3：扩展字符串（字符串是可迭代对象）
a = [1, 2]
a.extend("ab")
print(a)  # 输出：[1, 2, 'a', 'b']
```

#### 关键点
1. **效率**：`extend` 比多次调用 `append` 更高效，因为它一次性追加所有元素。
2. **适用场景**：常用于合并列表、追加多个元素，或将其他可迭代对象的内容融入列表。
3. **注意事项**：确保 `iterable` 是可迭代对象，否则会抛出 `TypeError`。

#### 注意
- 如果需要创建一个新列表而不是修改原列表，可以使用列表拼接（`+` 或 `list1 + list2`），但这会创建新对象，效率稍低。
- 示例：
  ```python
  a = [1, 2]
  b = [3, 4]
  c = a + b  # 创建新列表
  print(c)   # 输出：[1, 2, 3, 4]
  print(a)   # 输出：[1, 2]（原列表不变）
  ```

# python提高

## 1、random.choice

`random.choice` 是 Python 标准库 `random` 中的函数，用于**从序列中随机选一个元素**。

## 2、__getitem__

**很多 Python 操作到底层其实就是在调用 `__getitem__`**。

------

#### 📦 举几个直接调用它的操作：

| 操作                 | 实际触发的魔术方法                                           |
| -------------------- | ------------------------------------------------------------ |
| `obj[0]`             | `obj.__getitem__(0)`                                         |
| `for x in obj`       | 自动从 0 开始不断调用 `__getitem__(i)`，直到抛出 `IndexError` |
| `x in obj`           | 遍历 `__getitem__` 来查找匹配                                |
| `random.choice(obj)` | 随机选索引 → `__getitem__(index)`                            |
| `list(obj)`          | 用 `__getitem__` 来一个个取值                                |
| 切片 `obj[1:5]`      | 调用 `__getitem__(slice(1,5))`                               |

------

#### 🔧 本质原因：

因为 `__getitem__` 是 Python 中**序列协议（sequence protocol）**的一部分。
 只要你实现了它，你的类就被“视为序列”，自动支持以上功能。

------

#### ✅ 总结一句话：

> **`__getitem__` 是序列类的核心接口，一通百通。**

只要你实现它，Python 就能像处理列表一样处理你的对象。强大就强大在这。

## 3、UML

#### 1. 类的组成

- **类名**：类的名称，居中显示
- **属性**：类的成员变量，格式：`[可见性] 名称: 类型`
- **方法**：类的函数，格式：`[可见性] 方法名(参数): 返回类型`

**可见性符号**

- `+` 公开（public）
- `-` 私有（private）
- `#` 受保护（protected）

------

#### 2. 常见关系

| 关系类型 | 符号               | 说明                    |
| -------- | ------------------ | ----------------------- |
| 继承     | 空心箭头 ——▷       | 子类继承父类            |
| 实现     | 虚线空心箭头 —— -▷ | 类实现接口              |
| 关联     | 实线               | 类之间有引用关系        |
| 聚合     | 空心菱形——◇——      | “整体-部分”，弱拥有关系 |
| 组合     | 实心菱形——◆——      | “整体-部分”，强拥有关系 |
| 依赖     | 虚线箭头 —— -▷     | 临时使用关系            |

------

3. Python 类与 UML 对应

| UML元素  | Python示例                    |
| -------- | ----------------------------- |
| 类名     | `class MyClass:`              |
| 属性     | `self.attr`                   |
| 方法     | `def method(self, arg):`      |
| 继承     | `class SubClass(SuperClass):` |
| 实现接口 | 继承抽象基类（`abc.ABC`）     |
| 关联     | 类成员是其他类实例            |
| 聚合     | 成员变量引用其他类实例        |
| 组合     | 成员变量在生命周期紧密绑定    |

------

#### 4. UML 简单示例

```
+------------------+
|     Person       |
+------------------+
| - name: str      |
| - age: int       |
+------------------+
| + greet()        |
+------------------+

+------------------+       +------------------+
|     Student      |◄──────|     Person       |
+------------------+       +------------------+
| - student_id: int|       | - name: str      |
| + study()        |       | - age: int       |
+------------------+       +------------------+
```



------

## 4、`set` vs `frozenset` 对比

| 特性                | `set`                                   | `frozenset`                                |
| ------------------- | --------------------------------------- | ------------------------------------------ |
| **可变性**          | 可变，可以添加、删除元素                | 不可变，创建后元素不可修改                 |
| **支持的方法**      | `add()`, `remove()`, `pop()` 等可变操作 | 无 `add()`、`remove()`，只有不变的集合操作 |
| **哈希性**          | 不可哈希，不能作为字典键或集合元素      | 可哈希，可以作为字典键或集合的元素         |
| **用作键/集合元素** | 不可以                                  | 可以                                       |
| **常用场景**        | 需要频繁修改集合元素                    | 需要固定不变的集合，比如字典键、集合的元素 |

------

### 详细解释

#### 1. `set`（集合）

- **创建**：

  ```python
  s = set([1, 2, 3])
  ```

- **操作**：

  ```python
  s.add(4)
  s.remove(2)
  ```

- **不能作为字典的键或其他集合元素**：

  ```python
  d = {}
  d[s] = "value"  # 会报错：TypeError: unhashable type: 'set'
  ```

------

### 2. `frozenset`（不可变集合）

- **创建**：

  ```python
  fs = frozenset([1, 2, 3])
  ```

- **不可修改**，以下操作都会报错：

  ```python
  fs.add(4)  # AttributeError
  ```

- **可以作为字典键**：

  ```python
  d = {}
  d[fs] = "value"  # 允许
  print(d[fs])     # 输出 'value'
  ```

- **支持集合操作**（返回新的 `frozenset`）：

  ```python
  fs1 = frozenset([1, 2, 3])
  fs2 = frozenset([2, 3, 4])
  print(fs1 & fs2)  # frozenset({2, 3}) 交集
  print(fs1 | fs2)  # frozenset({1, 2, 3, 4}) 并集
  ```

------

### 总结

- **想要可变集合，选 `set`**
- **想要不可变、可哈希的集合，选 `frozenset`**
- 如果你想用集合当字典的键，一定用 `frozenset

## 5、解包运算符 *

  Python 不允许 仅用一个 *变量 进行赋值。*k 必须放在一个“结构”中，如元组或列表里，才能使用：因为 Python 需要明确这是一种序列解包形式（就像 a, b = ... 那样），哪怕只有一个变量，也必须有逗号！解包的对象可以是任何可迭代对象

## 6、Ellipsis对象

------

### 1. Ellipsis 对象 `...`

- `Ellipsis` 是 Python 的一个内置特殊常量，写作三个点 `...`。
- 它本身是一个单独的对象，类型是 `EllipsisType`。
- 你可以直接使用 `...`，它可以作为占位符或特殊标记。

```python
print(type(...))  # <class 'ellipsis'>
```

------

### 2. 在切片中用作多维切片占位符

最常见的用途是在多维数组或类似结构的切片操作中，用来表示“跨越所有剩余维度”。

比如：

```python
import numpy as np

arr = np.arange(27).reshape(3, 3, 3)

print(arr[1, ...])  # 等价于 arr[1, :, :]
print(arr[..., 2])  # 等价于 arr[:, :, 2]
```

这里 `...` 表示“剩余所有维度”，让你不用写所有冒号。

------

### 3. 作为代码占位符

有时候你写代码时，暂时还没写实现，可以用 `...` 占位，表示“这里以后会写代码”：

```python
def my_func():
    ...
```

相当于 `pass`，不过更简洁。

------

### 总结

- `Ellipsis` 是 Python 的内置常量，写作 `...`。
- 在多维索引切片里用来表示所有剩余维度。
- 也常用作代码占位符。

------

## 7、array类型

Python 提供了一个内建模块 `array`，用来存储**基本数值类型**的**高效数组**。

与 `list` 相比：

- `array` 只能存一种数据类型（数值类型为主）
- 更节省内存、更快，适合大规模数值处理
- 不如 `list` 灵活，但性能更好

------

### ✅ 1. 基本使用

```python
from array import array

# 创建一个整型数组
a = array('i', [1, 2, 3, 4])
print(a)        # array('i', [1, 2, 3, 4])
print(a[1])     # 2
```

------

### ✅ 2. 类型码说明（type code）

| 类型码 | 类型         | C 类型         | 占用字节 | 示例                   |
| ------ | ------------ | -------------- | -------- | ---------------------- |
| `'b'`  | 有符号字符   | signed char    | 1        | `array('b', [1, -2])`  |
| `'B'`  | 无符号字符   | unsigned char  | 1        | `array('B', [0, 255])` |
| `'h'`  | 有符号短整型 | signed short   | 2        |                        |
| `'H'`  | 无符号短整型 | unsigned short | 2        |                        |
| `'i'`  | 有符号整型   | signed int     | 4        |                        |
| `'I'`  | 无符号整型   | unsigned int   | 4        |                        |
| `'f'`  | 单精度浮点   | float          | 4        |                        |
| `'d'`  | 双精度浮点   | double         | 8        |                        |

------

### ✅ 3. 常用方法

```python
a = array('i', [1, 2, 3])

a.append(4)           # 添加一个元素
a.extend([5, 6])      # 添加多个元素
a.insert(1, 10)       # 指定位置插入
a.pop()               # 删除最后一个元素
a.remove(10)          # 删除指定值
a.index(2)            # 查找元素位置
a.reverse()           # 原地反转
```

------

### ✅ 4. 转换为/从 list

```python
lst = list(a)                    # array → list
a2 = array('i', lst)             # list → array
```

------

### ✅ 5. 二进制读写（可用于 `.bin` 文件）

```python
# 写入文件
with open('data.bin', 'wb') as f:
    a.tofile(f)

# 读取文件
with open('data.bin', 'rb') as f:
    b = array('i')
    b.fromfile(f, 6)  # 读取 6 个元素
```

------

### 🧠 array 和 list 对比

| 特性     | `array`        | `list`   |
| -------- | -------------- | -------- |
| 类型限制 | 同一数值类型   | 任意对象 |
| 内存效率 | 高（连续存储） | 低       |
| 访问速度 | 快             | 略慢     |
| 功能丰富 | 一般           | 非常丰富 |

------

### 🔧 小技巧：快速创建

```python
from array import array

zeros = array('f', [0.0] * 100)  # 100 个 float 0.0
```

------

### ✅ 总结

- 模块：`from array import array`
- 使用场景：处理大量**同类型数值数据**
- 优点：比 `list` 更快、更省内存
- 局限：不能混合类型、不支持字符串、对象等

------

## 8、deque类型

------

### 什么是 `deque`？

`deque`（双端队列，double-ended queue）是 Python 标准库 `collections` 模块中的一个**线程安全且高效的双端队列容器**，它支持在队列的两端快速地添加和删除元素。

------

### 特点

- **高效的两端插入和删除**，时间复杂度均为 O(1)
- 支持从左端和右端操作元素（append/appendleft，pop/popleft）
- 可以指定最大长度，超过时会自动丢弃最老的数据（循环队列效果）
- 线程安全，适合多线程环境使用

------

### 导入方式

```python
from collections import deque
```

------

### 创建 `deque`

```python
# 创建空deque
d = deque()

# 用可迭代对象初始化
d = deque([1, 2, 3, 4])
```

------

### 主要操作

| 方法                   | 说明                                 | 示例                   |
| ---------------------- | ------------------------------------ | ---------------------- |
| `append(x)`            | 在右侧添加元素                       | `d.append(5)`          |
| `appendleft(x)`        | 在左侧添加元素                       | `d.appendleft(0)`      |
| `pop()`                | 移除并返回右侧元素                   | `d.pop()`              |
| `popleft()`            | 移除并返回左侧元素                   | `d.popleft()`          |
| `extend(iterable)`     | 右侧批量添加                         | `d.extend([6,7])`      |
| `extendleft(iterable)` | 左侧批量添加（顺序反转）             | `d.extendleft([0,-1])` |
| `rotate(n=1)`          | 旋转队列，正数向右旋转，负数向左旋转 | `d.rotate(2)`          |
| `clear()`              | 清空所有元素                         | `d.clear()`            |

------

### 最大长度限制

```python
d = deque(maxlen=3)
d.extend([1, 2, 3])
print(d)  # deque([1, 2, 3], maxlen=3)

d.append(4)
print(d)  # deque([2, 3, 4], maxlen=3)  自动丢弃左侧最旧元素1
```

------

### 示例

```python
from collections import deque

d = deque([10, 20, 30])
d.append(40)             # 右端添加
d.appendleft(5)          # 左端添加

print(d)                 # deque([5, 10, 20, 30, 40])

print(d.pop())           # 40，右端弹出
print(d.popleft())       # 5，左端弹出

d.rotate(1)              # 右旋转一位
print(d)                 # deque([10, 20, 30])

d.rotate(-2)             # 左旋转两位
print(d)                 # deque([30, 10, 20])
```

------

### 适用场景

- 实现**队列**和**栈**结构（先进先出，后进先出）
- 滑动窗口算法（可用 `maxlen` 限制大小）
- 高效的线程安全数据缓冲区
- 需要频繁两端插入和删除的场景

------

### 总结

| 优点           | 说明                                    |
| -------------- | --------------------------------------- |
| 快速的两端操作 | append/pop/appendleft/popleft 均为 O(1) |
| 灵活的容量控制 | `maxlen` 参数支持自动丢弃最旧元素       |
| 线程安全       | 适合多线程环境使用                      |
| 标准库直接支持 | 无需额外安装库                          |

------

## 9、区分深浅拷贝

------

**浅拷贝（Shallow Copy）：**

- 创建一个新的容器对象，但容器内的元素仍然是原对象中元素的引用。
- 即，浅拷贝只复制了最外层的对象，内部的可变对象不会被复制。
- 修改拷贝对象中嵌套的可变元素，会影响原对象中的对应元素。

**深拷贝（Deep Copy）：**

- 递归地复制对象及其包含的所有子对象，创建完全独立的对象副本。
- 新对象和原对象之间没有任何共享部分，修改其中任意部分互不影响。
- 适用于需要完全独立副本的复杂对象。

---

## 10、@dataclass修饰的类变量

------

🔍 举个例子：

```python
from dataclasses import dataclass, field

@dataclass
class Animal:
    a: int
    d = []  # ← 类变量，所有实例共享同一个列表

# 创建两个实例
ani1 = Animal(1)
ani2 = Animal(2)

ani1.d.append("狗")
ani2.d.append("猫")

print(ani1.d)  # ['狗', '猫']
print(ani2.d)  # ['狗', '猫']
print(Animal.d)  # ['狗', '猫']
```

------

🔥 解释：

- `d = []` 没有类型注解，Python 把它当作**类变量**；
- 类变量在内存中**只有一份**，**所有实例共享这份值**；
- 所以 `ani1.d` 和 `ani2.d` 实际访问的是同一个列表对象。

------

💥 危险：共享可变类变量 = 潜在 bug

如果你把 `d` 设置为列表、字典、集合等**可变类型**：

```python
d = []   # ⚠️ 所有实例共用
```

那么任何一个实例修改它，**所有其他实例都会受到影响！**

------

✅ 正确做法：使用字段 + default_factory

如果你想让每个实例都有自己的列表，应该这样写：

```python
from dataclasses import dataclass, field

@dataclass
class Animal:
    a: int
    d: list = field(default_factory=list)  # ✔️ 每个实例独立的列表
```

### 测试：

```python
ani1 = Animal(1)
ani2 = Animal(2)

ani1.d.append("狮子")
ani2.d.append("老虎")

print(ani1.d)  # ['狮子']
print(ani2.d)  # ['老虎']
```

------

🎯 总结口诀：

| 写法                                    | 是否共享                 |
| --------------------------------------- | ------------------------ |
| `d = []`（类变量）                      | ✅ 会共享                 |
| `d: list = field(default_factory=list)` | ❌ 不共享（每个实例独立） |



如果 `d` 是一个 `int`，作为类变量，会不会被所有实例共享？

------

✅ 类变量，无论类型，都会被所有实例共享！

不管是列表、整数、字符串、还是其他对象，**只要是类变量（没有类型注解，直接赋值写在类里），它都被所有实例共享**。

------

举个简单的 `int` 示例：

```python
class Animal:
    d = 100  # 类变量，所有实例共享

ani1 = Animal()
ani2 = Animal()

print(ani1.d)  # 100
print(ani2.d)  # 100

# 改变类变量
Animal.d = 200

print(ani1.d)  # 200
print(ani2.d)  # 200

# 给实例赋值，会“屏蔽”类变量
ani1.d = 300

print(ani1.d)  # 300 （实例变量，覆盖了类变量）
print(ani2.d)  # 200 （仍是类变量）
print(Animal.d)  # 200
```

------

🧠 解释：

- `d = 100` 是类变量，所有实例默认访问的是这个类变量；
- 如果你通过实例赋值，比如 `ani1.d = 300`，这时 `ani1` 会生成自己的实例变量，**覆盖类变量**，但是其他实例不受影响；
- 但是只要实例没有重写这个属性，访问到的都是类变量的同一个值。

------

❌ 总结：

- **类变量（无类型注解、直接赋值）都是所有实例共享的，无论类型是 `int`、`str` 还是 `list`。**
- **可变类型（如列表）改动会直接影响所有实例。**
- **不可变类型（如整数）赋值时如果修改，会导致实例生成独立属性，不影响其他实例。**

------

如果你希望每个实例都有自己的整数值，应当使用带类型注解的字段：

```python
from dataclasses import dataclass

@dataclass
class Animal:
    d: int = 100  # 每个实例都会有自己的 d
```

这样每个实例的 `d` 是独立的，互不干扰。

------

假设你有这样的类和实例：

```python
class Animal:
    d = 0  # 类变量

dog = Animal()
cat = Animal()
```

------

访问和赋值 `d` 的规则：

1. 访问属性时（`dog.d`）

- Python 先查找实例对象 `dog` 自身的属性字典 `dog.__dict__`；
- 如果找不到，再去类 `Animal` 查找；
- 如果类也没有，再去父类查找，依此类推。

2. 给实例赋值时（`dog.d = 3`）

- 赋值**不会修改类变量**，而是在实例 `dog` 自己的字典里创建一个新的属性 `d`；
- 这样 `dog` 的 `d` 成为**实例变量**，覆盖了类变量。

------

基于你代码的示例：

```python
dog.d = 3       # 给dog实例赋值d，创建实例变量dog.d=3
dog.d           # 访问dog的实例变量，结果是3

cat.d           # cat没有自己的d，所以访问类变量，结果是0（或之后的值）

Animal.d = 78   # 改变类变量d=78

cat.d           # cat没有实例变量d，访问类变量，结果是78
dog.d           # dog有实例变量d=3，访问的是实例变量，不是类变量
```

------

所以总结：

| 操作          | dog.d 是啥？  | cat.d 是啥？ | 解释                          |
| ------------- | ------------- | ------------ | ----------------------------- |
| 初始化        | 0（类变量）   | 0（类变量）  | dog、cat共享类变量            |
| `dog.d = 3`   | 3（实例变量） | 0（类变量）  | dog创建实例变量，覆盖类变量   |
| `Animal.d=78` | 3（实例变量） | 78（类变量） | 类变量变了，但dog实例变量不变 |

------

⚠️ 重要提示：

- **实例赋值会隐藏类变量，不会修改类变量本身**；
- 只有直接修改类变量 `Animal.d = ...`，才会影响没有实例变量覆盖的实例。

------

如果你想让 `dog.d`、`cat.d` 都同步变化（即都指向同一个变量），**不要给实例赋值**，只操作类变量。

------

额外补充：

如果你用 `del dog.d` 删除实例变量 `d`，`dog.d` 会重新回到访问类变量：

```python
del dog.d
print(dog.d)  # 现在输出是 78（类变量值）
```

------

## 11、深浅拷贝再次区分

### 示例1

![image-20250613195721088](C:\Users\osquer\Desktop\typora图片\image-20250613195721088.png)

> **浅拷贝会创建一个新的外层容器对象，但其中的元素仍然是原对象的引用（不会递归拷贝）。**

浅拷贝只复制最外层容器，内部元素仍是原对象的引用。修改可变对象会影响原数据；修改不可变对象（如字符串）则会创建新对象，从而不会影响原数据。数值类型（int、float 等）是不可变对象，浅拷贝时不会创建新的数值对象，而是继续引用原对象。**字符串是不可变对象**，浅拷贝时不会创建新的字符串对象，而是**共享原引用**。当你“修改”字符串时，本质上是创建了一个新的字符串对象。“修改字符串”并不是真正的修改，而是**重绑定变量到新对象**。

只有在 **列表中嵌套的是可变对象（如：列表、字典、自定义对象）** 且你确实要**递归隔离所有层级**时，才使用 `deepcopy()`。大部分业务场景中，一层浅拷贝就能避免绝大多数问题了。

------

### 示例2

先明确两件事：

1. **列表的“外层容器”和“内部元素”是两种不同的东西**。
2. `list()` 会新建一个新的**外层列表容器**，但里面元素仍然是原来那个对象的引用。

------

你担心的问题是：

> 修改“内部元素”会改变“外层列表”的内容，那这怎么叫隔离？

------

**核心区别：修改列表的结构 vs 修改列表内部元素**

1. 修改列表的结构（添加、删除元素）

这类操作是针对**外层列表容器**本身。

- **外层容器是独立的**，`list()` 已经创建了一个新的外层列表对象，结构上的改动互不影响。

举个例子：

```python
group = ['Alice', 'Bob']
bus1 = Bus(group)  # Bus内部做了list(group)，浅拷贝了外层容器

bus1.drop('Alice')  # 修改bus1.passengers列表结构：删除元素

print(group)  # ['Alice', 'Bob']，group保持不变
```

**说明**：你改的是 `bus1.passengers` 的列表结构，`group` 还是原来的列表，没被改。

------

2. 修改列表里的元素（元素本身是可变的）

这类操作是针对**列表里的元素对象**。

- 元素对象是共享的引用，如果它是可变的（如另一个列表），修改它会影响所有引用它的地方。

例子：

```python
a = [[1, 2], [3, 4]]
b = list(a)  # 浅拷贝

b[0].append(99)  # 修改第一个元素（内部列表）

print(a)  # [[1, 2, 99], [3, 4]]
print(b)  # [[1, 2, 99], [3, 4]]
```

这里，`a` 和 `b` 是不同的列表，但它们的第一个元素是同一个列表对象，所以改了元素，两个都看到了变化。

------

#### 总结：

| 操作类型         | 是否隔离 | 说明                                             |
| ---------------- | -------- | ------------------------------------------------ |
| 修改外层列表结构 | 是       | `list()`生成了新的列表容器，结构改动不影响原列表 |
| 修改内层元素内容 | 否       | 元素是共享引用，改了会影响所有引用它的地方       |

------

#### 你的困惑点的本质

- **`list()` 隔离的是“列表容器”，不是容器内的元素**
- **如果想完全隔离元素，也就是“元素内容改动不影响”，需要用 `deepcopy()`**

------

#### 你可以这样想象：

- `group` 是一本书的“书架”（列表容器）
- 书架上的每本书（列表元素）都是同一本书（引用相同对象）

用 `list()` 是换了一个新的书架，放上原来的书；
 换了书架，移动书的位置互不影响；
 但**如果你翻开某本书，修改里面的内容，所有书架上的这本书都会改变**。

### 示例3

你的疑问非常关键，也是理解 Python 引用机制的核心。让我帮你**理清“外层容器修改”和“内层元素修改”的本质区别**，用更严谨的定义和具体例子帮你分清这两者：

------

#### 1. 什么是“外层容器修改”？

**外层容器修改**指的是对列表本身结构的操作，换句话说，就是：

- **添加、删除、替换列表里的元素**
- 改变列表包含了哪些元素（元素的“身份”或“数量”）

举例：

```python
lst = [1, 2, 3]

# 外层容器修改举例：
lst.append(4)      # 新增元素
lst.remove(2)      # 删除元素
lst[0] = 100       # 替换元素
```

这些操作改变的是列表本身“谁在列表里”这个事实。

------

#### 2. 什么是“内层元素修改”？

**内层元素修改**指的是修改列表里某个元素对象本身的状态或内容，**不改变列表中元素的身份或数量**，而是改动元素的内部数据。

这只有当元素本身是可变对象时才可能，比如列表、字典、自定义对象。

举例：

```python
lst = [[1, 2], [3, 4]]

# 内层元素修改举例：
lst[0].append(99)    # 修改第一个元素这个子列表的内容
lst[1][0] = 100      # 修改第二个元素这个子列表里的第一个元素
```

这两行都没有增删替换列表里的元素，只是改了元素“里面”的内容。

------

#### 3. 为什么两者不一样？

- **外层容器修改**会影响列表的结构，影响“列表里装了什么对象”，这时列表对象本身变化了。
- **内层元素修改**不改变列表里“放了哪些对象”，只是对象本身内容变了。

------

#### 4. 对你的疑问的回答

> “我的认知停留在修改外层不就是修改内层元素来体现的吗？”

严格说，不是这样的。**修改外层和修改内层是两个层级的概念，不是一个东西的两种表现。**

- 修改外层：换掉或增删了“元素这个引用”，列表结构变化了。
- 修改内层：元素引用没变，元素自身的内容变了。

------

#### 5. 举个综合例子

```python
lst = [[1, 2], [3, 4]]

# 外层容器修改：替换第一个元素（引用改变）
lst[0] = ['a', 'b']  

# 内层元素修改：修改第一个元素列表的内容（引用不变）
lst[1].append(99)
```

- 第一行是**外层容器修改**，`lst[0]` 从 `[1,2]` 变成 `['a','b']`，列表结构变了。
- 第二行是**内层元素修改**，`lst[1]` 仍然是原来的列表，只是内容变了。

------

#### 总结

| 概念         | 修改的对象               | 是否改变列表结构 | 示例操作                           |
| ------------ | ------------------------ | ---------------- | ---------------------------------- |
| 外层容器修改 | 列表里的元素引用（身份） | 是               | append, remove, 替换元素           |
| 内层元素修改 | 元素本身的内容           | 否               | 修改元素对象内部（append、赋值等） |

------

### 示例4

![image-20250613213538344](C:\Users\osquer\Desktop\typora图片\image-20250613213538344.png)

## 12、属性

在 Python 中，**属性**是**对象中与其状态相关的数据**，也可以理解为**类或实例中绑定的数据成员**。属性可以是变量，也可以是方法（函数），本质上是“对象.名字”的绑定。

------

### ✅ 一句话定义：

> **属性是属于对象的数据或行为接口，可以通过“点号（`.`）语法”访问。**

------

#### 🔍 举个简单例子：

```python
class Person:
    def __init__(self, name, age):
        self.name = name     # 属性：name
        self.age = age       # 属性：age

p = Person("Alice", 25)

print(p.name)  # 输出：Alice
print(p.age)   # 输出：25
```

在这个例子中：

- `p` 是 `Person` 类的一个实例；
- `p.name` 和 `p.age` 就是这个对象的“属性”；
- 它们是通过 `self.name = name` 和 `self.age = age` 在 `__init__` 构造方法中定义的。

------

#### 🧠 属性可以是几种类型：

| 类型                  | 示例                       | 说明                       |
| --------------------- | -------------------------- | -------------------------- |
| 实例属性              | `self.name = "Tom"`        | 每个实例独有的属性         |
| 类属性                | `Person.species = "human"` | 所有实例共享的属性         |
| 方法属性              | `def speak(self): ...`     | 本质上是函数，绑定到对象上 |
| 只读属性（@property） | `@property def age(self):` | 用于封装访问控制和逻辑     |

------

#### 🧪 动态添加属性（Python 的灵活性）

```python
p.gender = "female"  # 动态添加一个新属性
print(p.gender)      # 输出：female
```

------

#### ✅ 小总结：

> 在面向对象编程中，**属性描述了对象的“状态”**，和方法（描述“行为”）一起构成了对象的完整模型。

------

 
