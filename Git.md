# Git

## 1、概念知识

#### 1、head

`HEAD` 是 Git 中的**当前引用指针**。它指向你当前所在的 **commit** 或 **分支**。

#### 2、main

> **main = 默认分支**

它只是一个普通分支，但通常作为主开发支线使用。

#### 3、commit

commit 就是你把当前工作保存为一个版本，Git 会记录：文件内容、作者、时间、说明、以及它的父 commit。一个 commit 内部包含：

1. **树对象（tree）**：保存当前所有文件的快照
2. **父 commit**：上一条历史记录（除第一个 commit 没有父节点）
3. **作者信息**：你是谁
4. **提交信息（message）**：你为什么要提交
5. **时间戳**

#### 4、分支

分支 = 指向 commit 的指针，本质上是一个移动的指针，它指向某一个 commit（最新的 commit，也叫分支的 tip）。



## 2、指令

#### 1、`git commit`

 会把暂存区（index）里的内容打包成一个新的 commit，并写入当前分支的历史（在当前分支生成一个新的版本）。

#### 2、`git branch`

查看分支 / 创建分支（但不切换）

#### 3、`git checkout -b <name>`

创建并切换到新分支（最常用）

#### 4、`git checkout <name>` 

切换到已有分支

### 5、`git merge` 

把另一个分支的提交历史合并到当前分支。

### 6、`git checkout HEAD^   `

切换到当前 commit 的父 commit

#### 7`、git rebase`

把一个分支的修改“搬到”另一个分支的最前端，形成一条线性的历史。

### 8、`git checkout HEAD~3   `

切换到当前 commit 的前三个paren节点

### 9、` git branch -f main C6`

强制将main分支移动到 C6 commit

#### 10、 `git reset <commit>    ` 

把分支指针（如 main）移动到指定 commit，可以修改工作区和暂存区，可能会丢失历史。 **直接把时间轴倒退**，历史被抹掉.

#### 11、`git revert <commit>` 

 **新增一条“反向操作”**，历史完整，撤销动作可追溯

#### 12、`git cherry-pick <commit>` 

 从其他分支中挑选一个 commit，把它应用到当前分支。当然也可以挑选多个commit  :   git cherry-pick a c b

### 13、`git rebase -i HEAD~5`

表示从当前分支的上面第5个父节点开始进行交互式界面操作。

