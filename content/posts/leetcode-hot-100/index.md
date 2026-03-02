---
date: '2025-11-14T13:55:39+08:00'
draft: false
title: 'Leetcode Hot 100'
---

# 链表

## [LC160.相交链表](https://leetcode.cn/problems/intersection-of-two-linked-lists/?envType=problem-list-v2&envId=2cktkvj&)
![160](img/160.png)

题解：
参考[灵茶山艾府](https://leetcode.cn/problems/intersection-of-two-linked-lists/solutions/2958778/tu-jie-yi-zhang-tu-miao-dong-xiang-jiao-m6tg1/?envType=problem-list-v2&envId=2cktkvj&)题解
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        p1 = headA
        p2 = headB
        while p1 is not p2:
            p1 = p1.next if p1 else headB
            p2 = p2.next if p2 else headA
        return p1
```

**核心思想是两条链表满足 $(x+z)+y = (y+z)+x$，z为公共部分的长度**

# 树

## [LC236.二叉树的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/description/?envType=problem-list-v2&envId=2cktkvj&)

![236](img/236.png)

题解：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root in (None, p, q):
            return root
        left = self.lowestCommonAncestor(root.left,p,q)
        right = self.lowestCommonAncestor(root.right,p,q)
        if left and right:
            return root
        return left or right
```
## [LC437.路径总和II](https://leetcode.cn/problems/path-sum-ii/description/?envType=problem-list-v2&envId=2cktkvj&)
**给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。**
题解：
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def pathSum(self, root, targetSum):
        ans = []
        path = []
        def dfs(node, target):
            if node is None:
                return
            target-=node.val
            path.append(node.val)
            if node.left is None and node.right is None and target==0:
                ans.append(path.copy())
            else:
                dfs(node.left, target)
                dfs(node.right, target)
            path.pop()
        dfs(root, targetSum)
        return ans
```

## [LC236.二叉树的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/description/?envType=problem-list-v2&envId=2cktkvj&)
**给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。**
题解：临界条件：当前节点是空/p/q

如果左右子树都找到了，返回当前节点，如果只有左子树有结果，返回递归左子树的结果，相反返回右子树的递归结果。
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        def dfs(node, p, q):
            if node is p or node is q or node is None:
                return node
            left = dfs(node.left, p, q)
            right = dfs(node.right, p, q)
            if left and right:
                return node
            return left or right
        return dfs(root, p, q)
```

## [LC124.二叉树中的最大路径和](https://leetcode.cn/problems/binary-tree-maximum-path-sum/description/?envType=problem-list-v2&envId=2cktkvj&)
**给定一个非空二叉树，返回其最大路径和。**
题解：
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        ans = -inf
        def dfs(node):
            nonlcoal ans
            if node is None:
                return 0
            left = dfs(node.left)
            right = dfs(node.right)
            ans = max(ans, left+right+node.val)
            return max(max(left, right)+node.val, 0)
        dfs(root)
        return ans
```

## [LC199.二叉树的右视图](https://leetcode.cn/problems/binary-tree-right-side-view/description/?envType=problem-list-v2&envId=2cktkvj&)
**给定一个二叉树的 根节点 root，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。**

题解：层序遍历，每次取每一层的最后一个节点。
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
class Solution:
    # DFS先递归右子树
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        ans = []
        def dfs(node, depth):
            if node is None:
                return
            if len(ans)==depth:
                ans.append(node.val)
            dfs(node.right, depth+1)
            dfs(node.left, depth+1)
        dfs(root, 0)
        return ans
    # BFS层序遍历
    def rightSideViewBFS(self, root: Optional[TreeNode]) -> List[int]:
        ans = []
        if root is None: return ans
        q = [root]
        while q:
            ans.append(q[-1].val)
            tmp = q
            q = []
            for node in tmp:
                if node.left: q.append(node.left)
                if node.right: q.append(node.right)
        return ans
```

# 回溯

## [LC46.全排列](https://leetcode.cn/problems/permutations/description/?envType=problem-list-v2&envId=2cktkvj&)
**给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。**

题解：枚举位置，选填数字，把path看成N个位置，每个位置都有N种选法，但是要注意已经选过的数字不能再选。每一层递归负责填path[i]
```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        ans = []
        path = [0]*n
        visited = [False]*n
        def dfs(i):
            if i==n:
                ans.append(path.copy())
                return
            for j, v in enumerate(visited):
                if not v:
                    visited[j] = True
                    path[i] = nums[j]
                    dfs(i+1)
                    visited[j] = False
        dfs(0)
        return ans      
```

## [LC78.子集](https://leetcode.cn/problems/subsets/description/?envType=problem-list-v2&envId=2cktkvj&)
**给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。**

题解：选或不选
```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        path = []
        ans = []
        def dfs(i):
            if i==n:
                ans.append(path.copy())
                return
            
            # 不选nums[i]
            dfs(i+1)

            # 选nums[i]
            path.append(nums[i])
            dfs(i+1)
            path.pop()

        dfs(0)
        return ans
```

## [LC17.电话号码的字母组合](https://leetcode.cn/problems/letter-combinations-of-a-phone-number/description/?envType=problem-list-v2&envId=2cktkvj&)
**给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。**
题解：
```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        mps = {'2':'abc','3':'def','4':'ghi','5':'jkl','6':'mno','7':'pqrs','8':'tuv','9':'wxyz'}
        n = len(digits)
        path = ['']*n
        ans = []
        def dfs(i):
            if i==n:
                ans.append(''.join(path))
                return
            for c in mps[digits[i]]:
                path[i] = c
                dfs(i+1)
        dfs(0)
        return ans
            
```

## [LC22.括号生成](https://leetcode.cn/problems/generate-parentheses/description/?envType=problem-list-v2&envId=2cktkvj&)
**数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。**
题解：
```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        path = ['']*(2*n)
        ans = []
        def dfs(l, r):
            if r==n:
                ans.append(''.join(path))
                return
            if l<n:    
                path[l+r] = '('
                dfs(l+1, r)
            if r<l:
                path[l+r] = ')'
                dfs(l, r+1)
        dfs(0,0)
        return ans
```
# 图论

## [LC207.课程表](https://leetcode.cn/problems/course-schedule/description/?envType=problem-list-v2&envId=2cktkvj&)
**你这个学期必须选修 numCourses 门课程，记为 0 到 numCourses - 1 。**

**在选修某些课程之前需要一些先修课程。 先修课程按数组 prerequisites 给出，其中 prerequisites[i] = [ai, bi] ，表示如果要学习课程 ai 则 必须 先学习课程  bi 。**

**例如，先修课程对 [0, 1] 表示：想要学习课程 0 ，你需要先完成课程 1 。**
**请你判断是否可能完成所有课程的学习？如果可以，返回 true ；否则，返回 false。**

题解：
```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        g = [[] for _ in range(numCourses)]
        for a, b in prerequisites:
            g[b].append(a)
        colors = [0]*numCourses
        # 返回True表示有环
        # 1表示正在访问，2表示访问完成，0表示未访问
        def dfs(i):
            colors[i] = 1
            for j in g[i]:
                if colors[j]==1:
                    return True
                if colors[j]==0 and dfs(j):
                    return True
            colors[i] = 2
            return False
        for i, c in enumerate(colors):
            if c==0 and dfs(i):
                return False
        return True
```

## [LC200.岛屿数量](https://leetcode.cn/problems/number-of-islands/description/?envType=problem-list-v2&envId=2cktkvj&)
**给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。**

题解：**DFS**
```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        m,n = len(grid), len(grid[0])
        ans = 0
        def dfs(i,j):
            grid[i][j]='2'
            for x,y in (i+1,j),(i-1,j),(i,j+1),(i,j-1):
                if 0<=x<m and 0<=y<n and grid[x][y]=='1':
                    dfs(x,y)
        
        for i in range(m):
            for j in range(n):
                if grid[i][j]=='1':
                    ans+=1
                    dfs(i,j)
        return ans
```

## [LC994.腐烂的橘子](https://leetcode.cn/problems/rotting-oranges/description/?envType=problem-list-v2&envId=2cktkvj&)
**在给定的 m x n 网格 grid 中，每个单元格可以有以下三个值之一：**
**值 0 代表空单元格；**
**值 1 代表新鲜橘子；**
**值 2 代表腐烂的橘子。**
**每分钟，腐烂的橘子 周围 4 个方向上相邻 的新鲜橘子都会腐烂。**
**返回 直到单元格中没有新鲜橘子为止所必须经过的最小分钟数。如果不可能，返回 -1 。**

题解：**BFS**
```python
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        m,n = len(grid), len(grid[0])
        cnt = 0
        q = []
        ans = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j]==1:
                    cnt+=1
                if grid[i][j]==2:
                    q.append((i,j))
        while q and cnt:
            ans+=1
            tmp = q
            q = []
            for i,j in tmp:
                for x, y in (i-1,j),(i+1,j),(i,j-1),(i,j+1):
                    if 0<=x<m and 0<=y<n and grid[x][y]==1:
                        grid[x][y]=2
                        q.append((x,y))
                        cnt-=1
            
        return -1 if cnt else ans
```







        
            
