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

# 滑动窗口
## [LC3. 无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/description/?envType=problem-list-v2&envId=2cktkvj&)
**给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。**

题解：
```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        cnt = defaultdict(int)
        ans = 0
        left = 0
        for i, c in enumerate(s):
            cnt[c]+=1
            while cnt[c]>1:
                cnt[s[left]]-=1
                if cnt[s[left]]==0: cnt.pop(s[left])
                left+=1
            ans = max(ans, i-left+1)
        return ans  
```

## [LC438. 找到字符串中所有字母异位词](https://leetcode.cn/problems/find-all-anagrams-in-a-string/description/?envType=problem-list-v2&envId=2cktkvj&)
**给定两个字符串 s 和 p，找到 s 中所有 p 的 异位词 的子串，返回这些子串的起始索引。不考虑答案输出的顺序。**

题解：
```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        target = Counter(p)
        cnt = Counter()
        left = 0
        ans = []
        for i, c in enumerate(s):
            cnt[c]+=1
            while cnt>target:
                cnt[s[left]]-=1
                if cnt[s[left]]==0: cnt.pop(s[left])
                left+=1
            if cnt==target: ans.append(left)
        return ans
```

# 双指针
## [LC15.三数之和](https://leetcode.cn/problems/3sum/description/?envType=problem-list-v2&envId=2cktkvj&)
**给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足 i != j、i != k 且 j != k ，同时还满足 nums[i] + nums[j] + nums[k] == 0 。请你返回所有和为 0 且不重复的三元组。**

题解：
```python

class Solution:
    def threeSum(self, nums: list[int]) -> list[list[int]]:
        nums.sort()
        n = len(nums)
        ans = []
        for i in range(n-2):
            x = nums[i]
            if i>0 and x==nums[i-1]:
                continue
            if x+nums[i+1]+nums[i+2]>0:
                break
            if x+nums[-2]+nums[-1]<0:
                continue
            left = i+1
            right = n-1
            while left<right:
                s = nums[i]+nums[left]+nums[right]
                if s<0:
                    left+=1
                elif s>0:
                    right-=1
                else:
                    ans.append([x, nums[left], nums[right]])
                    left+=1
                    while left<right and nums[left-1]==nums[left]:
                        left+=1
                    right-=1
                    while left<right and nums[right+1]==nums[right]:
                        right-=1
        return ans
```

## [LC42. 接雨水](https://leetcode.cn/problems/trapping-rain-water/description/?envType=problem-list-v2&envId=2cktkvj&)
**给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。**

题解：
```python
class Solution:
    # 对于每个柱子，找左边的最大值和右边的最大值，较小的最大值为这个柱子的液面的高度，减去柱子本身的高度即为面积
    def trap(self, height: List[int]) -> int:
        n = len(height)
        pre_max = [0]*n
        pre_max[0] = height[0]
        for i in range(1,n):
            pre_max[i] = max(height[i], pre_max[i-1])
        suf_max = [0]*n
        suf_max[-1] = height[-1]
        for i in range(n-2,-1,-1):
            suf_max[i] = max(height[i], suf_max[i+1])
        ans = 0
        for i,h in enumerate(height):
            ans+=min(suf_max[i], pre_max[i])-h
        return ans

    def trap(self, height: List[int]) -> int:
        ans = pre_max = suf_max = 0
        left, right = 0, len(height)-1
        while left<right:
            pre_max = max(pre_max, height[left])
            suf_max = max(suf_max, height[right])
            if pre_max<suf_max:
                ans+=pre_max-height[left]
                left+=1
            else:
                ans+=suf_max-height[right]
                right-=1
        return ans

    # 用单调栈，横着计算面积
    def trap(self, height: List[int]) -> int:
        ans = 0
        st = []
        for i,h in enumerate(height):
            while st and height[st[-1]]<=h:
                bottom_h = height[st.pop()]
                if not st:
                    break
                left = st[-1]
                dh = min(height[left], h)-bottom_h
                ans+=dh*(i-left-1)
            st.append(i)
        return ans     
```

# 动态规划
## [LC198. 打家劫舍](https://leetcode.cn/problems/house-robber/description/?envType=problem-list-v2&envId=2cktkvj&)
**你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。**
**给定一个整数数组 nums ，表示一个环形街道上沿街的房屋，每个房屋内都藏有一定的现金。你不能同时偷窃相邻的两间房屋，否则会触发报警。**

题解：
```python
class Solution:
    # 递推
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        if n==1: return nums[0]
        f = [0]*n
        f[0] = nums[0]
        f[1] = max(nums[1], nums[0])
        for i in range(2, n):
            f[i] = max(nums[i]+f[i-2], f[i-1])
        return f[-1]

    # 记忆化递归
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        if n==1: return nums[0]
        @cache
        def dfs(i):
            if i==0: return nums[0]
            if i==1: return max(nums[0], nums[1])
            return max(nums[i]+dfs(i-2), dfs(i-1))
        return dfs(n-1)
```

## [LC72. 编辑距离](https://leetcode.cn/problems/edit-distance/description/?envType=problem-list-v2&envId=2cktkvj&)
**给你两个单词 word1 和 word2， 请返回将 word1 转换成 word2 所使用的最少操作数 。**
**你可以对一个单词进行如下三种操作：**
**插入一个字符**
**删除一个字符**
**替换一个字符**

题解：
```python
class Solution:
    # 递推
    def minDistance(self, word1: str, word2: str) -> int:
        m,n = len(word1), len(word2)
        f = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1):
            f[i][0] = i
        for j in range(n+1):
            f[0][j] = j
        for i in range(m):
            for j in range(n):
                if word1[i]==word2[j]:
                    f[i+1][j+1] = f[i][j]
                else:
                    f[i+1][j+1] = min(f[i][j], f[i][j+1], f[i+1][j])+1
        return f[-1][-1]
    
    # 记忆化递归
    def minDistance(self, word1: str, word2: str) -> int:
        m,n = len(word1), len(word2)
        @cache
        def dfs(i,j):
            if i==0: return j
            if j==0: return i
            if word1[i-1]==word2[j-1]:
                return dfs(i-1, j-1)
            else:
                return min(dfs(i-1,j-1), dfs(i,j-1), dfs(i-1,j))+1
        return dfs(m,n)
```





        
            
