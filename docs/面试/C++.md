# C++
## 算法题
!!! tip inline end
    满脑子都是[这道题](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/?envType=study-plan-v2&envId=top-interview-150)，然而怎么还是dp dp dp?!

### 例题一
输入一个vector，表示每天的股票价格，卖出后第二天冻结，不能买入，输出最大利润。

??? success "answer"

    ```c

    class Solution {
    public:
        /**
        * @brief 计算带有冷冻期的股票交易的最大利润
        * @param prices 每日股票价格的向量
        * @return 最大利润
        */
        int maxProfit(const std::vector<int>& prices) {
            // 如果价格天数少于2天，不可能完成一次交易
            if (prices.size() < 2) {
                return 0;
            }

            // 初始化三种状态
            // hold: 持有股票时的最大利润
            // sold: 刚卖出股票时的最大利润 (进入冷冻期)
            // rest: 不持有股票且不在冷冻期时的最大利润

            // 对第0天的状态进行初始化
            int hold = -prices[0];
            int sold = 0;
            int rest = 0;

            // 从第1天开始遍历
            for (size_t i = 1; i < prices.size(); ++i) {
                // 记录下前一天的 sold 状态，因为计算 rest 状态时需要它
                int prev_sold = sold;

                // 更新今天的 sold 状态：昨天必须持有，今天卖出
                sold = hold + prices[i];
                
                // 更新今天的 hold 状态：昨天持有 vs 昨天休息今天买入
                hold = std::max(hold, rest - prices[i]);
                
                // 更新今天的 rest 状态：昨天休息 vs 昨天卖了今天强制休息
                rest = std::max(rest, prev_sold);
            }

            // 最终的最大利润一定是在不持有股票的状态下产生的
            // （要么是刚卖出，要么是处于休息状态）
            return std::max(sold, rest);
        }
    };

    ```


!!! tip inline end
    使用 bitset 状态压缩
### 例题二

在一个$𝑛×𝑚$的整数矩阵$A$中，选一个子矩形（高、宽都 ≥ 2），其“强度”定义为四个角上的数的最小值，问能取得的最大强度。

??? success "answer"
    ```c
    #include <bits/stdc++.h>
    using namespace std;

    int max_strength_offline(vector<vector<int>>& A) {
        int n = A.size(), m = A[0].size();

        struct Cell {
            int val, r, c;
        };
        vector<Cell> cells;
        cells.reserve(n * m);

        for (int r = 0; r < n; r++) {
            for (int c = 0; c < m; c++) {
                cells.push_back({A[r][c], r, c});
            }
        }

        // 按值从大到小排序
        sort(cells.begin(), cells.end(),
            [](const Cell& a, const Cell& b) { return a.val > b.val; });

        // 每行一个 bitset
        vector<bitset<2005>> rowBits(n);  // 假设 m ≤ 2000，可调

        for (auto& cell : cells) {
            int r = cell.r, c = cell.c;
            rowBits[r].set(c);

            // 与所有其他行比较
            for (int r2 = 0; r2 < n; r2++) {
                if (r2 == r) continue;
                bitset<2005> common = rowBits[r] & rowBits[r2];
                if (common.count() >= 2) {
                    return cell.val; // 找到答案
                }
            }
        }
        return -1; // 如果不存在
    }

    ```

## 八股
### 例题一
如何判断 float 类型的数是否为0。
```c
bool isZero(double value) {
    return std::abs(value) < std::numeric_limits<double>::epsilon();
}
```