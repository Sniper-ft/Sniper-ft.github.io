# Hackthon2025
## step1 
设计一个评判最终答案

https://arxiv.org/pdf/2504.10337

感觉这个比较靠谱，公式是

![alt text](image.png)
- 修改 self_certainty 的计算方式（优化评估 trace 的指标）
- 设置一个截断阈值，如果“成绩”低于这个值就不算了，token级的降低开销
- 