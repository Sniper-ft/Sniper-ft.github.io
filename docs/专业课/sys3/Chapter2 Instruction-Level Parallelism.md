# Chapter 2: Instruction-Level Parallelism
## 1 Dependences
### 1.1 Data Dependences

!!! example
    ```
    FLD  F0, 0(R1)
    FADD.D  F4, F0, F2
    ```

### 1.2 Name Dependences
名是指指令所访问的寄存器或存储器单元的名称。如果两条指令使用相同的名，但是它们之间并没有数据流动，则称这两条指令存在名相关。

- F：浮点数运算
- ADD SUB 为一条路，MUL DIV 为另一条路
- anti dependence：指令j写的名与指令i读的名相同,则称指令i和j发生了反相关。 反相关指令之间的执行顺序是必须严格遵守的，以保证i读的值是正确的
- output dependence：

!!! example 
    1. anti dependence:
    ```
    FDIV.D     F2, F6, F4
    FADD.D     F6, F0, F12
    FSUB.D     F8, F6, F14
    ```

    Change F6 as S

    2. Output dependence:
    ```
    FDIV.D     F2, F6, F4
    FADD.D     F6, F0, F12
    FSUB.D     F2, F6, F14
    ```

    Change F2 as S

### 1.2 Control Dependences
s1 只在 s2 成立/不成立时执行（分支跳转）
## 2 Hazards
### 2.1 Structure Hazard
### 2.2 Data Hazard
- RAW：Read After Write
- WAR：Write After Read
- WAW: Write After Write
### 2.3 Control Hazard
#### Branch History Table (BHT)
- 1-bit predictor: Inner loop branches mispredicted twice!
- 2-bit predictor:  Inner loop branches mispredicted only once!
## 2 Dynamic Scheduling
### 2.1 stages
- Issue (IS): decode, check for structural hazards (in-order issue)
- Read Operands (RO): Wait until no data hazards, then read operands (out of order execution)

||IS|EX|WB|
|---|---|---|---|
|FLD F6, 34(R2)|1|3|4|
|FLD F2, 45(R3)|2|4|5|
|FMUL.D F0, F2, F4|3|6~15|16|
|FSUB.D F8, F6, F2|4|6~7|8|
|FDIV.D F10, F0, F6|5|17~56|57|
|FADD.D F6, F8, F2|6|9~10|11|


