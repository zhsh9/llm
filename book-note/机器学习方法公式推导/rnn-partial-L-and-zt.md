1. 假设使用交叉熵损失函数和softmax输出：

   $$L = -\sum_{t} y_t \log(p_t)$$
   $$p_t = \text{softmax}(z_t)$$

2. 应用链式法则：

   $$\frac{\partial L}{\partial z_t} = \frac{\partial L}{\partial p_t} \cdot \frac{\partial p_t}{\partial z_t}$$

3. 计算交叉熵对 pt 的偏导数：

   $$\frac{\partial L}{\partial p_t} = -\frac{y_t}{p_t}$$

4. Softmax 函数对 zt 的偏导数：

   $$\frac{\partial p_t}{\partial z_t} = p_t(1 - p_t)$$

5. 结合步骤3和4：

   $$\frac{\partial L}{\partial z_t} = -\frac{y_t}{p_t} \cdot p_t(1 - p_t) = -y_t(1 - p_t)$$

6. 展开并化简：

   $$\frac{\partial L}{\partial z_t} = -y_t + y_tp_t$$

7. 由于 yt 是one-hot编码，ytpt = pt：

   $$\frac{\partial L}{\partial z_t} = -y_t + p_t$$

8. 最终结果：

   $$\frac{\partial L}{\partial z_t} = p_t - y_t$$
