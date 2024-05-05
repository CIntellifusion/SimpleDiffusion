# Unconditional Diffusion

## 1 Dataset

Celeb64 example: 





# Debug

在采样的过程中发现: noise预测的值会越来越大 

但是如果不更新x的话，只更新t的话，noise的值就是稳定的

所以推断问题是因为分布不断偏移



## normalization

之前的初始化用的是transform.normalize(0.5...)

现在改成线性的结果 让重建的时候是可逆的 而且是-1,1 ,中心为0

lambda: x: (x-0.5)/2

但是重建的时候分布的均值依然是0.5开始的 



## 学习率调整

应该先用不调整学习率方法进行训练

现在发现不减小学习率可以降低到更低的值

之前降低学习率的方法因为学习率降低过快 会收敛在比较高的值





