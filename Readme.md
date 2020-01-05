# PyTorch Implementation of  Learning without Forgetting for multi-class



A PyTorch Implementation of [Learning without Forgetting](https://arxiv.org/pdf/1606.09282.pdf).

The LwF Implement for multi-class

About LwF.MC,you can read [iCaRL: Incremental Classifier and Representation Learning](https://arxiv.org/abs/1611.07725)

## requirement

python3.6

Pytorch1.3.0 linux

PIL



## run

```shell
python -u main.py
```





# Result

Resnet18+CIFAR100



| incremental step    | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9|average|
| ------------------- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| test accuracy | 80.06 |59.15|53.334|47.778|42.196|38.82|36.592|32.996|30.442|27.97|40.591|
									
