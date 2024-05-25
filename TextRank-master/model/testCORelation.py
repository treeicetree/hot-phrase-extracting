# -*- coding: UTF-8 -*-
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# a = np.random.rand(4,3)

a = np.array([[1,0.001,0.2,0.86],
              [0.001,1,0.0002,0.03],
              [0.2,0.0002,1,0.26],
              [0.86,0.03,0.26,1]])

#plt.rcParams['font.sans-serif'] = ['SimHei']
fig, ax = plt.subplots(figsize = (12,12))
#二维的数组的热力图，横轴和数轴的ticklabels要加上去的话，既可以通过将array转换成有column
#和index的DataFrame直接绘图生成，也可以后续再加上去。后面加上去的话，更灵活，包括可设置labels大小方向等。
sns.heatmap(pd.DataFrame(np.round(a,2), columns = ['Severe cases', 'Decision deployment', 'Epidemic situation','Masses of the people'], index = ['Severe cases', 'Decision deployment', 'Epidemic situation','Masses of the people']),
                annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True, square=True, cmap="Blues")
#sns.heatmap(np.round(a,2), annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True,
#            square=True, cmap="YlGnBu")
# ax.set_title('二维数组热力图', fontsize = 18)
ax.set_ylabel('KeyPhrase', fontsize = 18)
ax.set_xlabel('KeyPhrase', fontsize = 18) #横变成y轴，跟矩阵原始的布局情况是一样的
plt.savefig('./out.png')
plt.show()

