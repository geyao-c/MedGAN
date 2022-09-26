import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style='darkgrid')  # 图形主题

plt.rcParams['savefig.dpi'] = 600
plt.rcParams['figure.dpi'] = 600

df = pd.DataFrame()
# df['Iter'] = list(range(5000, 35000, 5000))
df['Iter'] = [5000, 10000, 15000, 20000, 25000, 30000]
print(df['Iter'])
adagan_accu = [51.19, 59.422, 60.40, 60.61, 61.83, 60.95]
wgan_accu = [49.59, 50.10, 52.75, 50.81, 51.49, 52.85]
elist = [1, 2, 3, 4, 5, 6]

df['Accuracy(%)'] = wgan_accu
ax = sns.lineplot(data=df, x='Iter', y='Accuracy(%)', label='WGAN', palette='b', marker='o')

df['Accuracy(%)'] = adagan_accu
# df['Accuracy'] = elist
print(df['Accuracy(%)'])
ax = sns.lineplot(data=df, x='Iter', y='Accuracy(%)', label='AdaGAN', palette='r', marker='o')


ax.axhline(50, ls='--', c='r', label="baseline")
# ax2.axhline(30, ls='--')
ax.set_ylim(47, 63)
plt.legend(loc="best")
plt.savefig('image.jpg', bbox_inches='tight', pad_inches=0.1)
plt.show()




