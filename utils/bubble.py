import pickle
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

with open('userdict.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
# with open('userdictdirichlet.pkl', 'rb') as f:
#     loaded_dict = pickle.load(f)

print(loaded_dict)

# data = {0: [790, 718, 769, 861, 768, 655, 708, 969, 861, 946], 1: [765, 944, 752, 846, 887, 774, 393, 904, 36, 792],
#               2: [858, 771, 801, 776, 902, 0, 0, 750, 0, 719], 3: [822, 838, 816, 836, 720, 0, 0, 822, 0, 842],
#               4: [723, 724, 890, 745, 724, 0, 0, 809, 0, 922], 5: [798, 885, 773, 440, 909, 0, 0, 815, 0, 776],
#               6: [862, 622, 953, 0, 730, 0, 0, 754, 0, 477], 7: [874, 0, 813, 0, 812, 0, 0, 755, 0, 0],
#               8: [798, 0, 831, 0, 789, 0, 0, 935, 0, 0], 9: [865, 0, 757, 0, 914, 0, 0, 641, 0, 0]}

# pddata = pd.DataFrame.from_dict(loaded_dict, columns=['class0', 'class1', 'class2', 'class3', 'class4', 'class5', 'class6',
#                                                      'class7', 'class8', 'class9'])

pddata = pd.DataFrame.from_dict(loaded_dict)
print(pddata)
sns.scatterplot(pddata)
plt.show()
