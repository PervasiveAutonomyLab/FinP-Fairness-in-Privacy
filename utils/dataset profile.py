import pickle
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

with open('userdict.pkl', 'rb') as f:
    userdict = pickle.load(f)

with open('userdictdirichlet.pkl', 'rb') as g:
    userdictdirichlet = pickle.load(g)

udict = pd.DataFrame.from_dict(userdict)
print('udict', udict)

udictdirichlet = pd.DataFrame.from_dict(userdictdirichlet)
print('udictdirichlet', udict)