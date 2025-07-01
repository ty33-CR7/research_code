from ucimlrepo import fetch_ucirepo 
import pandas as pd
# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 

# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets 
z = wine_quality.data.original
print(z)
z=z.drop(["quality"],axis=1)
df=pd.concat([X,y],axis=1)
print(z.shape)
#(6497, 11)
z.to_csv("wine_quality.csv",index=False)