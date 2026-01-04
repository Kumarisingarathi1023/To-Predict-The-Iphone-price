import pandas
import matplotlib.pyplot as p 
from sklearn.linear_model import LinearRegression


d=pandas.read_csv("iphoneprices.csv")
#p.scatter(d['version'],d['price'])
#p.show()
m=LinearRegression()
m.fit(d[['version']],d[['price']])
print(m.predict([[20]]))