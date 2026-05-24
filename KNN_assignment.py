import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

#data read
data=pd.read_csv(r"C:\flutter_projects\INTERNSHIP\class3\assignment\KNN_Dataset.csv")
x=data[["Temperature"]]
y=data[["Fuel_Consumption"]]

#training
model=KNeighborsRegressor(n_neighbors=3) #k=3
model.fit(x,y)

#predicted output
prediction=model.predict([[58]])
print("FUEL CONSUMPTION AT 58 DEGREE CELSIUS OPERATING TEMPERATURE IS:",prediction)