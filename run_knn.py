import pandas

from sklearn.neighbors import KNeighborsRegressor
import kfold_template

dataset = pandas.read_csv("abalone.csv")
dataset = dataset.sample(frac=1)

# gives us 8th column
target = dataset.iloc[:,8]
# ages is rings +1.5
target = target+1.5
target = target.values 

data = dataset.iloc[:,0:8]
pandas.get_dummies(data, columns = ["Sex"])
data = data.values 
# print(data)

#true is bc its continuous 

trials = []
for w in ["uniform","distance"]
	for k in range(1,50):
		machine = KNeighborsRegressor(n_neighbors=10, weights="uniform")
		return_values = kfold_template.run_kfold(machine, data, target, 4, True)
		average_return_value = sum(return_values)/len(return_values)
		# print(average_return_values)
		trials.append((average_return_value, k, w))

# 0 is the first element of trials.append (average_return_value)
trials.sort(key=lambda x: x[0])
print(trials[:5])

