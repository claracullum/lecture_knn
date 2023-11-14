import pandas

from sklearn.neighbors import KNeighborsClassifier
import kfold_template

dataset = pandas.read_csv("abalone.csv")
dataset = dataset.sample(frac=1)

dataset['Age Group'] = pandas.cut(dataset['Rings'], [0,9,11,13,100], label=[1,2,3,4])
dataset = dataset.drop(['Rings'], axis=1)

# gives us 8th column
target = dataset.iloc[:,8]
# ages is rings +1.5, so we dont need it in this program 
# target = target+1.5
target = target.values 

data = dataset.iloc[:,0:8]
pandas.get_dummies(data, columns = ["Sex"])
data = data.values 
# print(data)

#false is bc not continuous 

trials = []
for w in ["uniform","distance"]:
	for k in range(1,50):
		machine = KNeighborsClassifier(n_neighbors=10, weights="uniform")
		return_values = kfold_template.run_kfold(machine, data, target, 4, False)
		average_return_value = sum(return_values)/len(return_values)
		# print(average_return_values)
		trials.append((average_return_value, k, w))

# 0 is the first element of trials.append (average_return_value)
trials.sort(key=lambda x: x[0], reverse=True)
print(trials[:5])
#gives you 5 best parameters 
