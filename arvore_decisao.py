import pandas as pd

d = pd.read_csv('risco.csv', sep=';')
print(len(d))
print(d)
print()

d=pd.get_dummies(d,columns=['HC','DI','GA','RND'])
print(d)

d['RC']=d.apply(lambda row: 0 if (row['Risco']) == 'alto'
                else 1 if (row['Risco']) == 'moderado'
                else 2, axis=1)
print()

print(d.head())
print()

#d['teste']=d.apply(lambda row:10 if (row['DI_baixa'])==1 and (row['RND_0 a 15'])==1 else 15,axis = 1)
#print(d.head())
#print()

d = d.sample(frac=1)
d_train = d
d_test = d
d_train_att = d_train.drop(['RC'],axis=1)
d_train_pass = d_train['RC']

from sklearn import tree
t = tree.DecisionTreeClassifier(criterion="entropy")
t = t.fit (d_train_att,d_train_pass)

tree.export_graphviz(t, out_file="risco.dot",label="all",impurity=False,proportion=True,
                     feature_names=list(d_train_att),class_names=['alto','moderado','baixo'],
                     filled=True,rounded=True)

t.predict([[0,1,0,0,1,0,1,0,0,1]])
