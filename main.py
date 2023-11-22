import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier



import os
for dirname, _, filenames in os.walk('C:\\Users\\36709\\mobile'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


data = pd.read_csv('C:\\Users\\36709\\mobile\\mobile.csv')
data.isnull().sum
data.dropna()
data.duplicated()


convert = {
    'item_brand_name': {'apple': 0, 'samsung': 1, 'huawei': 2, 'mione': 3, 'blackberry': 4, 'g-plus': 5, 'lamborghini':6,'thuraya':7, 'cat': 8, 'sony': 9},
    'Number_Of_SIM': {'Single SIM & E-SIM': 0, 'Single Sim & E-Sim': 1, 'DUAL Physical SIM': 2, 'Sm-G975FcwhXSg': 3},
    'name': {'Apple iPhone 11 with FaceTime - 256GB': 0, 'Samsung Galaxy Note 10 Plus - 256GB': 1, 'Apple iPhone 11 Pro Max with FaceTime - 512GB': 2, 'Samsung Galaxy S20 Plus Dual SIM - 128GB': 3,
        'Apple iPhone Xs Max Without FaceTime - 512GB': 4,'Apple Iphone XS Max With Facetime - 512 GB': 5,'Apple iPhone 11 Pro with FaceTime - 512GB': 6,'Apple iPhone Xs Without FaceTime - 64GB': 7,
        'Samsung Galaxy S20 Ultra Dual SIM - 128GB': 8,'Samsung Galaxy Z Flip Dual SIM - 256GB':9,'Apple iPhone 11 Pro with FaceTime - 256GB':10,'Huawei P40 Pro Single SIM and E-SIM - 256GB': 11,
        'Samsung Galaxy Fold Dual SIM - 512GB':12,'Apple iPhone 7 Plus with FaceTime - 256GB':13,'Huawei Mate 20 Pro Dual Sim - 128 GB':14,'Apple Iphone XR With Face Time - 64 GB':15,
        'Apple iPhone 7 Plus with FaceTime - 128GB':16,'Apple iPhone 8 Plus without FaceTime - 64GB':17,'Sony Xperia 1 Dual SIM - 128GB':18,'Apple iPhone XR without Face Time - 64GB':19,'Apple iPhone Xs Max Dual SIM With FaceTime - 64GB':20,
        'Mione K1 ROM 3GB RAM 32GB 5.99 Inch display 4G - Black':21,'Cat S61 Dual Sim':22,'Thuraya XT -Pro with Prepay SIM Card':23,'Porsche design Huawei Mate 20 RS Dual SIM - 512GB':24,'Apple iPhone Xs Without FaceTime - 256GB':25,
        'Huawei Mate RS Porsche Design Dual SIM - 256GB':26,'Lamborghini 88 Tauri Dual Sim - 64GB':27,'Apple iPhone 11 Pro Max Dual SIM with FaceTime - 64GB':28,'Apple Iphone XS With Facetime - 512 GB':29,'Huawei Mate 9 Porsche Design Dual Sim - 256GB':30,
        'Google Glass Explorer Edition XE-C':31,'Blackberry P\'9981 Porsche Design - 8GB':32,'Samsung Galaxy S10 Plus Dual Sim - 1Tb':33},
    'Generation':{'4G LTE':0,'Gold':1,'5G':2,'5G - Black':3,'Silver':4,'Space Gray':5,'5G - Grey':6,'4G LTE - Black':7,'4G LTE - Purple':8,'4G':9,
        'Black':10,'64GB':11
        ,'Space Grey':12,'5G - Light Blue':13,'White':14,'Emerald Green':15,'Jet Black':16,'5G - Silver Frost':17,'White ':18,'4G LTE - Light Blue':19},
    'Color' : {'Black':0,'4 GB Ram':1,'Silver':2,'Midnight Green':3,'Red':4,'Space Gray':5,'Gold':6,'White':7,'Ceramic White':8,'6 GB Ram':9,'Purple':10,'3 GB Ram':11
        ,'Green':12,'Aura Glow':13,'Aura Black':14,'Blue':15,'Aura White':16,'Black/Grey':17,'6.5 Inch':18,'Yellow':19}
}

data = data.replace(convert)


xc=['name', 'Generation', 'Color', 'Number_Of_SIM','item_brand_name']
y=['0','1','3','4','5','6','7','8','9','10']
all_inputs =data[xc]
all_classes = data['item_brand_name']


(x_train,x_test,y_train,y_test)=train_test_split(all_inputs,all_classes,train_size=0.7 ,random_state= 1 )


clf = DecisionTreeClassifier(random_state=0)
clf.fit(x_train,y_train)
score= clf.score(x_test,y_test)
print(score)


from sklearn.tree import export_graphviz
from io import StringIO
from IPython.display import Image
import pydotplus
import graphviz


dot_data = StringIO()
export_graphviz(clf,out_file=dot_data,filled=True,rounded=True,special_characters=True,feature_names= xc,class_names=y)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('Tree.png')
Image(graph.create_png())
