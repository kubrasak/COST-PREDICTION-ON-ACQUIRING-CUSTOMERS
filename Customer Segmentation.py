import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)




################################################
#  Exploratory Data Analysis
################################################
df = pd.read_csv(r"C:\Users\KUBRA SAK\PycharmProjects\pythonProject\dsmlbc_9_gulbuke\Homeworks\Kubra_Sak\Project\media prediction and its cost.csv")



df["marj"]=((df['store_sales(in millions)']- df[ 'store_cost(in millions)'])/df["unit_sales(in millions)"])*df["units_per_case"]



df.loc[df["avg. yearly_income"] == "$10K - $30K", "NEW_WELFARE_LEVEL"]=20
df.loc[df["avg. yearly_income"] == "$30K - $50K", "NEW_WELFARE_LEVEL"]=40
df.loc[df["avg. yearly_income"] == "$50K - $70K", "NEW_WELFARE_LEVEL"]=60
df.loc[df["avg. yearly_income"] == "$70K - $90K", "NEW_WELFARE_LEVEL"]=80
df.loc[df["avg. yearly_income"] == "$90K - $110K", "NEW_WELFARE_LEVEL"]=100
df.loc[df["avg. yearly_income"] == "$110K - $130K", "NEW_WELFARE_LEVEL"]=120
df.loc[df["avg. yearly_income"] == "$130K - $150K", "NEW_WELFARE_LEVEL"]=140
df.loc[df["avg. yearly_income"] == "$150K +", "NEW_WELFARE_LEVEL"]=180


knn_df=df.copy()

knn_df.head()


knn_df=knn_df[["cost",  "marj"]]



#knn_df["NEW_MARGIN"]=(knn_df["store_sales(in millions)"]-knn_df["store_cost(in millions)"])/knn_df["store_sales(in millions)"]

#knn_data=knn_df.drop(["store_cost(in millions)","store_sales(in millions)","unit_sales(in millions)"], axis=1)

knn_data=knn_df.copy()

sc = MinMaxScaler((0, 1))
knn_sc = sc.fit_transform(knn_data)
df[0:5]

kmeans = KMeans(n_clusters=5).fit(knn_sc)
kmeans.get_params()


clusters_kmeans = kmeans.labels_
knn_data["clusters_kmeans"]=clusters_kmeans

knn_data.groupby("clusters_kmeans").median()

df["cluster"] = clusters_kmeans

df.to_csv("knn_cluster.csv")

plt.scatter(df.loc[:,"store_sales(in millions)"], df.loc[:,"store_cost(in millions)"],c = clusters_kmeans, s = 50, cmap = "viridis")
plt.show()


plt.scatter(knn_df.loc[:,"cost"], df.loc[:,"marj"],c = clusters_kmeans, s = 50, cmap = "viridis")
plt.show()

import seaborn as sns

knn_data.head()

sns.boxplot(y=knn_data["cost"], x=knn_data["clusters_kmeans"])
plt.show()

sns.boxplot(y=knn_data["store_sales(in millions)"], x=knn_data["clusters_kmeans"])
plt.show()

sns.boxplot(y=knn_data["NEW_MARGIN"], x=knn_data["clusters_kmeans"])
plt.show()

sns.boxplot(y=knn_data["units_per_case"], x=knn_data["clusters_kmeans"])
plt.show()
################################
# Determining the Optimum Number of Clusters
################################

kmeans = KMeans()
ssd = []
K = range(1, 10)

for k in K:
    kmeans = KMeans(n_clusters=5).fit(knn_sc)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 10))
elbow.fit(knn_sc)
elbow.show()

elbow.elbow_value_