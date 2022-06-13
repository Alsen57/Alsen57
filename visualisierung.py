#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 00:05:46 2022

@author: alisentuerk
"""

#visualisierungen
import plotly
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import plotly.tools as tls
import plotly.io as pio
pio.renderers.default='browser'


#%%
# import dataframe

frame=pd.read_csv("/Users/alisentuerk/Desktop/IconPro-Task/dataset.csv")
target=frame["target_class"]
frame=frame.drop(columns=["target_class"])

#drop additional categorical classes if not binarized
frame=frame.drop(columns=["C"])
#%%
# PCA inklusive Color coding 3 Achsen
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


framePCA = StandardScaler().fit_transform(frame)
#%%
#2D
pca = PCA(n_components=2)
framePCA=pca.fit_transform(framePCA)
framePCA=pd.DataFrame(framePCA)
framePCA.columns = ["pca1","pca2"]
plt.scatter(framePCA["pca1"],framePCA["pca2"],c=target)
plt.show()
#%%3D
pca = PCA(n_components=3)
framePCA=pca.fit_transform(framePCA)
framePCA=pd.DataFrame(framePCA)
framePCA.columns = ["pca1","pca2","pca3"]
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(framePCA["pca1"],framePCA["pca2"],framePCA["pca3"],c=target)
plt.show()

#%%
#Pandas profiling
from pandas_profiling import ProfileReport
profile = ProfileReport(frame, title="Pandas Profiling Report", explorative=True)
profile.to_file("/Users/alisentuerk/Desktop/report.html")

#%%
#umap
import plotly.express as px
from umap import UMAP
import pandas as pd

data=pd.read_csv("/Users/alisentuerk/Desktop/IconPro-Task/dataset.csv")
data=data.iloc[:1000,:]
target=data["target_class"]
features=data.drop(columns=["target_class","C"])

umap_2d = UMAP(n_components=2, init='random', random_state=0)
umap_3d = UMAP(n_components=3, init='random', random_state=0)

proj_2d = umap_2d.fit_transform(features)
proj_3d = umap_3d.fit_transform(features)

fig_2d = px.scatter(
    proj_2d, x=0, y=1,
    color=target)

fig_3d = px.scatter_3d(
    proj_3d, x=0, y=1, z=2,
    color=target)
fig_3d.update_traces(marker_size=5)

fig_2d.show()
fig_3d.show()

#%%
from sklearn.manifold import TSNE
import plotly.express as px

data=pd.read_csv("/Users/alisentuerk/Desktop/IconPro-Task/dataset.csv")
data=data.iloc[:1000,:]
target=data["target_class"]
features=data.drop(columns=["target_class","C"])

tsne = TSNE(n_components=3, random_state=0)
projections = tsne.fit_transform(features, )

fig = px.scatter_3d(
    projections, x=0, y=1, z=2,
    color=target)
fig.update_traces(marker_size=8)
fig.show()

#%%
#correlation matrix
import plotly.express as px
import pandas as pd
data=pd.read_csv("/Users/alisentuerk/Desktop/IconPro-Task/dataset.csv")
target=data["target_class"]
features=data.drop(columns=["target_class","C"])
columnNames=features.columns
fig = px.scatter_matrix(features, dimensions=columnNames, color=target)
fig.show()

#%%
#shapley values
import pandas as pd
data=pd.read_csv("/Users/alisentuerk/Desktop/IconPro-Task/dataset.csv")
target=data["target_class"]
features=data.drop(columns=["target_class","C"])

import xgboost
import shap
model_xgb = xgboost.XGBRegressor(n_estimators=10, max_depth=2).fit(features, target)

# explain the GAM model with SHAP
explainer_xgb = shap.Explainer(model_xgb, features)
shap_values = explainer_xgb(features)

#%%
shap.plots.bar(shap_values)
shap.plots.bar(shap_values.abs.max(0))
shap.plots.beeswarm(shap_values)
