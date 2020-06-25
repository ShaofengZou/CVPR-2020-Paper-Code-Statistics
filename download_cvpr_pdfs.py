# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os

# %% [markdown]
# ## Download the html

# %%
os.system("mkdir 'cache'")
os.system("wget 'http://openaccess.thecvf.com/CVPR2020.py' -O 'cache/CVPR2020.html'")

# %% [markdown]
# ## Download the papers

# %%
from urllib.request import urlretrieve
import os
os.makedirs('pdf', exist_ok=True)
 
file = open(os.path.join('cache', 'CVPR2020.html'), 'r')
html = file.read()
 
while html.find('.pdf') != -1:
    # find the first .pdf file
    paper = html[:html.find('.pdf')+4]
    paper = paper[paper.rfind('"')+1:]

    # download the pdf into the content folder
    urlretrieve('http://openaccess.thecvf.com/' + paper, 'pdf/' + paper[paper.rfind('/')+1:])

    # update the html so it no longer contains that pdf
    html = html[html.find('.pdf')+1:]


# %%


