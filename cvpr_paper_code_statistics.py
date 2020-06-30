# %%
import os
import re
import tqdm
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from unidecode import unidecode
from multiprocessing import Pool
from sklearn.feature_extraction.text import TfidfVectorizer

year = '2020'
os.makedirs('info', exist_ok=True)

def get_score(row):
    score = row.stars if not row.stars == '' else 0
    if len(row['github']) > 0:
        score += 0.5
    return score

def get_paper_info():
    print('Getting paper info...')
    info = {}
    base_url = 'http://openaccess.thecvf.com/CVPR%s.py' % year
    soup = BeautifulSoup(requests.get(base_url).content, "html.parser")
    for a in tqdm.tqdm(soup.find_all('a')):
        link = a.get('href')
        if link is None or link[-5:] != '.html':
            continue
        link = "http://openaccess.thecvf.com/{}".format(a.get('href'))
        paper = BeautifulSoup(requests.get(link).content, "html.parser")
        title = unidecode(paper.find('div', {'id': 'papertitle'}).text.lstrip())
        authors = unidecode(paper.find('i').text).split(',  ')
        abstract = unidecode(paper.find('div', {'id': 'abstract'}).text.strip())
        info[title.lower()] = {'authors': authors, 'abstract': abstract, 'title': title}
    return info

def get_row(row):
    return {'author': row['authors'][0],
        'github': row['github'] if 'github' in row else '',
        'stars': row['stars'] if 'stars' in row else '',
        'task 1': row['tasks'][0] if 'stars' in row and row['tasks'][0] is not None else '',
        'task 2': row['tasks'][1] if 'stars' in row and row['tasks'][1] is not None else '',
        'task 3': row['tasks'][2] if 'stars' in row and row['tasks'][2] is not None else '',
        'title': row['title'],
        }


def get_github(rowi):
    row = info[rowi]
    # print(row)
    url = 'https://paperswithcode.com/search?q={}'.format(row['title'].replace(' ', '+'))
    results = BeautifulSoup(requests.get(url).content, 'html.parser')
    has_code = []
    for res in results.find_all('div', {'class': 'infinite-item'}):
        link = res.find('a', {'class': 'badge-dark'})
        if unidecode(link.text.strip()) == 'Code':
            has_code.append('https://paperswithcode.com{}'.format(link.get('href')))
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([row['abstract']]).A
    Y = []
    papers = []
    tasks = np.empty(5, dtype=object)
    for link in has_code:
        paper = BeautifulSoup(requests.get(link).content, 'html.parser')
        papers.append(paper)
        paperabs = unidecode(paper.find('div', {'class': 'paper-abstract'}).find('p').text.strip().replace('\n', ' ').replace('...', ' ').replace(' (read more)', ''))
        Y.append(paperabs)

        results = paper.find('div', {'class': 'col-md-5 paper-section'}).find('div', {'class': 'paper-section-title'}).find('div', {'class': 'paper-tasks'}).find_all('ul', {'class':'list-unstyled'})
        for idx, tmp in enumerate(results):
            task = tmp.find('a').get('href').split('/')[-1]
            tasks[idx] = task

    if len(Y) > 0:
        Y = vectorizer.transform(Y).A
        scores = np.matmul(Y, X.T)[:,0]
        best_ix = np.argmax(scores)
        if scores[best_ix] > .85:
            paper = papers[best_ix]
            code = paper.find('div', {'class': 'paper-implementations'}).find('div', {'class': 'row'})
            github = code.find('a', {'class': 'code-table-link'}).get('href')
            stars = int(unidecode(code.find_all('div', {'class': 'paper-impl-cell'})[1].text).strip().replace(' ', '').replace(',', ''))
            return rowi, {'github': github, 'stars': stars, 'tasks': tasks}
 
    return rowi, {}

def get_scholar(rowi):
    row = info[rowi]   
    
    proxies = {"http": "http://127.0.0.1:{}/".format(proxy), "https": "http://127.0.0.1:{}/".format(proxy)}
    url = 'https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q={}'.format(row['title'].replace(' ', '+'))
    content = requests.get(url, proxies=proxies).content.decode('latin-1')
    search_reference_rates = re.findall('Cited by .*?</a>', content, re.MULTILINE)
    reference_rates = []
    for search_reference_rate in search_reference_rates:
        if search_reference_rate:
            search_reference_rate_id_start = 9
            search_reference_rate_id_end = search_reference_rate.find('<')
            reference_rate = search_reference_rate[search_reference_rate_id_start:search_reference_rate_id_end]
            print('reference_rate:', reference_rate)
            reference_rates.append(reference_rate)
    return rowi, {'reference_rate': reference_rates}     

# %% [markdown]
# ## Step 1. Get paper information

# %%
paper_info_filename = os.path.join('cache', 'CVPR%s_paper_info.npy' % year)
try:
    info = np.load(paper_info_filename, allow_pickle=True).tolist()
    print('Load paper information for CVPR %s successfully' % year)
except:
    info = get_paper_info()
    np.save(paper_info_filename, info)
    print('Save paper information for CVPR %s successfully' % year)

# %% [markdown]
# ## Step 2. Get github information

# %%
print('Getting github info...')
github_info_filename = os.path.join('cache', 'CVPR%s_github_info.npy' % year)
try:
    github_info = np.load(github_info_filename, allow_pickle=True).tolist()
    print('Load github information for CVPR %s successfully' % year)
except:
    github_info = {}
#     tmp = list(info.keys())[:20]
    for key in tqdm.tqdm(info.keys()):
        try:
            row, single_info = get_github(key)
            github_info[row] = single_info
        except:
            print('Failed in paper:', key)
    np.save(github_info_filename, github_info)
    print('Save github information for CVPR %s successfully' % year)


# %%
for k, val in github_info.items():
    info[k].update(val)

# %% [markdown]
# ## Step 3. Save to csv files

# %%
list(info.items())[0]


# %%
info_result = map(get_row, info.values())
df = pd.DataFrame(info_result)
df['score'] = df.apply(get_score, axis=1)
df = df.sort_values(['score', 'github'], ascending=False)
df = df[['title', 'author', 'task 1', 'task 2', 'task 3', 'github', 'stars']]
df.index = range(1, len(df) + 1)
outname = os.path.join('info', 'CVPR%s_info.csv' % year)
print('Saving to ', outname)
df.to_csv(outname)


# %%
df

# %% [markdown]
# ## 4. Topics anaysis

# %%
topics_list = df['task 1'].tolist() + df['task 2'].tolist() + df['task 3'].tolist()
all_topics= set(topics_list)
all_topics


# %%
from collections import Counter
import matplotlib.pyplot as plt
keyword_counter = Counter(topics_list)


# %%
topic_count = pd.DataFrame([(ele, keyword_counter[ele]) for ele in keyword_counter], columns=['topic', 'count']).sort_values(by=['count'], ascending=False)
topic_count.to_csv(os.path.join('info', 'CVPR%s_topic_count.csv' % year))
topic_count


# %%
keyword_counter['image-super-resolution']


# %%
# Show N most common keywords and their frequencies
num_keyowrd = 75
keywords_counter_vis = keyword_counter.most_common(num_keyowrd)

plt.rcdefaults()
fig, ax = plt.subplots(figsize=(8, num_keyowrd//5), dpi=144)

key = [k[0] for k in keywords_counter_vis] 
value = [k[1] for k in keywords_counter_vis] 
y_pos = np.arange(len(key))
ax.barh(y_pos, value, align='center', color='green', ecolor='black', log=True)
ax.set_yticks(y_pos)
ax.set_yticklabels(key, rotation=0, fontsize=10)
ax.invert_yaxis() 
for i, v in enumerate(value):
    ax.text(v + 3, i + .25, str(v), color='black', fontsize=10)
# ax.text(y_pos, value, str(value))
ax.set_xlabel('Frequency')
ax.set_title('CVPR %s Submission Top %d Topics' % (year, num_keyowrd))

plt.savefig(os.path.join('info', 'CVPR%s_topics.png' % year), bbox_inches='tight', pad_inches=0)
plt.show()


# %%
# Show the word cloud forming by keywords
from wordcloud import WordCloud
wordcloud = WordCloud(max_font_size=64, max_words=160, 
                      width=1280, height=640,
                      background_color="black").generate(' '.join(topics_list))
plt.figure(figsize=(16, 8), dpi=144)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.savefig(os.path.join('info', 'CVPR%s_wordcloud.png' % year), bbox_inches='tight', pad_inches=0)
plt.show()

# %% [markdown]
# ## Search papers according to keyword and topic

# %%
def pull_papers(paper_infos, topics, search_range='all'):
    target_title = pd.DataFrame()
    for topic in topics:
        tmp = paper_infos[paper_infos['title'].str.contains(topic.capitalize())]
        target_title = pd.concat([target_title, tmp]).drop_duplicates()

    target_topic = pd.DataFrame()
    for topic in topics:
        for i in range(3):
            tmp = paper_infos[paper_infos['task %d' % (i+1)].str.contains(topic)]
            target_topic = pd.concat([target_topic, tmp]).drop_duplicates()

    if search_range == 'title':
        return target_title
    elif search_range == 'topic':
        return target_topic
    elif search_range == 'all':
        return pd.concat([target_title, target_topic]).drop_duplicates()   


# %%
pull_papers(df, ['denoising'])

# %% [markdown]
# ## Classify topic

# %%
topic_class = pd.read_excel('topic_class.xlsx')


# %%
topic_class


# %%
def count_class_topic(topic_class, target_class):
    target_keyword = []
    topics = list(topic_class[target_class])
    clean_topics = [x for x in topics if str(x)!='nan']
    num_keyowrd = len(clean_topics)
    print('num_keyowrd of %s is:%d' % (target_class, num_keyowrd))
    for x in clean_topics:
        target_keyword.extend([x]*keyword_counter[x])

    target_keyword_counter = Counter(target_keyword)

    # Show N most common keywords and their frequencies
    keywords_counter_vis = target_keyword_counter.most_common(num_keyowrd)

    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(8, num_keyowrd//5), dpi=144)

    key = [k[0] for k in keywords_counter_vis] 
    value = [k[1] for k in keywords_counter_vis] 
    y_pos = np.arange(len(key))
    ax.barh(y_pos, value, align='center', color='green', ecolor='black', log=True)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(key, rotation=0, fontsize=10)
    ax.invert_yaxis() 
    for i, v in enumerate(value):
        ax.text(v + 3, i + .25, str(v), color='black', fontsize=10)

    ax.set_xlabel('Frequency')
    ax.set_title('Topic of class %s' % (target_class))

    plt.savefig(os.path.join('info', 'Topic of class %s.png') % (target_class.upper()), bbox_inches='tight', pad_inches=0)
    plt.close()


# %%
count_class_topic(topic_class, 'a')
count_class_topic(topic_class, 'b')
count_class_topic(topic_class, 'c')
count_class_topic(topic_class, 'd')


# %%
def get_and_save(target_class):
    topics = list(topic_class[target_class])
    clean_topics = [x for x in topics if str(x)!='nan']
    clean_infos = pull_papers(df, clean_topics, 'topic').reset_index(drop=True)
    print('%s have %d papers' % (target_class, len(clean_infos)))
    save_path = os.path.join('info', '%s.csv' % target_class.upper())
    clean_infos.to_csv(save_path)
    return clean_infos


# %%
get_and_save('a')


# %%
get_and_save('b')


# %%
get_and_save('c')


# %%
get_and_save('d')


# %%



# %%


