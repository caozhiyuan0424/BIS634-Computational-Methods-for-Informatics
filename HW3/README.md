<h1 align = "center">BIS634 - Assignment3</h1> 
<p align="right">Name: Zhiyuan Cao; NetID: zc347</p>

## Exercise 1

### Q1

I use the requests module as follows to acquire PMID of 1000 Alzheimers papers from 2022.

![EX1_1](/Users/caozhiyuan/Desktop/BIS634-HW3/README_img/EX1_1.png)

Similarly I do the same for Cancer papers.

![EX1_2](/Users/caozhiyuan/Desktop/BIS634-HW3/README_img/EX1_2.png)

### Q2

I save two json files as instructed. See my github file for detail. Part of the json file is as follows.

![EX1_3](/Users/caozhiyuan/Desktop/BIS634-HW3/README_img/EX1_3.png)

For the abstract, if the article have multiple AbstractText fields, I store all the parts by simply concatenating with a space in between.  

Pros:

- Easy to implement
- Whole structure less complicated

Cons:

- Abstract becomes a single paragraph

### Q3

By using 

```python
set(AlzheimersList) & set(CancerList)
```

I find that there is no overlap in the two sets of papers that I identified.



## Exercise 2

### Q1

I compute the SPECTER embedding of each paper in EX1 as follows. See my code for detail.

![EX2_1](/Users/caozhiyuan/Desktop/BIS634-HW3/README_img/EX2_1.png)

### Q2

I load the paper using the following code

```python
f1 = open('Alzheimers.json')
f2 = open('Cancer.json')

Alzheimers = json.load(f1)
Cancer = json.load(f2)
```

Then I process using the following code 

```python
import tqdm

# we can use a persistent dictionary (via shelve) so we can stop and restart if needed
# alternatively, do the same but with embeddings starting as an empty dictionary
embeddings = {}
for pmid, paper in tqdm.tqdm(papers.items()):
    data = [paper["ArticleTitle"] + tokenizer.sep_token + get_abstract(paper)]
    inputs = tokenizer(
        data, padding=True, truncation=True, return_tensors="pt", max_length=512
    )
    result = model(**inputs)
    # take the first token in the batch as the embedding
    embeddings[pmid] = result.last_hidden_state[:, 0, :].detach().numpy()[0]

# turn our dictionary into a list
embeddings = [embeddings[pmid] for pmid in papers.keys()]
```

### Q3

Then I perform PCA using the following code

```python
from sklearn import decomposition
pca = decomposition.PCA(n_components=3)
embeddings_pca = pd.DataFrame(
    pca.fit_transform(embeddings),
    columns=['PC0', 'PC1', 'PC2']
)
embeddings_pca["query"] = [paper["query"] for paper in papers.values()]
```

The result is 

![EX2_2](/Users/caozhiyuan/Desktop/BIS634-HW3/README_img/EX2_2.png)

### Q4

Finally, I plot three scatter plots for PC0 vs PC1, PC0 vs PC2, and PC1 vs PC2. 

![EX2_3](/Users/caozhiyuan/Desktop/BIS634-HW3/README_img/EX2_3.png)

![EX2_4](/Users/caozhiyuan/Desktop/BIS634-HW3/README_img/EX2_4.png)

![EX2_5](/Users/caozhiyuan/Desktop/BIS634-HW3/README_img/EX2_5.png)

**Comment**: PC0 vs. PC1 performs the best among these three graphs because it uses the principle components with the biggest two eignevalues. The two class are separated pretty far away. PC0 vs. PC1 preform worse than PC0 vs. PC2, but better than PC1 vs. PC2. Finally, PC1 vs. PC2 is the worst, and we can hardly tell one class from another. 



## Exercise 3

### Q1

I write a class ``SIR_Model()`` to realize an Explicit Euler method to plot *i(t)*. There are several main functions inside the class.

- ``__init__``: Set parameters of s, i, r, beta and gamma. The default values are s=133999, i=1, r=0, beta=2, gamma=1, Tmax=30.
- ``update``: To update the value of s, i and r for one iteration.
- ``evolve``: Evolve the population information to Tmax. 
- ``plot_graph``: plot the graph for t vs. population.

See my code for detailed implementation. 

### Q2

By these initial values, the time course of the number of infected individuals are plotted below. 

![EX3_1](/Users/caozhiyuan/Desktop/BIS634-HW3/README_img/EX3_1.png)

### Q3

Two functions help me obtain that values. 

- I write a function named ``time_of_peak``. When t = 16, the number of infected people reaches its peak.
- I write another function named ``iValue_of_peak``. 26033 people are infected at the peak.

See my code for detailed implementation. 

### Q4

After that, I vary $\beta$ and $\gamma$ to be different values ranging from 0.1 to 3 and 0.5 to 2 respectively. Then I plot on a heat map how the time of the peak of the infection depends on these two variables. The result is shown below

![EX3_2](/Users/caozhiyuan/Desktop/BIS634-HW3/README_img/EX3_2.png)

### Q5

I do the same for the number of individuals infected at peak. The result is shown below.

![EX3_3](/Users/caozhiyuan/Desktop/BIS634-HW3/README_img/EX3_3.png)



## Exercise 4

### Dataset Identification 

The dataset I find: https://www.kaggle.com/datasets/harikrishnareddyb/used-car-price-predictions

It is "Used car price predictions". 

### Dataset Description

The data contains 8 columns, which means 8 variables. Some sample data are shown below.

| Price | Year | Mileage | City             | State | Vin               | Make  | Model      |
| ----- | ---- | ------- | ---------------- | ----- | ----------------- | ----- | ---------- |
| 8995  | 2014 | 35725   | El Paso          | TX    | 19VDE2E53EE000083 | Acura | ILX6-Speed |
| 10888 | 2013 | 19606   | Long Island City | NY    | 19VDE1F52DE012636 | Acura | ILX5-Speed |
| 8995  | 2013 | 48851   | El Paso          | TX    | 19VDE2E52DE000025 | Acura | ILX6-Speed |

Specifically, 

- Price: Target Variable.
- Year: Year of the car purchased.
- Mileage: The no.of kms drove by the car.
- City: In which city it was sold.
- State: In which state it was sold.
- Vin: A unique number for a car.
- Make: Manufacturer of the car.
- Model: The model(name) of the car.

The key variables have been explicitly specified, which means we do not need to do further derive. 

No variable is redundant. 

Variable "State" can be predicted from variable "City". 

There are 852123 data points available. 

The data is in standard format. 

### Terms of use & Restrictions

I do not have to officially apply to get access to the data. These does not exist certain type of analyses I can't do. 

### Data Exploration

I plot some histograms:

![EX4_1](/Users/caozhiyuan/Desktop/BIS634-HW3/README_img/EX4_1.png)

![EX4_2](/Users/caozhiyuan/Desktop/BIS634-HW3/README_img/EX4_2.png)

![EX4_3](/Users/caozhiyuan/Desktop/BIS634-HW3/README_img/EX4_3.png)

From these figures, I can observe that most cars have price lower than 100000. Most cars are purchased in 2014 or 2015. And the mileage tems to be less than 50000. 

### Data Cleaning Needs

There is no need to perform data cleaning.  Because I have checked all the dataset and there is no missing data, which means the dataset has already been cleaned. 



## Appendix: Python Code

### Exercise 1

```python
from Bio import Entrez
import xml.dom.minidom as m
import requests
import json

def getid_from_term(num, term):
    r = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        f"esearch.fcgi?db=pubmed&retmode=xml&retmax={num}&term={term}"
    )
    doc = m.parseString(r.text)
    IdLists = doc.getElementsByTagName("Id")
    IdList = [IdLists[i].childNodes[0].wholeText for i in range(num)]
    return IdList

AlzheimersList = getid_from_term(1000, 'Alzheimer+AND+2022[pdat]')
print(f"The IDs for 1000 Alzheimer papers from 2022 are {AlzheimersList}")

def getid_from_term(num, term):
    r = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        f"esearch.fcgi?db=pubmed&retmode=xml&retmax={num}&term={term}"
    )
    doc = m.parseString(r.text)
    IdLists = doc.getElementsByTagName("Id")
    IdList = [IdLists[i].childNodes[0].wholeText for i in range(num)]
    return IdList

CancerList = getid_from_term(1000, 'Cancer+AND+2022[pdat]')
print(f"The IDs for 1000 Cancer papers from 2022 are {CancerList}")
```

```python
def get_info(pmid, query):
    r = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        f"efetch.fcgi?db=pubmed&retmode=xml&id={pmid}"
    )
    doc = m.parseString(r.text)
    dict_articles = {}
    for i in doc.getElementsByTagName("PubmedArticle"):
        PMID = i.getElementsByTagName("PMID")[0].childNodes[0].wholeText
        title_i = i.getElementsByTagName("ArticleTitle")[0].childNodes # title containing italics
        title = ""
        for item in title_i:
            title += item.toxml()
        try:
            abstracts = i.getElementsByTagName("Abstract")[0].getElementsByTagName("AbstractText")
            abstract = ""
            for item in (abstracts[0].childNodes):
                abstract += item.toxml()
            # If there are more than one abstract, concancate with space between them
            if len(abstracts) > 1:
                for j in range(1,len(abstracts)):
                    abstract += " "
                    for item in (abstracts[j].childNodes):
                        abstract += item.toxml()
        except:
            abstract = ""
        dict_article = {"ArticleTitle": title,
                       "AbstractText": abstract,
                       "query": query}
        dict_articles[PMID] = dict_article
    return dict_articles

IDstr = ""
for item in AlzheimersList[:400]:
    IDstr += item
    IDstr += ","
IDstr = IDstr[:-1]
result = get_info(IDstr, "Alzheimer")

IDstr = ""
for item in AlzheimersList[400:800]:
    IDstr += item
    IDstr += ","
IDstr = IDstr[:-1]
temp = get_info(IDstr, "Alzheimer")
result.update(temp)

IDstr = ""
for item in AlzheimersList[800:]:
    IDstr += item
    IDstr += ","
IDstr = IDstr[:-1]
temp = get_info(IDstr, "Alzheimer")
result.update(temp)



with open("Alzheimers.json", "w") as outfile:
    json.dump(result, outfile)
```

```python
IDstr = ""
for item in CancerList[:400]:
    IDstr += item
    IDstr += ","
IDstr = IDstr[:-1]
result = get_info(IDstr, "Cancer")

IDstr = ""
for item in CancerList[400:800]:
    IDstr += item
    IDstr += ","
IDstr = IDstr[:-1]
temp = get_info(IDstr, "Cancer")
result.update(temp)

IDstr = ""
for item in CancerList[800:]:
    IDstr += item
    IDstr += ","
IDstr = IDstr[:-1]
temp = get_info(IDstr, "Cancer")
result.update(temp)



with open("Cancer.json", "w") as outfile:
    json.dump(result, outfile)
```

### Exercise 2

```python
import json
import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
model = AutoModel.from_pretrained('allenai/specter')

f1 = open('Alzheimers.json')
f2 = open('Cancer.json')

Alzheimers = json.load(f1)
Cancer = json.load(f2)
```

```python
embeddings = embedding_(papers)
```

```python
from sklearn import decomposition
pca = decomposition.PCA(n_components=3)
embeddings_pca_all = pd.DataFrame(
    pca.fit_transform(embeddings),
    columns=['PC0', 'PC1', 'PC2']
)
embeddings_pca_all["query"] = [paper["query"] for paper in papers.values()]
```

```python
plt.scatter(embeddings_pca_all[embeddings_pca_all['query'] == 'Alzheimer']['PC0'], embeddings_pca_all[embeddings_pca_all['query'] == 'Alzheimer']['PC1'], label="Alzheimer")
plt.scatter(embeddings_pca_all[embeddings_pca_all['query'] == 'Cancer']['PC0'], embeddings_pca_all[embeddings_pca_all['query'] == 'Cancer']['PC1'], label="Cancer")
plt.xlabel("PC0")
plt.ylabel("PC1")
plt.legend()
plt.show()
```

```python
plt.scatter(embeddings_pca_all[embeddings_pca_all['query'] == 'Alzheimer']['PC0'], embeddings_pca_all[embeddings_pca_all['query'] == 'Alzheimer']['PC2'], label="Alzheimer")
plt.scatter(embeddings_pca_all[embeddings_pca_all['query'] == 'Cancer']['PC0'], embeddings_pca_all[embeddings_pca_all['query'] == 'Cancer']['PC2'], label="Cancer")
plt.xlabel("PC0")
plt.ylabel("PC2")
plt.legend()
plt.show()
```

```python
plt.scatter(embeddings_pca_all[embeddings_pca_all['query'] == 'Alzheimer']['PC1'], embeddings_pca_all[embeddings_pca_all['query'] == 'Alzheimer']['PC2'], label="Alzheimer")
plt.scatter(embeddings_pca_all[embeddings_pca_all['query'] == 'Cancer']['PC1'], embeddings_pca_all[embeddings_pca_all['query'] == 'Cancer']['PC2'], label="Cancer")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.show()
```

### Exercise 3

```python
import matplotlib.pyplot as plt
import numpy as np
class SIR_Model():
    
    def __init__(self, s=133999, i=1, r=0, beta=2, gamma=1, Tmax=30):
        self._s = s
        self._i = i
        self._r = r
        self._beta = beta
        self._gamma = gamma
        self._Tmax = Tmax
        self._N = s + i + r
        
    def update(self):
        s_new = self._s - self._s*self._i*self._beta/self._N
        i_new = self._i + self._s*self._i*self._beta/self._N - self._gamma*self._i
        r_new = self._r + self._gamma*self._i
        self._s = s_new
        self._i = i_new
        self._r = r_new
        return self._s, self._i, self._r
    
    def evolve(self):
        s = [self._s,]
        i = [self._i,]
        r = [self._r,]
        t = range(self._Tmax)
        for time in range(self._Tmax-1):
            s_temp, i_temp, r_temp = self.update()
            s.append(s_temp)
            i.append(i_temp)
            r.append(r_temp)
        return s, i, r

    def time_of_peak(self):
        s, i, r = self.evolve()
        return np.argmax(i)
    
    def iValue_of_peak(self):
        s, i, r = self.evolve()
        return max(i)
    
    def plot_graph(self):
        s, i, r = self.evolve()
        t = range(self._Tmax)
        plt.plot(t,s,label='s')
        plt.plot(t,i,label='i')
        plt.plot(t,r,label='r')
        plt.xlabel("Time")
        plt.ylabel("Population")
        plt.legend()
        plt.savefig("README_img/EX3_1.png")
        plt.show()

sir = SIR_Model()
sir.plot_graph()
```

```python
def heatplot_peak_of_time():
    x_lim = 30
    y_lim = 30
    peak_time = [[0]*x_lim for i in range(y_lim)]
    beta_x = np.linspace(0.1, 3, x_lim)
    gamma_y = np.linspace(0.5, 2, y_lim)
    for i in range(len(beta_x)):
        for j in range(len(gamma_y)):
            beta_x[i] = round(beta_x[i],2)
            gamma_y[i] = round(gamma_y[i],2)
            peak_time[i][j] = SIR_Model(beta=beta_x[i], gamma=gamma_y[j], Tmax=1000).time_of_peak()
    sns.heatmap(peak_time, xticklabels=beta_x, yticklabels=gamma_y)
    plt.xlabel("beta")
    plt.ylabel("gamma")
    plt.title("The time of the peak of the infection vs. beta & gamma")
    plt.savefig("README_img/EX3_2.png")
    plt.show()
    
print(f"The number of infected people peak at t = {SIR_Model().time_of_peak()}, and {round(SIR_Model().iValue_of_peak())} people are infected at the peak.")
heatplot_peak_of_time()
```

```python
def heatplot_peak_of_number():
    x_lim = 30
    y_lim = 30
    peak_time = [[0]*x_lim for i in range(y_lim)]
    beta_x = np.linspace(0.1, 3, x_lim)
    gamma_y = np.linspace(0.5, 2, y_lim)
    for i in range(len(beta_x)):
        for j in range(len(gamma_y)):
            beta_x[i] = round(beta_x[i],2)
            gamma_y[i] = round(gamma_y[i],2)
            peak_time[i][j] = SIR_Model(beta=beta_x[i], gamma=gamma_y[j], Tmax=1000).iValue_of_peak()
    sns.heatmap(peak_time, xticklabels=beta_x, yticklabels=gamma_y)
    plt.xlabel("beta")
    plt.ylabel("gamma")
    plt.title("The Number of infection at the peak vs. beta & gamma")
    plt.savefig("README_img/EX3_3.png")
    plt.show()
    
heatplot_peak_of_number()
```

### Exercise 4

```python
import pandas as pd
df = pd.read_csv("true_car_listings.csv")

import matplotlib.pyplot as plt
plt.hist(df['Price'], bins=60)
plt.xlabel("Price")
plt.ylabel("Count")
plt.savefig("README_img/EX4_1.png")
plt.show()

plt.hist(df['Year'], bins=20)
plt.xlabel("Year")
plt.ylabel("Count")
plt.savefig("README_img/EX4_2.png")
plt.show()

plt.hist(df['Mileage'], bins=100)
plt.xlabel("Mileage")
plt.ylabel("Count")
plt.savefig("README_img/EX4_3.png")
plt.show()
```
