# Genetic Algorithm and Machine Learning-based Stock Selection (GA + ML Models)

>  This project is a **decision-support** for stock analysis.

> It may help you analyze stocks, but if you lose money, don’t come looking for me.
 
It combines:  
- **Genetic Algorithm (GA):** Explores and selects the most informative feature subsets.  
- **Machine Learning Models (XGBoost, SVM, RF, etc.):** Act as evaluators to measure predictive power of selected features.  

The GA iteratively evolves feature subsets based on prediction accuracy, converging toward an optimal combination.  

---

## Project Workflow

### 0) Dataset Splitting & Evaluation
- **Time-based split:** Train on earlier intervals, test on forward intervals (avoid leakage).  
- **Metrics:** `Accuracy`, `F1`, `ROC-AUC` (or `annualized return`, `Sharpe` for strategy evaluation).  
- **Validation:** `Walk-forward` or `TimeSeriesSplit`.  

---

### 1) Preprocessing

```python
def preprocessing(file):
    #* 讀取資料
    stock_data = pd.read_excel(file)
    stock_data = stock_data.drop(columns=['證券代碼'])
    #? print(stock_data)

    #* 擷取股票年月與簡稱
    stock_year = stock_data['年月'].unique()
    stock_name = stock_data['簡稱'].unique()
    #? print(stock_year)
    #? print(stock_name)

    #* 移除第2009年
    stock = stock_data[stock_data['年月'].isin(stock_year[:-1])]
    #? print(stock)
    
    return stock_data
```

---

### 2) Normalization

```python
def normalization(data):
    for col in data.columns[2:-2]:
        data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
        
    return data
def split_train_test(stock, TV):
    #* 擷取股票年月與
    stock_year = stock['年月'].unique()
    
    basic_train_data = stock[stock['年月'].isin(stock_year[:TV])]
    train_label = basic_train_data['ReturnMean_year_Label']
    train_data = basic_train_data.drop(columns=['簡稱', '年月', 'Return', 'ReturnMean_year_Label'])

    basic_test_data = stock[stock['年月'].isin(stock_year[TV:])]
    test_label = basic_test_data['ReturnMean_year_Label']
    test_data = basic_test_data.drop(columns=['簡稱', '年月', 'Return', 'ReturnMean_year_Label'])
    
    return train_data, train_label, test_data, test_label
```

---

### 3) Chromosome Encoding & Decoding
```python
def decode(chromosome):
    #? print(chromosome)
    dna_len = len(chromosome)
    num = dna_len // 5
    code = np.ones((num), dtype='int')
    
    index = 0
    for i in range(0, dna_len, 5):
        new_chromosome = chromosome[i:i+5]
        new_dna_len = len(new_chromosome)
        for j in range(new_dna_len):
            gene = new_dna_len - j - 1
            code[index] += new_chromosome[gene] * (2**j)
        index += 1
    
    return code

```

---

### 4) Initial Population

```python
def initial_population(pop_num, dna_length):
    population = []
    for i in range(pop_num):
        chromosome = np.ones(dna_length)     
        chromosome[:int(0.3 * dna_length)] = 0             
        np.random.shuffle(chromosome)
        population.append(chromosome)
    # print(population)
    return population

```

---

### 5) Fitness Function

```python
def fitness(data, pop):
    train_data, train_label, test_data, test_label = data

    scores = []
    model = XGBClassifier()
    for chromosome in pop:
        weight = decode(chromosome)
        
        #* 特徵加權
        for i in range(len(weight)):
            train_data[train_data.columns[i]] = train_data.iloc[:, i] * (weight[i] / 32)
            
        model.fit(train_data, train_label) 
        predictions = model.predict(test_data)
        test_score = metrics.accuracy_score(test_label, predictions)
        scores.append(test_score)

    scores, pop = np.array(scores), np.array(pop)
    inds = np.argsort(scores)
    
    return list(pop[inds, :][::-1]), list(scores[inds][::-1])  #my_list[start(開始的index):end(結束的index):sep(間隔)]

```

---

### 6) Selection

```python
def selection(data, pop):
    ran_F = random.choices(pop, k=2)
    ran_M = random.choices(pop, k=2)
    
    ran_F, sort_F = fitness(data, ran_F)
    ran_M, sort_M = fitness(data, ran_M)

    father = ran_F[0]
    mother = ran_M[0]

    return father, mother

```

---

### 7) Crossover

```python
def cross(data, pop, cross_rate=0.7):
    pop_size, dna_len = np.array(pop).shape
    new_pop = []
    while len(new_pop) != pop_size:
        father, mother = selection(data, pop)

        if np.random.rand() < cross_rate:
            cross_point = np.random.randint(low=0, high=dna_len)
            cut_F = father[cross_point:].copy()
            cut_M = mother[cross_point:].copy()
            father[cross_point:] = cut_M
            mother[cross_point:] = cut_F
            
        new_pop.extend([father, mother])
        
    return new_pop

```

---

### 8) Mutation

```python
def mutation(cross_pop, mutation_rate=0.005):
    pop_size, dna_len = np.array(cross_pop).shape
    new_pop = []
    for i in range(pop_size):
        if np.random.rand() < mutation_rate:
            mutation_point = np.random.randint(low=0, high=dna_len)
            if cross_pop[i][mutation_point] == 0:
                cross_pop[i][mutation_point] = 1
            else:
                cross_pop[i][mutation_point] = 0
                
        new_pop.append(cross_pop[i])

    return new_pop

```

---

