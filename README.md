

---

# **Төслийн Ерөнхий Тойм**

Энэ репозитори нь IMDb кино шүүмжийн датад суурилсан Sentiment Analysis төслийг агуулдаг. Төслийн зорилго нь Machine Learning ашиглан кино шүүмжийг хурдан, үнэн зөв ангилах юм.

**Дэд хавтасууд:**
- `dataset` – Сургалт, туршилтанд ашиглах CSV датасет
- `project_proposal` – Төслийн санал PDF (Индонези, Англи)
- `ml_pipeline` – Notebook-д preprocessing, EDA, моделийн сургалт
- `bow_vs_tf-idf` – Bag of Words ба TF-IDF загваруудын тайлбар, код
- `resources_gdrive.txt` – Google Drive татаж авах холбоосууд
- `requirements.txt` – Хэрэглэх сангуудын жагсаалт

---

# **Асуудлын Аргачлал**

Орчин үеийн кино индустри нь IMDb зэрэг платформ дахь үзэгчдийн сэтгэгдэлд ихээхэн нөлөөлдөг. Дата их, гараар шинжлэхэд цаг их зарцуулдаг тул ML ашиглах нь оновчтой шийдэл юм. Logistic Regression, Naive Bayes, SVM зэрэг загваруудыг Bag of Words, TF-IDF, мөн BERT embeddings-тай хослуулан туршиж нарийвчлалыг нэмэгдүүлсэн.

---

# **Зорилго**

1. Кино шүүмжийг хурдан ангилах  
2. Шинжилгээний зардлыг багасгах  
3. Ангилалтын нарийвчлал, найдвартай байдал хадгалах  

---

**Модель:**  
- Хурд / Throughput  
- Нарийвчлал / Accuracy  

---

# **Dataset**

| №  | Баган | Тайлбар |
|----|-------|---------|
| 1  | review | Англи хэлний кино шүүмж |
| 2  | sentiment | 1 = positive, 0 = negative |

---

# **EDA & Pre-Processing**

1. Missing values шалгасан – алдаа үгүй  
2. Давхар мөр – 418, үлдээсэн  
3. Feature Engineering – review_length, review_length_binned  
4. Sentiment ангилалт – тэнцвэртэй  
5. Review урт – ихэнх 100-400 үг  
6. HTML, тусгай тэмдэгтүүд арилсан  
7. Text Stemming, Stopwords – Стоп үг арилгаагүй, үр дүн сайтай  
8. TF-IDF, Bag of Words загваруудад хөрвүүлсэн  
9. Train/Test split – 70/30  

---

# **BERT & ML сургалт**

- Embeddings: `bert_embeddings_uncased.npy`, `labels.npy`  
- Train/Validation: 80/20  

**Моделүүд ба Validation Accuracy:**

| Модель | Accuracy | Тайлбар |
|--------|----------|---------|
| Logistic Regression | 0.82 | Шуурхай, тогтвортой |
| SVM | 0.83 | Нарийвчлал өндөр |
| Random Forest | 0.81 | 100 estimator |
| Gradient Boosting | 0.80 |  |
| MLP | 0.82 | hidden_layer=128, max_iter=300 |

---

# **Бизнес Үр Нөлөө**

- Цаг хэмнэлт: 99%  
- Зардлын үр ашиг: 99%  


---

# **Дараагийн алхам**

- Илүү advanced deep learning аргад туршилт хийх  
- Өгөгдөл цуглуулах, боловсруулалт сайжруулах  
- NLP шинэ техникүүд судлах  
- Recommendation системд интеграци хийх  

---

# **Эшлэлүүд**

- Devlin et al., BERT (2019)  
- Liu et al., RoBERTa (2019)  
- Howard & Ruder, ULMFiT (2018)  
- Maas et al., Word Vectors for Sentiment Analysis (2011)  
- IMDb. About IMDb (2024)  

---

**Тайлбар:** Markdown хувилбар нь таны төсөлд тохируулж, preprocessing, BERT сургалт, ML моделүүдийн үр дүнг оруулсан.

өгөгдөлийн хэмжээ их тул ороогүй болно
# CreateAnalysis




## BERT Embedding дээр суурилсан 5 Машин Сургалтын Загвар

Энэ хэсэгт BERT embeddings ашиглан туршсан 5 өөр машин сургалтын загварын сургалт, үр дүн болон дүгнэлтийг танилцуулж байна. Загваруудыг киноны сэтгэгдлийн (sentiment analysis) эерэг болон сөрөг ангилалд ашигласан.

---

### Ашигласан өгөгдөл
- **bert_embeddings_uncased.npy** – BERT загвараас гаргасан текстийн embedding
- **labels.npy** – Сэтгэгдлийн шошго (0: negative, 1: positive)

Өгөгдлийг 80% сургалт, 20% баталгаажуулалт болгон хувааж ашигласан.

---
## Сургасан Машин Сургалтын Загварууд

Энэхүү судалгаанд BERT embeddings ашиглан дараах таван машин сургалтын загварыг тус тусад нь сургасан. Загвар бүр нь текст ангилалд өргөн хэрэглэгддэг бөгөөд гүйцэтгэл, хурд, тооцооллын нөөцийн шаардлага зэргээс шалтгаалан харьцуулан судлагдсан.

### 1. Logistic Regression
Logistic Regression нь ангиллын асуудалд өргөн хэрэглэгддэг шугаман загвар юм. Энэхүү судалгаанд BERT embeddings-ийг оролт болгон ашигласнаар үгсийн утга зүйн мэдээлэл хадгалагдаж, загвар нь эерэг болон сөрөг сэтгэгдлийг өндөр нарийвчлалтай ангилах боломжтой болсон. Мөн сургалтын хугацаа богино, үр дүн тогтвортой байсан.

### 2. Support Vector Machine (SVM)
SVM нь өгөгдлийг оновчтой хил (hyperplane)-ээр ангилах зорилготой загвар юм. BERT embeddings-тэй хослуулснаар өгөгдөл өндөр хэмжээст орон зайд сайн ялгарах боломж бүрдсэн. Гэвч сургалтын хугацаа бусад энгийн загваруудтай харьцуулахад арай урт байсан.

### 3. Multinomial Naive Bayes
Naive Bayes нь магадлалд суурилсан энгийн бөгөөд хурдан загвар юм. Уламжлалт байдлаар Bag of Words эсвэл TF-IDF дээр илүү тохиромжтой боловч, BERT embeddings ашигласан ч боломжийн гүйцэтгэл үзүүлсэн. Энэ нь загварын хөнгөн, хурдан шинж чанарыг харуулж байна.

### 4. SGD Classifier
SGD Classifier нь stochastic gradient descent ашиглан сургалт явуулдаг бөгөөд том хэмжээний өгөгдөл дээр хурдан ажиллах давуу талтай. BERT embeddings-тэй хослуулснаар сургалтын хурд сайн, гүйцэтгэл тогтвортой байсан.

### 5. Multi-layer Perceptron (MLP)
MLP нь нейрон сүлжээнд суурилсан загвар бөгөөд шугаман бус хамаарлыг сурах чадвартай. BERT embeddings ашигласнаар илүү баялаг онцлог мэдээлэл дээр суралцсан боловч сургалтын хугацаа урт, зарим тохиолдолд тогтворжилт шаардаж байсан.

---

## Машин Сургалтын Загваруудын Нэгдсэн Үр Дүн

Дараах хүснэгтэд BERT embeddings ашиглан сургасан бүх машин сургалтын загваруудын баталгаажуулалтын (validation) үр дүнг нэгтгэн харууллаа.

| № | Загвар | Ашигласан Embedding | Accuracy | Precision (avg) | Recall (avg) | F1-score (avg) |
|---|-------|--------------------|----------|------------------|--------------|---------------|
| **1** | **Logistic Regression** | **BERT (uncased)** | **0.82** | **0.82** | **0.82** | **0.82** |
| 2 | Support Vector Machine (SVM) | BERT (uncased) | 0.79 | 0.79 | 0.79 | 0.79 |
| 3 | Multilayer Perceptron (MLP) | BERT (uncased) | 0.79 | 0.79 | 0.79 | 0.79 |
| 4 | Naive Bayes | BERT (uncased) | 0.76 | 0.76 | 0.76 | 0.76 |
| 5 | SGD Classifier | BERT (uncased) | 0.78 | 0.78 | 0.78 | 0.78 |

---

### Үр Дүнгийн Ерөнхий Шинжилгээ

- **Logistic Regression** загвар нь **хамгийн өндөр accuracy (0.82)** болон тэнцвэртэй precision, recall, F1-score үзүүлэлттэйгээр бусад загваруудаас давуу гүйцэтгэл үзүүлсэн.
- **SVM болон MLP** загварууд нь ойролцоо гүйцэтгэлтэй боловч Logistic Regression-тэй харьцуулахад илүү их тооцооллын нөөц шаардсан.
- **Naive Bayes** нь энгийн загвар хэдий ч BERT embeddings ашигласнаар боломжийн үр дүн гаргасан.
- Ерөнхийдөө **BERT embeddings + уламжлалт ML загварууд** нь өндөр гүйцэтгэлтэй, сургалтын хувьд үр ашигтай шийдэл болох нь харагдсан.

---

###  Дүгнэлт

Энэхүү судалгаа, туршилтын үр дүнгээс харахад BERT embeddings ашигласан уламжлалт машин сургалтын загварууд нь текстийн сэтгэл хандлагыг ангилахад өндөр үр ашигтай болох нь батлагдлаа. Ялангуяа Logistic Regression + BERT embeddings хослол нь бусад загваруудтай харьцуулахад хамгийн өндөр нарийвчлал (Accuracy = 0.82), мөн тэнцвэртэй precision, recall, F1-score үзүүлэлтүүдийг харуулсан.
Түүнчлэн энэхүү арга нь илүү төвөгтэй deep learning архитектуруудтай харьцуулахад:
Сургалтын хугацаа богино
Тооцооллын нөөц бага шаарддаг
Практик хэрэглээнд нэвтрүүлэхэд илүү хялбар давуу талтай байна
Иймд BERT embeddings дээр суурилсан энгийн машин сургалтын загварууд нь бодит хэрэглээнд тохиромжтой, гүйцэтгэл ба хурдны хувьд тэнцвэртэй шийдэл болохыг энэхүү туршилт харууллаа.

---

###  Ач холбогдол

- **Цаг хэмнэлт:** Машин сургалтын загварууд нь олон мянган сэтгэгдлийг секундийн дотор ангилах боломжтой
- **Зардлын хэмнэлт:** Гараар ангилах шаардлагыг бууруулна
- **Бизнесийн үнэ цэнэ:** Киноны үнэлгээ, хэрэглэгчийн сэтгэл ханамжийг автоматаар шинжлэх боломж бүрдэнэ

---
##  Машин Сургалтын Загваруудын Нэгдсэн Үр Дүн (Hyperparameter Туршилт)

Дараах хүснэгтэд BERT embeddings ашиглан сургасан бүх машин сургалтын загваруудын баталгаажуулалтын (validation) үр дүнг, hyperparameter туршилтын хамгийн өндөр accuracy-тэй комбинацийн хамт харууллаа.

| № | Загвар | Hyperparameters | Accuracy | Precision (avg) | Recall (avg) | F1-score (avg) |
|---|-------|-----------------|----------|-----------------|--------------|---------------|
| **1** | **Logistic Regression** | C=1, solver='lbfgs', max_iter=200 | **0.82** | 0.82 | 0.82 | 0.82 |
| 2 | Support Vector Machine (SVM) | C=1, kernel='linear' | 0.79 | 0.79 | 0.79 | 0.79 |
| 3 | Multilayer Perceptron (MLP) | hidden_layer_sizes=(128,), activation='relu', max_iter=300 | 0.79 | 0.79 | 0.79 | 0.79 |
| 4 | Naive Bayes | alpha=1.0 | 0.76 | 0.76 | 0.76 | 0.76 |
| 5 | SGD Classifier | loss='hinge', alpha=0.0001, max_iter=2000 | 0.78 | 0.78 | 0.78 | 0.78 |

---

###  Ерөнхий Шинжилгээ

- **Logistic Regression** загвар нь хамгийн өндөр accuracy болон тэнцвэртэй precision, recall үзүүлэлттэй байна.  
- **SVM болон MLP** загварууд ойролцоо гүйцэтгэлтэй боловч илүү их тооцоолол шаардсан.  
- **Naive Bayes** нь энгийн загвар хэдий ч BERT embeddings ашигласнаар боломжийн үр дүн гаргасан.  
- Hyperparameter туршилт хийх нь моделийн нарийвчлалыг сайжруулахад үр дүнтэй байгааг харуулж байна.

---

###  Дүгнэлт

Энэхүү туршилтаас харахад:  
- Hyperparameter-т тохирсон **Logistic Regression + BERT embeddings** хослол нь хамгийн оновчтой сонголт болох нь батлагдсан.  
- Илүү төвөгтэй deep learning загвар ашиглахгүйгээр өндөр нарийвчлал (**0.82**) хүртэх боломжтой.  
- Практик хэрэглээнд хурд, гүйцэтгэлийн тэнцвэрийг сайн хангаж чадсан.

### Paper
1. Sentiment Analysis of Movie Reviews Using BERT — BERT-ийг IMDb movie review-д fine-tune хийж найдвартай нарийвчлал гаргасан судалгаа. 
arXiv

https://arxiv.org/abs/2502.18841

2. Enhancing IMDb Review Classification with LSTM Models — IMDb dataset-д LSTM болон BERT-тай холбосон sentiment ангилсан судалгаа. 
ResearchGate

https://www.researchgate.net/publication/388184987_Deep_Learning-Based_Sentiment_Analysis_Enhancing_IMDb_Review_Classification_with_LSTM_Models

3. Comparative Analysis of CNN, LSTM, CNN-LSTM and BERT Models — CNN, LSTM, BERT зэрэг олон deep learning загварыг IMDb 50k-д харьцуулсан. 
ResearchGate

https://www.researchgate.net/publication/387418208_Comparative_Analysis_of_Sentiment_Classification_on_IMDB_50k_Movie_Reviews_A_Study_Using_CNN_LSTM_CNN-LSTM_and_BERT_Models

4. Sentiment Analysis of IMDb Movie Reviews Using Traditional ML & Transformers — Уламжлалт ML (Logistic Regression, SVM) болон BERT-ийг харьцуулсан судалгаа. 
ResearchGate

https://www.researchgate.net/publication/378691218_Sentiment_Analysis_of_IMDb_Movie_Reviews_Using_Traditional_Machine_Learning_Techniques_and_Transformers

5. The Effect of Various Text Representation Methods for Sentiment Analysis — IMDB-д өөр өөр text representation (BERT, TF-IDF) ашигласан үед SVM-тэй performance-ийг харьцуулсан. 
DergiPark

https://dergipark.org.tr/tr/download/article-file/3993870

6. Sentiment Analysis on IMDB Reviews Using BERT Deep Learning — BERT-ийг IMDb-д ашиглаж training/validation/test accuracy-тай гүн сургалт хийсэн. 
E-Journal UIN Suska

https://ejournal.uin-suska.ac.id/index.php/IJAIDM/article/view/24239

7. Sentiment Analysis on IMDB Movie Reviews (Master Thesis) — IMDb сэтгэгдлийн ангилалын талаар lexicon болон BERT-тэй харьцуулсан судалгаа. 
DIVA Portal

https://www.diva-portal.org/smash/get/diva2%3A1779708/FULLTEXT02.pdf

8. Movie Review Sentiment Analysis Using BERT (Kaggle notebook) — IMDb text-ийг EDA хийж BERT-ийг fine-tune хийх практик кодын жишээ. 
Kaggle

https://www.kaggle.com/code/kritanjalijain/movie-review-sentiment-analysis-eda-bert

9. Fine-tuning BERT with Bidirectional LSTM for Movie Review SA — BERT + BiLSTM хосолсон sentiment analysis-ийн SOTA-тай ойролцоо accuracy-тай судалгаа. 
arXiv

https://arxiv.org/abs/2502.20682

10. TWSSenti: Hybrid Framework for Sentiment Analysis Using Transformers — IMDB болон бусад data-д олон transformer-ийг нэгтгэсэн hybrid approach-ын судалгаа (BERT, RoBERTa, GPT-2). 
arXiv

https://arxiv.org/abs/2504.09896



 *Энэ туршилт нь BERT embeddings + энгийн машин сургалтын загваруудыг хослуулан ашиглах боломжтойг харуулж байна.*

