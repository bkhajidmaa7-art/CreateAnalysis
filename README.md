<div style="text-align: center;">
    <img src="https://drive.google.com/uc?id=1g4t3dYxccZsuzx0xXx6saaRWvGCXGi6G" alt="Project Banner" width="900" height="200">
</div>

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
# CreateAnalysis
