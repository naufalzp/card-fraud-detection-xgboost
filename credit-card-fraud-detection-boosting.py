# Card Fraud Detection Boosting</center>

#importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('./'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#importing dataset
df = pd.read_csv("DATASET/creditcard.csv")
df.head(5)

# Data processing and undersampling

# "Time" tidak diperlukan untuk klasifikasi sehingga saya hanya menghapus feature dari dataset :

df = df.drop("Time", axis=1)

# Kita perlu menstandarisasi fitur 'Amount' sebelum modeling.
# Untuk itu, kita menggunakan fungsi StandardScaler dari sklearn. Kemudian, kita hanya perlu menghapus feature lama :

from sklearn import preprocessing
scaler = preprocessing.StandardScaler()

#standarisasi Amount
df['std_Amount'] = scaler.fit_transform(df['Amount'].values.reshape (-1,1))

#menghapus kolom Amount
df = df.drop("Amount", axis=1)

sns.countplot(x="Class", data=df)

# Dataset ini sangat tidak seimbang,
# Ini masalah besar karena classifier akan selalu memprediksi class yang paling umum tanpa menganalisis fitur-fitur, dan akan tampak memiliki akurasi tinggi, meski salah. Untuk mengatasi ini, kita akan menggunakan undersampling acak.
# 
# Undersampling acak melibatkan pemilihan contoh secara acak dari class mayoritas dan menghapusnya dari dataset pelatihan. Meskipun sederhana dan efektif, teknik ini bisa menghapus informasi yang penting untuk menentukan "boundary" antara class. Ini berarti mungkin, atau bahkan kemungkinan besar, informasi yang berguna akan dihapus.
# 
# <center>Cara kerja undersampling :</center>
# <center><img src= "https://miro.medium.com/max/335/1*YH_vPYQEDIW0JoUYMeLz_A.png">
# 
# Untuk undersampling, kita bisa menggunakan paket imblearn dengan fungsi RandomUnderSampler!

from imblearn.under_sampling import RandomUnderSampler 

undersample = RandomUnderSampler(sampling_strategy=0.5)

cols = df.columns.tolist()
cols = [c for c in cols if c not in ["Class"]]
target = "Class"

#define X and Y
X = df[cols]
Y = df[target]

#undersample
X_under, Y_under = undersample.fit_resample(X, Y)

test = pd.DataFrame(Y_under, columns = ['Class'])

#visualisasi hasil undersampling
fig, axs = plt.subplots(ncols=2, figsize=(13,4.5))
sns.countplot(x="Class", data=df, ax=axs[0])
sns.countplot(x="Class", data=test, ax=axs[1])

fig.suptitle("Class repartition before and after undersampling")
a1=fig.axes[0]
a1.set_title("Before")
a2=fig.axes[1]
a2.set_title("After")

# Setelah Seimbang, langkah terakhir sebelum pemodelan adalah membagi data menjadi sampel train dan test. Set test akan terdiri dari 20% data.
# 
# Kita akan menggunakan dataset pelatihan untuk melatih model kita dan kemudian mengevaluasinya pada set test:
# 
# <center><img src="https://data-flair.training/blogs/wp-content/uploads/sites/2/2018/08/1-16.png"></center>
# <center>Untuk membagi data, kita bisa menggunakan fungsi train_test_split dari sklearn</center>

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_under, Y_under, test_size=0.2, random_state=1)

# Modeling Ensemble learning : Boosting (XGBoost)

#importing packages for modeling
from xgboost import XGBClassifier

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve

# ### <center> Cara Kerja XGBoost :</center>
# 
# ![](https://d1rwhvwstyk9gu.cloudfront.net/2020/02/XG-Boost-FINAL-01.png)
# 
# Metode ensemble sequential, juga dikenal sebagai "boosting", membuat serangkaian model yang mencoba memperbaiki kesalahan dari model sebelumnya dalam urutan tersebut. Model pertama dibangun pada data pelatihan, model kedua memperbaiki model pertama, model ketiga memperbaiki model kedua, dan seterusnya.

# Training model

model = XGBClassifier(random_state=2)
xgb = model.fit(X_train, y_train)

# Predict

y_pred_xgb = model.predict(X_test) 

# Score

print("Accuracy XGB:",metrics.accuracy_score(y_test, y_pred_xgb))
print("Precision XGB:",metrics.precision_score(y_test, y_pred_xgb))
print("Recall XGB:",metrics.recall_score(y_test, y_pred_xgb))
print("F1 Score XGB:",metrics.f1_score(y_test, y_pred_xgb))

# Confusion matrix
# <!-- pengertian singkat -->
# Confusion matrix adalah tabel yang digunakan untuk mengevaluasi kinerja sebuah model klasifikasi. Tabel ini menunjukkan perbandingan antara prediksi model dengan nilai aktual (ground truth). Dalam matriks ini, ada empat elemen penting:
# 
# - <strong>True Positive (TP)</strong>: Jumlah kasus positif yang diprediksi benar oleh model. Misalnya, dalam kasus deteksi penipuan kartu kredit, ini adalah jumlah transaksi penipuan yang benar-benar terdeteksi sebagai penipuan.
# 
# - <strong>True Negative (TN)</strong>: Jumlah kasus negatif yang diprediksi benar oleh model. Ini adalah jumlah transaksi tidak penipuan yang benar-benar terdeteksi sebagai tidak penipuan.
# 
# - <strong>False Positive (FP)</strong>: Jumlah kasus negatif yang diprediksi salah sebagai positif oleh model. Ini adalah jumlah transaksi tidak penipuan yang salah terdeteksi sebagai penipuan (false alarm).
# 
# - <strong>False Negative (FN)</strong>: Jumlah kasus positif yang diprediksi salah sebagai negatif oleh model. Ini adalah jumlah transaksi penipuan yang salah terdeteksi sebagai tidak penipuan (missed fraud).

matrix_xgb = confusion_matrix(y_test, y_pred_xgb)
cm_xgb = pd.DataFrame(matrix_xgb, index=['not_fraud', 'fraud'], columns=['not_fraud', 'fraud'])

sns.heatmap(cm_xgb, annot=True, cbar=None, cmap="Blues", fmt = 'g')
plt.title("Confusion Matrix XGBoost"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()

y_pred_xgb_proba = model.predict_proba(X_test)[::,1]
fpr_xgb, tpr_xgb, _ = metrics.roc_curve(y_test,  y_pred_xgb_proba)
auc_xgb = metrics.roc_auc_score(y_test, y_pred_xgb_proba)
print("AUC XGBoost :", auc_xgb)

#ROC
plt.plot(fpr_xgb,tpr_xgb,label="XGBoost, auc={:.3f})".format(auc_xgb))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('XGBoost ROC curve')
plt.legend(loc=4)
plt.show()

xgb_precision, xgb_recall, _ = precision_recall_curve(y_test, y_pred_xgb_proba)
no_skill = len(y_test[y_test==1]) / len(y_test)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='black', label='No Skill')
plt.plot(xgb_recall, xgb_precision, color='orange', label='XGB')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend()
plt.show()

# Conclusion

print("Accuracy XGB:",metrics.accuracy_score(y_test, y_pred_xgb))
print("Precision XGB:",metrics.precision_score(y_test, y_pred_xgb))
print("Recall XGB:",metrics.recall_score(y_test, y_pred_xgb))
print("F1 Score XGB:",metrics.f1_score(y_test, y_pred_xgb))
print("AUC XGBoost :", auc_xgb)


