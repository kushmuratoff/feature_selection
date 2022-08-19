# Feature Selection metodlari
<b>Feature Selection</b> nima va u nima maqsadda ishlatiladi degan savolga bugun biz 12 ta metodni 3 ta kategoriyaga bo'lgan holda o'rganib chiqamiz. Biz ba'zi muammolarni yechishda model quramiz va model natijasiga bizning datadagi qaysi bir ustun(feature)i ko'proq tasir qilishini bilishimiz uchun biz datadagi ustunlar sonini kamaytirib faqatgaina natijaga tasir qilayotgan eng muhimlarini tanlab olishimiz kerak.

<br>
Feature Selectionning asosiy vazifalari
* Target value bilan orasidagi bog'liqlik maksimum bo'lgan featurelarni saqlab qoladi va qolganlarini o'chirib yuvoradi.
* Train va Test classifierdagi hisoblash murakkabligi va hisoblash vaqtini kamaytiradi va natijasi <i>cost-effective</i> model hosil qiladi [Overfitting](https://machinelearningmastery.com/overfitting-and-underfitting-with-machine-learning-algorithms/)ni oldini olgan holda learning algoritmni modelini yaxshilaydi.
 
 
 ## Feature Selectionning 3 ta turi
 Klassifikatsiya(Classifier) masalasiga qanday bog'langanligiga qarab feature selection metodi 3 ta kategoriyaga bo'linadi.
* Filter
* Wrapper
* Embedded

## Filter Metodlar
Klassifikatsiya qilinmasdan oldin learning algoritm [bias](https://www.bmc.com/blogs/bias-variance-machine-learning/#:~:text=What%20is%20bias%20in%20machine,assumptions%20in%20the%20ML%20process.)i bilan feature selection algoritimni biasi aralashib ketmasligini uchun feature selectionni tanlab oladi.<br> Ular asosan ranker sifatida qaraladi va featurelarini yaxshisidan yomoniga qarab tartiblaydi. 
<br>
Featurelaring ranki dataning <i>muhumlilik xususiyatiga(intrinsic properties)</i> bog'liq, misol uchun dispersiya, [consistency](https://en.wikipedia.org/wiki/Data_consistency), masofa va koorelatsiya.
<br>
Hozirgi paytda juda ko'p filter metodlari mavjud va yangilari ham regualarniy ishab chiqarilmoqda.
<br>
## Wrapper Metodlar
Wrapper metodi mechine learining algoritmlarini <i>qora quti (black box evaluator)</i> orqali eng yaxshi featurelaring qisim toplamini topishda ishlatiladi.
<br>
Amaliyotda bazi qidirish strategiyalari va algoritmlarni modellashtirishning kombinatsiyalari wrapper sifatida ishalatiladi.
<br>
Qachonki ko'p featureli katta dadaset bilan ishalganda u juda ko'p hisoblash resursini va vaqtini talab qiladi va wrapper metodlari orqali hisoblashini soddalashtirsak bo'ladi.
## Embedded Metodlar
Embedded metodlar filter va wrapper metodalar orasidagi ko'prik vazifasini bajarib beradi.
<br>
Statsitik kriteryalar orqali bazi featurelarni tanlab oladi va machine learning algoritmlarini orqali eng yaxshi kalssifikatsiyali natijalarni tanlab oladi.
<br>
Har bir itaratsiyada featurelar qisimtoplamini qayta sinflamasdan wrapper metodining hisoblanish murkkabligini kamaytiradi va modelni featurega bog'loqlik qilib qo'yadi.
<br>
Feature selection learning phase(model qurilish vaqti)da amalga oshiriladi yani u model fitting  va feature selectionni bir vaqting o'zida amalga oshiriladi.
<br>
Bitta minus tarafi tanlab olish klasslarga bog'liq

<br>
Kerakli kutubxonalarni qo'shamiz va hisoblash ishlarini boshlaymiz

```python
import pandas as pd
import numpy as np
```

```python
# Read in data into a dataframe 
data = pd.read_excel('Data1_Feature_selection.xlsx')
# Display top of dataframe
data.head()
X = data.drop(columns =['ID','BEq(cr)','C1_me','C2_bs','Set'])
y = data['BEq(cr)']
```

Datasetni ikkiga bo'lib oldik Features va Target qismlarga.
<br>
Bu funksiya bizga eng yaxshi ya'ni targetga eng ko'p tasir ko'rsatayotgan featurelarni ko'rsatadi.

``` python
def getFeature(mask):
    new_features = [] # The list of your K best features

    for bool, feature in zip(mask, X.columns):
        if bool:
            new_features.append(feature)
            
    return new_features

```
## Filter methods
### 1. Chi-square
Hosil bo'lgan feature klass qiyamtiga tegishli bo'lmasiligni taxmin qiladigan [divergensiya](https://en.wikipedia.org/wiki/Divergence_(statistics)#:~:text=In%20statistics%20and%20information%20geometry,another%20on%20a%20statistical%20manifold.) qiymatini hisoblaydigan χ² statistik testga asoslangan bir o'lchovli filter.
<br>
Boshqa bir o'chovli metodlarga o'xshab biz har bir feature bilan target orasidagi  χ²ni hisoblaymiz va ular orasidagi bog'liqlikni kuzatamiz.
<br>
Agar target qiymati ozod had bo'lsa yani featurelarga bog'liq bo'lmasa natija past qiymat chiqaradi va aksincha bog'liq bo'lsa featurelarning muhimligi oshadi.
<br>
χ² ning yuqori qiymati o'sha featureni targetga juda yaqinligini anglatadi.
<br>

```python
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
X_norm = MinMaxScaler().fit_transform(X)
chi_selector = SelectKBest(chi2, k=6)
chi_selector.fit(X_norm, y)
chi_features = getFeature(chi_selector.get_support())
chi_features
```

Biz target qiymatga juda ko'p tasir ko'rsatyotgan eng yaxshi 6 ta featureni tanlab oldik.
<br>
χ² yordamida featurelarni tanlab olish haqida ko'proq [ma'lumot.](https://towardsdatascience.com/chi-square-test-for-feature-selection-in-machine-learning-206b1f0b8223#:~:text=In%20feature%20selection%2C%20we%20aim,hypothesis%20of%20independence%20is%20incorrect.)

<br>

### 2. Mutual Information
Bu metod ham filter metodlaridan biri va uni [Information Gain](https://machinelearningmastery.com/information-gain-and-mutual-information/#:~:text=Information%20gain%20can%20also%20be,between%20the%20two%20random%20variables.) deb ham atashadi.
<br>
Bu metodning hisoblash tezligi juda yaxshi va formulasi oddiy shuning uchun u ko'p qo'llaniladigan bir o'lchovli feature selection metodlaridan biri hisoblanadi.
<br>
Bir vaqtining o'zida har bir featurening [entopiyasi](https://www.analyticsvidhya.com/blog/2020/11/entropy-a-key-concept-for-all-data-science-beginners/) pasayishini hisoblaydi.
<br>
Agar featurening hisoblangan <i>information gain</i>ning qiymati katta bo'lsa targetga shuncha bog'liq bo'ladi.
<br>
Bu metod muhim bo'lmagan featurelarni ishlatmaydi chunki featurelar bir o'lchovli yo'l bilan aniqlanadi.
<br>
Featurelarning kanteksini hisobga olmasadan ularni erkli(independently) deb baholash metodiga asoslangan va u <b>myopic</b> deb nomlanadi.

```python
from sklearn.feature_selection import mutual_info_classif
mi_selector = SelectKBest(mutual_info_classif, k=6)
mi_selector.fit(X, y)
mi_feature = getFeature(mi_selector.get_support())
mi_feature
```
Va natijada biz eng yaxshi 6 ta featureni <b> Mutual Information</b> metodi orqali topdik. Bu metod haqida [batafsil ma'lumot.](https://machinelearningmastery.com/information-gain-and-mutual-information/#:~:text=Information%20gain%20can%20also%20be,between%20the%20two%20random%20variables.) 

### 3. Anova F-value
ANOVA(Analysis of Variance) -statistik metod va u 2 va undan ortiq gruxlarning o'rtacha qiymatining bir biridan farqli ekanligini tekshiraidagan metod.Bu metod ham bir o'lchovli filter metodi hisoblanadi va  klaslar orasidagi individual featurelarning ajralib turishini dispersiyadan foydalanilgan holda hisoblaydi.
<br>

```python
#3 Anova F-value
from sklearn.feature_selection import f_classif
anov_selector = SelectKBest(f_classif, k=6)
anov_selector.fit(X, y)
anova_feature = getFeature(anov_selector.get_support())
anova_feature
```
Bu metod haqida [batafsil ma'lumot](https://towardsdatascience.com/anova-for-feature-selection-in-machine-learning-d9305e228476)

### 4. Variance Threshold
Variance Threshold feature selection uchun oddiy yechim hisoblanadi. Ishlash davomida har bir feature uchun dispersiya hisoblanadi va u berilgan qiymat(threshold)dan kichik bo'lsa o'sha featrure o'chiriladi. Bu metodda faqat dispersiya hisoblanadi yani featurelarning bir birga bo'lgan bog'liqligi va featurening target qiymatiga bog'liqligi hisoblanmaydi. Bu filter metodining minus tarafi hisoblandi.

```python
#4 Variance Threshold
from sklearn.feature_selection import VarianceThreshold
var_selector = VarianceThreshold(threshold=1)
var_selector.fit_transform(X)
var_feature = getFeature(var_selector.get_support())
len(var_feature)
```
Bu metod haqida [batafsil ma'lumot](https://www.analyticsvidhya.com/blog/2020/10/feature-selection-techniques-in-machine-learning/#:~:text=The%20variance%20threshold%20is%20a,same%20value%20in%20all%20samples.)

### 5. Fisher Score
Fisher Score supervised turdagi datalar bilan ishlaganda eng yaxshi metodlardan biri hisoblanadi va o'rtacha qiymat va dispersiyadan foydalangan holda featurelar rankini hisoblaydi. Featurelar bir hil qiymatli egezemplyarlar bilan bir hil klaslarda va turli hil qiymatli egzemplyarlardan bilan turli hil klaslarda eng yaxshi hisoblanadi. Boshqa bir o'lchovli metodlar singari bu metod ham featurelarni induvidual tarzda baholaydi va targetga tasiri past qiymatlarni natija sifatida ishlatmaydi.

```python
#5. Fisher Score
from skfeature.function.similarity_based import fisher_score
score = fisher_score.fisher_score(X.to_numpy(), y.to_numpy())
score
```
### 6. MultiSURF
MultiSURF metodi [relief](https://medium.com/@yashdagli98/feature-selection-using-relief-algorithms-with-python-example-3c2006e18f83) metodinging kengaytirilgan usuli hisoblandi va uning multi-classi ReliefF filter feature selection metodi orqali hisoblanadi. 
<br>
Orginal relief metodi namunalrni datasetdan tasodifiy tarzda oladi va bir hil va turli hil klaslarga eng yaqin bo'lgan qo'shnisiga joylashtiradi.
<br>
Har bir atribut uchun eng yaqin qo'shini atributi qiymati bilan namuna qilib olingan atributni qiymati bog'liqlilik natijasini yangilash uchun solishtiriladi.
<br>
Asosiy ma'no shundaki ya'ni foydali atribut har hil klaslar namunlarida bir biridan farq qilib turish kerak va bir hil klasslarda bir hil qiymatga ega bo'lishi kerak.
<br>
Boshqa relief metodlarga qaraganda MultiSURF eng bog'liqli feature selection qiymatlarini muammo turlarining doirasiga qarab hosil qiladi.

```python
#6. MultiSURF
from skrebate import MultiSURF
fs = MultiSURF(n_jobs=-1, n_features_to_select=X.columns)
fs.fit(X.values, y.values)
fs
```
## Wrapper Methods
### 1. Recursive Feature Elimination
Modelni iterativ tarzda train qiladigan va har safar muhimlilik ko'rsatkichi eng kam bo'lgan featureni o'chirib tashlaydi va algoritm vaznini kriteriya sifatida ishlatadigan keng tarqalgan metodlardan [biri](https://machinelearningmastery.com/rfe-feature-selection-in-python/) hisoblanadi.
<br>
Bu ko'p faktorli metod bo'lgani uchun bir nechta featurelarning bog'liqligini bir vaqtning o'zida tekshiradi.
<br>
Barcha featurelar tekshirilmagunicha har bir iteratsiyada bittadan feature ochiriladi va stekga joylashtiriladi.
<br>
Hisoblash effektivligini oshirish maqsadida har bir qadamda bittadan oriq featureni o'chirib tashlashimiz mumkin.
<br>
Bu metod hisoblash tezligi yaxshi emas.

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
rfe_selector = RFE(estimator=LogisticRegression(), 
                   n_features_to_select=6, step=1, verbose=-1)
rfe_selector.fit(X_norm, y)
rfe_feature = getFeature(rfe_selector.get_support())
rfe_feature
```
### 2. Permutation Importance
Permutation Importance - model natijasini kamayishini bitta featureni tasodifiy olish orqali aniqlanadi.
<br>
Bu pretsedura feature va target orasidagi aloqni buzadi ya'ni model baholanishi qiymati pasayishi modelni featurelarga bog'liqligini ko'rsatadi.
<br>
Bu jarayonda kuzatiloyotgan muhimlik p-value

```python
from eli5.sklearn import PermutationImportance
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
perm = PermutationImportance(LogisticRegression(solver='lbfgs', max_iter=10000), random_state=42, cv=10)
perm.fit(X, y)
perm_selector = SelectFromModel(perm,max_features=6).fit(X, y)
```

mehod haqida [batafsil](https://stackoverflow.com/questions/71417212/eli5-sklearn-permutationimportance-typeerror-check-cv-takes-from-0-to-2)

```python
perm_feature = getFeature(perm_selector.get_support())
perm_feature
```
### 3. SHAP
SHAP - bu har qanday machine learning modelini tushinitra oladigan yagona(унифицированный) yondoshuv. 
<br>
U lokal tushunchalar bilan o'yin nazaryasiga bog'langan, birqancha oldingi metodlarni birlashtirish va bo'lish mumkin bo'lgan hollarni qaytako'rsatish va taxminlarga asoslangan lokal aniq qo'shimcha feature usuli.
<br>
Songi yilllarda bu metod eng ko'p ishlatiladigan metodlardan hisobalnadi.

```python
import shap
import xgboost
model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
```

```python
import matplotlib.pylab as pl
xgboost.plot_importance(model)
pl.title("xgboost.plot_importance(model)")
pl.show()
```

### 4. Boruta
Bu algoritm Random Forestga asoslangan wrapper algoritimi hisoblandi.
<br>
U statistik test qiymati random probesdan kam bog'liqlik bo'lgan featurelarni itertativ tarzda o'chiradi.

```python
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=6)
boru_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=1)
boru_selector.fit(X.values, y.values)
accept = X.columns[boru_selector.support_].to_list()
```
link for more [info](https://towardsdatascience.com/feature-selection-with-borutapy-f0ea84c9366)

## Embedded Methods
### 1. Embedded Random Forest
[Bu](https://towardsdatascience.com/feature-selection-using-random-forest-26d7b747597f) embedded feature selection Random Forest algoritmdan foydalanadi. 
<br>
Random Forestda har bir daraxt barcha featurelarni yoku barcha kuzatuchilarni ko'rmasligi mumkin va bu overfitting bo'lishini kamaytiradi va de-koorelatsiya bo'lishini taminlaydi. 
<br>
Har bir nodeda data 2 ga bo'linadi. bir biriga o'xshash va bir biriga o'xshash bo'lmagan qismlarga.
<br>
Featurening muhimligi o'sha bo'laklarga qanchalik bog'liqligiga asoslanadi.

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), max_features=6)
embeded_rf_selector.fit(X, y)
embeded_rf_feature = getFeature(embeded_rf_selector.get_support())
embeded_rf_feature
```

### 2. Embedded LightGBM
Bu embedded feature selection LGB algoritmdan foydalanadi.


```python
from lightgbm import LGBMClassifier
from sklearn.feature_selection import SelectFromModel
lgbc=LGBMClassifier(n_estimators=500, learning_rate=0.05,
                    num_leaves=32, colsample_bytree=0.2,                                           
                    reg_alpha=3, reg_lambda=1, min_split_gain=0.01,    
                    min_child_weight=40)
embeded_lgb_selector = SelectFromModel(lgbc, max_features=6)
embeded_lgb_selector.fit(X, y)
embeded_lgb_feature = getFeature(embeded_lgb_selector.get_support())
embeded_lgb_feature
```


