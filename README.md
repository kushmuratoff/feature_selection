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


