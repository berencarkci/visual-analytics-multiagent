# Veri Kartı — SFT Eğitim Verisi (data/sft_train.jsonl)

## Özet

| | |
|---|---|
| Örnek sayısı | 323 |
| Format | chat-JSONL (system / user / assistant) + meta |
| Üretici | `evaluation/make_sft_data.py` (seed=42, deterministik) |
| Kesişim kontrolü | `evaluation/check_contamination.py` — temiz (60 benchmark sorusuna karşı, birebir + bulanık @0.85) |

## Kaynak karışımı

| Kaynak | Adet | Açıklama |
|---|---|---|
| template | 344 | Soru, cevap şablonundan türetildi — hedef kurgu gereği doğru |
| handwritten | 27 | Muğlak / serbest ifadeli sorular, elle seçilmiş hedefler |
| failure_targeted | 9 | Gözlemlenen model hatalarının doğru cevapları (`failure_examples.py`, elle genişletilebilir) |

Grafik tipi dağılımı: bar 144, line 121, histogram 44, scatter 33, pie 25, box 13.
Veri seti dağılımı: retail 240, energy 98, mall 42.

## Tasarım kuralları

1. **Insight'lar işaret eder, iddia etmez.** Hedef insight'lar hiçbir zaman şemadan
   hesaplanamayacak bir sayı içermez ("The chart shows which category leads...").
   Gerekçe: girdide veri yokken sayılı hedeflerle eğitmek, modele kendinden emin
   sayı uydurmayı öğretirdi — baseline'ın tam da düzeltmeye çalıştığımız zaafı.
2. **Guardrail'ler kural metni olarak değil, örnek olarak öğretilir.** Hedeflerde
   6+ kategorili pie yoktur, kesikli x ilişkiler box'a, histogramlar groupby'sız
   gider. Model kuralı okumaz, hep böyle yapılmış görür.
3. **Kısa sistem prompt'u, few shot yok.** Format bilgisi ağırlıklara taşınmalı,
   baseline'ın 1260 token'lık dondurulmuş prompt'u ölçüm için el değmeden durur.
   (SFT sonrası kısa vs uzun prompt farkı dev split'te ayrıca ölçülecek.)
4. **Her hedef pydantic şemasından geçirilerek doğrulanır** (ChartRecommendation) —
   şema kayması üretim anında patlar, eğitime sızmaz.

## Kesişim kontrolü süreci

İlk üretim koşumunda kontrol **25 çakışma** yakaladı (4'ü birebir): şablon
ifadeleri, aynı veri setlerine aynı niyetlerle yazılmış benchmark'la doğal
olarak örtüşüyordu. Tüm şablon ifadeleri benchmark kalıplarından uzaklaştırılarak
yeniden yazıldı, bir failure örneği (benchmark sorusunun kendisiydi) yeniden
ifade edildi. Nihai set temiz. Kontrol, test split sorularını programatik olarak
karşılaştırır, hiçbir insan okumaz, hiçbir model eğitilmez, mühür bozulmaz.

## Genişletme akışı

Yeni bir model hatası gözlemlendiğinde: `evaluation/failure_examples.py`'ye
kayıt ekle → `python evaluation/make_sft_data.py` koş → kesişim kontrolü
otomatik tekrar çalışır → temizse dosya yeniden yazılır.

## Gözlemden veriye: iki tur

Set iki kez, canlı koşum bulgularıyla genişletildi:

1. **Filtre kaybı.** SFT sonrası dev koşumunda "Technology kategorisinin 2018'deki
   aylık satışları" sorusu `filter=None` ile geldi, ama insight kısıtı yine de
   iddia ediyordu. Kök neden: setteki tüm filtre örnekleri kategorik groupby'lı
   bar grafiğiydi, filtre + zaman groupby kombinasyonu hiç yoktu. Filtreli zaman
   serisi ve iki koşullu filtre bankaları eklendi (filtreli örnek 9 → 32).
   Sonraki koşumda aynı soru doğru filtreyle geldi.
2. **Anomali → bar.** İki anomali sorusu `bar` döndü, oysa bar anomali için
   izinli listede bile yok. Kök neden: 349 örnekte yalnızca 6 anomali örneği
   vardı ve hepsi tek granülerlikteydi. Banka gün/hafta/ay granülerliklerine ve
   "Identify the ... whose X ran abnormally high" kalıbına genişletildi
   (anomali örneği 6 → 43). Sonraki koşumda grafik-niyet uyumsuzluğu sıfırlandı.

## Veri kaynakları ve lisans

Sorular üç genel veri setinin şemaları üzerinden üretildi (Superstore satış,
Mall müşteri, UCI Appliances enerji, kaynak ve temizlik adımları data/README.md'de).
Soru metinleri ve hedefler bu proje içinde üretilmiştir.