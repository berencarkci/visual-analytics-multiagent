# Veri Kartı — Eğitim ve Tercih Verisi

## Özet

| | |
|---|---|
| Tek çağrı örneği | 671 (`data/sft_train.jsonl`) |
| Ajan formatlı örnek | 3375 (`data/sft_agents_train.jsonl`, 5 format) |
| Tercih çifti | 430 (`data/dpo/dpo_train.jsonl`) |
| Format | chat-JSONL (system / user / assistant) + meta |
| Üretici | `evaluation/make_sft_data.py`, `make_agent_sft_data.py`, `make_dpo_pairs.py` (seed=42, deterministik) |
| Kesişim kontrolü | `evaluation/check_contamination.py` — temiz (60 benchmark sorusuna karşı, birebir + bulanık @0.85) |

## Kaynak karışımı (tek çağrı seti)

| Kaynak | Adet | Açıklama |
|---|---|---|
| template | 634 | Soru, cevap şablonundan türetildi — hedef kurgu gereği doğru |
| handwritten | 27 | Muğlak / serbest ifadeli sorular, elle seçilmiş hedefler |
| failure_targeted | 10 | Gözlemlenen model hatalarının doğru cevapları (`failure_examples.py`, elle genişletilebilir) |

Sıralama dağılımı: `date_asc` 189, `value_desc` 181, `value_asc` 55,
`date_desc` 20, sıralamasız 226.
Türetilmiş ölçü içeren örnek: 71 (`days_between` 24, `ratio` 24, `diff` 23).

## Tasarım kuralları

1. **Tek çağrı insight'ları işaret eder, iddia etmez.** Hedef insight'lar hiçbir
   zaman şemadan hesaplanamayacak bir sayı içermez ("The chart shows which
   category leads..."). Gerekçe: girdide veri yokken sayılı hedeflerle eğitmek,
   modele kendinden emin sayı uydurmayı öğretirdi — baseline'ın tam da
   düzeltmeye çalıştığımız zaafı.
2. **Ajan formatlı Insight hedefleri sayı içerir.** Kural hiçbir zaman "sayı
   kullanma" değildi, "göremediğin sayıyı söyleme"ydi. Insight ajanı hesaplanmış
   istatistikleri gördüğü için sayı alıntılamak doğru davranıştır.
3. **Guardrail'ler kural metni olarak değil, örnek olarak öğretilir.** Hedeflerde
   6+ kategorili pie yoktur, kesikli x ilişkiler box'a, histogramlar groupby'sız
   gider. Model kuralı okumaz, hep böyle yapılmış görür.
4. **Kısa sistem prompt'u, few shot yok.** Format bilgisi ağırlıklara taşınmalı,
   baseline'ın 1260 token'lık dondurulmuş prompt'u ölçüm için el değmeden durur.
5. **Her hedef pydantic şemasından geçirilerek doğrulanır** (ChartRecommendation) —
   şema kayması üretim anında patlar, eğitime sızmaz.
6. **Ajan formatlı örnekler ajanların kendi prompt kurucularından üretilir.**
   `make_agent_sft_data.py`, `_build_plan_messages()` / `_build_viz_messages()` /
   `_build_insight_messages()` fonksiyonlarını import eder. Eğitim girdisi ile
   çıkarım girdisi birebir aynı kalır, prompt drift'i yapısal olarak imkânsızdır.

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
otomatik tekrar çalışır → temizse dosya yeniden yazılır → ajan formatlı set için
`make_agent_sft_data.py` koşulur.

## Gözlemden veriye: beş tur

Set beş kez, canlı koşum bulgularıyla genişletildi.

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

3. **Şema genişletmesi.** Canlı etiketleme oturumları dört sınır gösterdi (bkz.
   [backend.md](backend.md), Bölüm 6). Yeni mekanizmalar için bankalar eklendi:
   artan/azalan sıralama ve türetilmiş ölçüler. **Prompt'a kural yazmak tek
   başına yetmedi** — ilk denemede model yeni mekanizmaları hiç üretmedi, örnek
   bankaları eklendikten sonra üretmeye başladı. Set 381 → 471 örneğe çıktı (v2).

4. **Supervisor formatı ve tuzak bankaları (v3).** Ajan formatlı sete beşinci
   format olarak `supervisor` (niyet sınıflandırma) eklendi; `filter_aggregation`
   tuzak bankası ve `diff`/anomali örnekleri genişletildi. `diff` 2 → 23'e
   çıkınca yetenek testi 5/8 → 8/8 oldu.

5. **Kategorik dağılım, olumsuzlama, alan-dışı sağlamlık (v4–v5).** Kategorik
   dağılım, olumsuzlama ("... hariç") ve iç içe pay (nested share) bankaları ve
   istatistik sağlamlık düzeltmeleri eklendi. Tek çağrı seti 671, ajan formatlı
   set 3375 örneğe (671 × 5 format, + 20 intent-only supervisor sorusu) çıktı.

Turların ortak dersi: yapısal bir davranışın öğrenilmesi için ~20+ örnek
gerekiyor. `diff` ölçüsü v2'de 2 örnekle eğitilip yetenek testinde hiç
kullanılmamıştı; v3'te 23 örneğe çıkarılınca öğrenildi.

## Tercih verisi

DPO çiftleri `data/dpo/` altında. Üretim `evaluation/make_dpo_pairs.py`,
puanlama `evaluation/rubric.py`. Dağılımlar ve gerekçeler için
[training.md](training.md).

Promptlar yine ajanların kendi prompt kurucularından üretilir. Prompt
değiştiğinde çiftler geçersizleşir ve yeniden üretilir: şema genişletmesinden
sonra 412 çift atılıp 430 çift baştan üretildi.

Kesişim kontrolü DPO tarafında da çalışır — promptlarda benchmark sorusu
bulunursa üretim durur.

**Belirsiz çiftler eğitim dışı.** Rubriğin tek ham puanla ayırdığı çiftler
(`pairs_*_unclear.jsonl`) eğitime girmez: zayıf sıralama gürültülü eğitim
sinyali demek. Dosyalar repoda durur, karar geri alınabilir.

## Etiketleme verisi

Space'in Preference Labeling sekmesi canlı tercih etiketi toplar
(`labels/preference_labels.jsonl`, HF Dataset'e push edilebilir). Amaç
doğrulama: rubriğin otomatik etiketlemesi insan yargısıyla ne kadar uyuşuyor.

Hedef başlangıçta 150-300 çiftti. Rol "üretim"den "doğrulama"ya kaydığı ve tek
etiketleyici bulunduğu için 40-60'a indirildi: bu aralık uyum oranını %10-15
hata payıyla raporlamaya yetiyor ve ~45 dakika sürüyor.

Etiketleme ajan çıktısı (JSON) üzerinden yapılır, grafik üzerinden değil — DPO
çiftleri modelin ürettiği token dizisi olmak zorunda. İki önyargı kontrolü:
adaylar karıştırılarak gösterilir (konum önyargısı), rubriğin kararı seçimden
önce gizlenir (çıpalama önyargısı).

Etiketleme oturumlarının asıl çıktısı uyum oranı değil, yedi bilinen sınırın
keşfi oldu (bkz. [backend.md](backend.md), Bölüm 6). Mekanik ölçüm bunların
hiçbirini yakalamamıştı.

_Uyum oranı etiketleme tamamlanınca eklenecek._

## Veri kaynakları ve lisans

Sorular üç genel veri setinin şemaları üzerinden üretildi (Superstore satış,
Mall müşteri, UCI Appliances enerji; kaynak ve temizlik adımları
`data/README.md`'de). Soru metinleri ve hedefler bu proje içinde üretilmiştir.