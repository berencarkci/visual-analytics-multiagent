# Donuk test bölünmesi sonuçları (22 soru)

> Bu ölçüm **bir kez** koşuldu ve donuktur. Koşum, v4 eğitiminden ve v4 kod
> yamalarından (kategorik dağılım guardrail'i, `ALLOWED_CHARTS`'a `bar`,
> tek-değer agregasyonu, türetilmiş tarih filtreleri, metin kolonu toplama)
> **önce** alınmıştır. Sonuçlar `compare_arms.py --split test` çıktısıdır.
>
> Reproducibility için `docs/evaluation.md`'ye ölçümün alındığı commit hash'i
> yazılmalıdır: kod sonradan değiştiği için aynı komut bugün farklı sayılar
> üretir.

## Eğitim ekseni — tek çağrı

| metrik | base_single | sft_v2_single | sft_v3_single |
|---|---|---|---|
| şema geçerli % | 100.0 | 100.0 | 100.0 |
| kolonlar mevcut % | 100.0 | 100.0 | 100.0 |
| grafik-niyet uyumu % | 95.5 | 95.5 | 95.5 |
| uydurma sayı % | 0.0 | 0.0 | 0.0 |
| retry gerekti % | 0.0 | 0.0 | 0.0 |
| sn / soru | 5.5 | 4.6 | 4.6 |

Tek çağrıda üç model ayrışmıyor. Sebep: tek çağrı hiçbir şeyi çalıştırmıyor,
plan kâğıt üstünde geçerli görünüyor. Ayrışma multi-agent kollarında ortaya
çıkıyor, çünkü orada dönüşüm gerçekten uygulanıyor.

## Eğitim ekseni — multi-agent (SFT veri genişletmesi)

| metrik | base_multi | sft_v2_multi | sft_v3_multi |
|---|---|---|---|
| şema geçerli % | 86.4 | 100.0 | 100.0 |
| kolonlar mevcut % | 100.0 | 100.0 | 100.0 |
| grafik-niyet uyumu % | 94.7 | 95.5 | **100.0** |
| **niyet doğruluğu %** | **81.8** (18/22) | **90.9** (20/22) | **95.5** (21/22) |
| insight grounded % | 100.0 | 100.0 | 100.0 |
| reviewer geçti % | 86.4 | 100.0 | 100.0 |
| retry gerekti % | 0.0 | 0.0 | 0.0 |
| **zincir durması (n)** | **3** | **0** | **0** |
| retry yardım etti (n) | 0 | 0 | 0 |
| sn / soru | 6.4 | 5.7 | 5.8 |

Held-out'ta monoton artış: 18 → 20 → 21. Base model 3 soruda zinciri kırıyor,
SFT ikisini de sıfırlıyor. v3 yapısal metriklerde %100'e ulaşan tek kol.

## Eğitim ekseni — tercih optimizasyonu (hepsi SFT-v2 tabanlı)

| metrik | sft_v2_multi | dpo_all_multi | dpo_real_multi |
|---|---|---|---|
| şema geçerli % | 100.0 | 100.0 | 100.0 |
| kolonlar mevcut % | 100.0 | 100.0 | 100.0 |
| grafik-niyet uyumu % | 95.5 | 100.0 | 100.0 |
| niyet doğruluğu % | 90.9 (20/22) | 95.5 (21/22) | 90.9 (20/22) |
| insight grounded % | 100.0 | 100.0 | 100.0 |
| reviewer geçti % | 100.0 | 100.0 | 100.0 |
| zincir durması (n) | 0 | 0 | 0 |
| sn / soru | 5.7 | 6.1 | 6.1 |

DPO adaptörleri SFT-v2 üstüne eğitildiği için v2 ile karşılaştırılır, v3 ile
değil — aksi halde "tercih optimizasyonu" ile "daha fazla SFT verisi" karışır.

## Mimari ekseni

| metrik | base_single | sft_v3_single | sft_v3_multi_noeval | sft_v3_multi |
|---|---|---|---|---|
| şema geçerli % | 100.0 | 100.0 | 100.0 | 100.0 |
| kolonlar mevcut % | 100.0 | 100.0 | 100.0 | 100.0 |
| grafik-niyet uyumu % | 95.5 | 95.5 | 100.0 | 100.0 |
| niyet doğruluğu % | – | – | 95.5 | 95.5 |
| insight grounded % | – | – | – | 100.0 |
| uydurma sayı % | 0.0 | 0.0 | – | – |
| reviewer geçti % | – | – | – | 100.0 |
| retry gerekti % | 0.0 | 0.0 | 0.0 | 0.0 |
| zincir durması (n) | – | – | 0 | 0 |
| retry yardım etti (n) | – | – | – | 0 |
| sn / soru | 5.5 | 4.6 | 5.7 | 5.8 |

Reviewer test bölünmesinde hiç devreye girmedi (v3 hiç başarısız olmadı), yani
katkısı burada ölçülemiyor. Dev bölünmesinde base model üzerinde 1 soruyu
kurtardı (`retry_helped = 1`) — zayıf modelde mimarinin telafi edici etkisi
orada görünüyor.

---

## Dev + test birleşik niyet doğruluğu (60 soru)

| kol | dev (38) | test (22) | toplam |
|---|---|---|---|
| base_multi | 35 | 18 | **53/60** |
| sft_v2_multi | 35 | 20 | **55/60** |
| **sft_v3_multi** | 35 | 21 | **56/60** |
| dpo_all_multi | 34 | 21 | **55/60** |
| dpo_real_multi | 33 | 20 | **53/60** |

Dev'de üç SFT kolu 35/38'de eşitti ama **hata kümeleri farklıydı**; ayrışma
ancak test bölünmesinde göründü. DPO-all, kendi tabanı olan v2 ile toplamda
tam berabere (55/55): dev'de −1, test'te +1. Doğru ifade "DPO gerileme yarattı"
değil, **"DPO'nun etkisi her iki yönde de gürültü içinde; ölçülebilir bir
değişiklik üretmedi"**.

## Raporlanacak ana sonuç

Bu görevde ve bu ölçekte kazanç **hedefli SFT verisinden** geldi, tercih
optimizasyonundan gelmedi. DPO başarısız bir deneme değil, hangi aracın hangi
işe yaradığını gösteren gerekçelendirilmiş bir bulgudur: DPO var olan bir
davranışı cilalar, olmayan bir yeteneği kuramaz.