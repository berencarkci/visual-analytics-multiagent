# Değerlendirme

Ölçüm dosyaları `evaluation/`, çıktılar `evaluation/results/`. Amaç iki iddiayı
kanıta bağlamak: multi-agent tek-ajana göre daha doğru mu, ve tercih
optimizasyonu (DPO) SFT'nin ötesine taşıyor mu.

## Benchmark

`evaluation/benchmark.json` — **60 soru**, üç veri seti üzerinde elle yazıldı.

| eksen | dağılım |
|---|---|
| tip (7) | comparison 10 · relationship 11 · trend 9 · distribution 9 · filter_aggregation 9 · anomaly 7 · composition 5 |
| veri seti | retail 22 · energy hourly 22 · mall 16 |
| split | **dev 38 · test 22** |

Her soru `{id, dataset, question, type, specificity, ground_truth, split}` taşır.
`ground_truth` insan tarafından yazılmış beklenen plan/grafiktir.

## Dev / test bölünmesi

`evaluation/make_split.py` — katmanlı (stratified: tip × veri seti), `seed=42`,
`test_ratio=0.30`. Her katman test'te en az bir soruyla temsil edilir.

**Test split donuktur.** Bir kez koşulur, sonra dokunulmaz; `make_split.py`
mevcut bir split'i üzerine yazmayı reddeder. Tüm iterasyon (veri genişletme,
hiperparametre, hata taraması) **dev** üzerinde yapılır — test yalnızca final
karşılaştırma içindir. Eğitim verisinin benchmark'a sızmadığı
`evaluation/check_contamination.py` ile her veri üretiminde denetlenir (birebir +
bulanık @0.85, 60 soruya karşı).

## İki eksen

`evaluation/compare_arms.py` kolları iki eksende koşturur:

- **Eğitim ekseni:** base → SFT-v2 → SFT-v3 → DPO. "Daha iyi ağırlık" etkisini
  ölçer.
- **Mimari ekseni:** tek-çağrı (uzun prompt) → multi-agent. "Daha iyi sistem"
  etkisini ölçer. `sft_v3_multi_noeval` kolu Evaluation/retry katmanını kapatarak
  onun katkısını ayırır.

DPO kolları SFT-**v2** tabanlı olduğu için v2 ile karşılaştırılır, v3 ile değil —
aksi halde "tercih optimizasyonu" ile "daha fazla SFT verisi" karışırdı.

## Metrikler

Kol başına: `schema_valid`, `columns_exist`, `chart_fits_type`,
`intent_correct`, `insight_grounded`, `insight_invented`, `eval_passed`,
`chain_stopped` (n), `retry_helped` (n), `seconds/question`. Çıktı
`results/arm_comparison.json` (özetler) ve `results/comparison_cache.json` (her
kolun her soruya cevabı).

Ölçüm notları (ayrıntı [training.md](training.md)): uydurma-sayı metriği soru/şema/
filtreden türeyen sayıları eler; `has_transform` soru tipine koşulludur; tek seed
+ greedy decoding olduğu için birkaç puanlık farklar tek soru demektir.

## Donuk test sonuçları

`evaluation/results/test_split_results.md` (22 soru, **bir kez**, v4'ten önce).
Held-out'ta niyet doğruluğu monoton yükseldi:

| | base | SFT-v2 | SFT-v3 |
|---|---|---|---|
| niyet doğruluğu | 18/22 | 20/22 | **21/22** |
| zincir durması | 3 | 0 | 0 |
| reviewer geçti % | 86.4 | 100 | 100 |

v4/v5 bu ölçümden sonra geldiği için ayrı **problarla** değerlendirildi (aşağı).

## Problar (benchmark'ın göremediği kapsam)

Benchmark, genişletmeden önce yazıldığı için yeni mekanizmaları ve alan-dışı
genellemeyi sınamaz. Bu boşlukları hedefli problar kapatır:

| prob | dosya | ne | sonuç |
|---|---|---|---|
| yetenek | `capability_probe.json` | türetilmiş ölçü / sıralama yönü (v3) | 8/8 |
| sağlamlık | `robustness_probe.json` | kenar durumlar, tuzak sorular (v4) | 22/25 |
| alan-dışı | `cross_domain_probe.json` | eğitimde görülmeyen şemalar (v5) | 16/16 |
| plan kalitesi | `plan_accuracy.py` / `plan_accuracy.json` | headline benchmark'ın düz bıraktığı plan ekseni | — |

Ek olarak `scan_dev_failures.py` dev split'i tam zincirden geçirip gerçek
hataları toplar (DPO çiftlerinin gözleme dayalı kısmının kaynağı).

## Rubrik (tercih puanlaması)

`evaluation/rubric.py` — altı boyutlu, format-farkındalıklı (0/1/2): schema_validity,
column_selection, transform_correctness, chart_appropriateness, groundedness,
clarity, intent_correctness. Sert kapılar (olmayan kolon, izinsiz grafik, ters
sıralama) doğrudan 0. `calibrate_rubric.py`: projede gözlenmiş 10 hata modu,
10/10 hakem yargısıyla uyumlu — rubrik hakemle çelişirse rubrik düzeltilir.

## Tekrar üretilebilirlik

Tüm koşumlar `seed=42`, çıkarım `temperature=0.0` (greedy). Bir ölçüm "komut +
adaptör id + kod commit'i" ile tam tanımlıdır; kod sonradan değiştiği için donuk
test sonuçları alındığı commit'e bağlıdır (bkz. `test_split_results.md` başlığı).
