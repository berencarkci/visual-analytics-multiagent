# Eğitim Metodolojisi

Veri setinin içeriği için [data.md](data.md), sistemin çalışma mantığı için
[backend.md](backend.md).

## Model Seçimi

Qwen2.5-3B-Instruct seçildi. Baseline testlerinde grafik ve kolon
seçiminde ilk denemede geçerli JSON üretti, muğlak soruları da doğru çözdü
(ör. "split across segments" → pie + count_distinct). Bu yüzden daha küçük
varyantlara (1.5B) inme ihtiyacı görülmedi. Cihaz: yerelde MacBook (MPS),
final koşumlarda Colab/Kaggle (CUDA).

> Not: `HFClient` varsayılanı bir süre yanlışlıkla 1.5B'de kalmıştı ve Space ile
> tüm multi-agent koşumları 1.5B ile çalışıyordu (sonradan fark edilip
> düzeltildi). Bundan önce kataloglanan hata modları 1.5B'nin hatalarıdır.

## Neden eğitim

Baseline, dondurulmuş 1260 token'lık few-shot prompt kullanıyor. Hedef, format
bilgisini prompt'tan **ağırlıklara taşımak**: kısa prompt, daha az token, daha
hızlı çıkarım ve prompt mühendisliğine bağımlı olmayan davranış.

## SFT

İki aşamada yapıldı.

### Aşama 1 — tek-çağrı SFT

Baseline'ın işini öğretir: şema + soru → tam grafik önerisi JSON'u.

- **Veri:** `data/sft_train.jsonl`, 381 örnek (şablon 344 · elle yazılmış 27 ·
  hata-hedefli 10)
- **Sistem promptu:** `SFT_SYSTEM` (~250 token, few-shot yok). Dondurulmuş
  baseline promptu el değmeden durur — baseline ölçümü için gerekli.

### Aşama 2 — çok görevli ajan eğitimi

Aşama 1 adaptörü ajanlara takıldığında Insight ajanı bozuldu: istatistik
sözlüğünü JSON dökümü olarak geri kusmaya başladı. Model 381 örnek boyunca "tam
JSON üret" diye koşullanmıştı, Insight "bir cümle yaz" dediğinde eğitilmiş şekli
dayatıyordu.

Çözüm: her tek-çağrı örneğini ajanların **kendi prompt kurucularıyla** üç ajan
formatına ayrıştırmak.

| format | girdi | hedef |
|---|---|---|
| `single_call` | şema + soru | tam öneri JSON'u |
| `data_analyst` | şema + niyet + soru | `{target_columns, transform}` |
| `visualization` | soru + niyet + veri özeti + izinli liste | `{chart_type, reason}` |
| `insight` | soru + hesaplanmış istatistikler | `{insight}` |

`evaluation/make_agent_sft_data.py`, `_build_plan_messages()`,
`_build_viz_messages()` ve `_build_insight_messages()` fonksiyonlarını **import
ederek** çağırır. Yani eğitim girdisi ile çıkarım girdisi birebir aynıdır; prompt
değişirse veri otomatik takip eder. Prompt drift'i yapısal olarak imkânsızdır.

Visualization ve Insight örnekleri için dönüşüm **gerçek veride çalıştırılır**,
böylece veri özeti ve istatistikler sahicidir.

**Insight hedefleri sayı içerir** — tek-çağrı formatının tersine. Kural hiçbir
zaman "sayı kullanma" değildi, "göremediğin sayıyı söyleme"ydi. Insight ajanı
hesaplanmış istatistikleri gördüğü için sayı alıntılamak doğru davranıştır. Odak
başına üç ifade varyantı üretilir (ezber yerine biçim öğrenilsin). Doğrulama:
üretilen 381 Insight hedefinin **381'i de** sistemin kendi `verify_grounded`
fonksiyonundan geçiyor.

**Supervisor eğitime dahil değil.** Duman testinde 16 koşumun hepsinde doğru
niyeti buldu, hiç kural yedeğine düşmedi — öğretecek bir şey yok.

Çıktı: `data/sft_agents_train.jsonl`, 381 × 4 = **1524 örnek**, tek adaptör.

### Aşama 3 — şema genişletmesi ve v2 eğitimi

Canlı etiketleme oturumları dört sınır ortaya çıkardı (bkz.
[backend.md](backend.md), Bölüm 6): artan sıralama yok, filtre sözdizimi
tanımsız, türetilmiş metrik hesaplanamıyor, uydurma alanlar sessizce yutuluyor.
Dördü de promptu değiştirdi; prompt değişince eğitim verisi de değişir — üretici
prompt kurucularından import ediyor — yani yeniden eğitim zorunlu oldu.

**Kritik ders: prompt'a kural yazmak yetmiyor.** İlk denemede yeni sıralama
yönleri ve türetilmiş ölçüler yalnızca prompta eklendi ve model bunları hiç
üretmedi. Örnek bankaları eklendikten sonra üretmeye başladı. Anomali
deneyiminde de aynıydı: 6 örnekle hata, 43 örnekle düzelme.

| | v1 | v2 |
|---|---|---|
| tek çağrı örneği | 381 | 471 |
| ajan formatlı örnek | 1524 | 1884 |
| eval loss | 0.0351 | **0.0276** |
| süre (T4) | 94 dk | 126 dk |

Yeni mekanizmalar için örnek sayıları: `value_asc` 36, `date_desc` 20,
`days_between` 24, `ratio` 20, `diff` 2.

**Yetenek testi** (`evaluation/probe_new_capabilities.py`). Duman testi ve dev
split genişletmeden önce yazıldığı için yeni mekanizmaları hiç sınamıyor: içlerinde
türetilmiş ölçü ya da artan sıralama gerektiren tek soru yok. Probe o boşluğu
dolduruyor — 8 soru, her biri bir mekanizmayı zorunlu kılıyor, ve kontrol hem
doğru mekanizmayı hem motorda çalışabilirliği istiyor. Sonuç: **5/8**.

| mekanizma | eğitim örneği | probe |
|---|---|---|
| `days_between` | 24 | 2/2 |
| `date_desc` | 20 | 1/1 |
| `value_asc` | 36 | 1/2 |
| `ratio` | 20 | 1/2 |
| `diff` | **2** | **0/1** |

Örnek sayısı ile başarı arasındaki ilişki net: `diff` 2 örnekle eğitildi ve hiç
öğrenilmedi. Bu tür yapısal davranışlar ~20+ örnek istiyor.

Üç başarısızlık üç ayrı türde ve üçü de DPO tercih çiftine dönüştürüldü:
`agg` slotuna ifade koyma (şema ihlali), "lose money" için yanlış sıralama yönü
(semantik hata), `diff` mekanizmasını hiç kullanmama.

Regresyon yok: multi-agent duman testi 8/8 PASSED, dev split taramasında
Evaluation başarısızlığı 1 → 0.

### Aşama 4 — v3, v4, v5

v2'den sonra set üç tur daha genişledi; her tur bir gözlem bankasıyla adreslendi.
Sürüm başına adaptör HF'te ayrı tutulur (bkz. [Adaptörler](#adaptörler)), **canlı
Space/app v5'i kullanır** (`app/main.py`, `DEFAULT_ADAPTER`).

| sürüm | eklenen | ölçüm |
|---|---|---|
| **v3** | supervisor formatı (5. ajan formatı) + `filter_aggregation` tuzak bankası + `diff`/anomali genişletmesi | yetenek testi 5/8 → **8/8** (`capability_probe.json`); donuk test split'inde yapısal metriklerde %100'e ulaşan tek kol, niyet 21/22 |
| **v4** | kategorik dağılım + olumsuzlama + nested share bankaları + istatistik sağlamlık düzeltmeleri | sağlamlık probu **22/25** (`robustness_probe.json`) |
| **v5** | alan-dışı genelleme turu (eğitimde görülmeyen şemalar) | alan-dışı prob **16/16** (`cross_domain_probe.json`) |

Donuk test split ölçümü (`evaluation/results/test_split_results.md`, 22 soru) base /
v2 / v3'ü karşılaştırır ve **v4'ten önce** alınmıştır — v4/v5 ayrı problarla
(sağlamlık, alan-dışı) değerlendirildi. v3 held-out'ta niyet doğruluğunu
monoton yükseltti: base 18/22 → v2 20/22 → v3 21/22.

Veri büyümesi: tek çağrı seti v2'de 471, v5'te **671**; ajan formatlı set v2'de
1884 (4 format), v5'te **3375** (5 format — supervisor eklendi).

### Hiperparametreler

Tüm ayarlar config dosyalarında; bir koşum "config + seed" ile tam tanımlıdır.

| | tek-çağrı | çok görevli |
|---|---|---|
| config | `training/config_sft.yaml` | `training/config_sft_agents.yaml` |
| veri | `sft_train.jsonl` (güncel 671) | `sft_agents_train.jsonl` (güncel 3375) |
| epoch | 4 | 3 |
| max_seq_length | 1024 | 1536 |

Ortak ayarlar:

```yaml
base_model: Qwen/Qwen2.5-3B-Instruct
load_in_4bit: true                # NF4, double quant
bnb_4bit_compute_dtype: float16   # T4/P100 bf16 desteklemiyor
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules: [q,k,v,o,gate,up,down]_proj   # tüm linear katmanlar
learning_rate: 2.0e-4
lr_scheduler_type: cosine
warmup_ratio: 0.03
max_grad_norm: 0.3
optim: paged_adamw_8bit
per_device_train_batch_size: 2
gradient_accumulation_steps: 4    # efektif batch 8
gradient_checkpointing: true
val_ratio: 0.10
load_best_model_at_end: true      # metric: eval_loss
seed: 42
```

**QLoRA (4-bit NF4).** Ücretsiz T4/P100 bf16 desteklemez. Base model 4-bit'e
kuantize edilir (~6 GB → ~2 GB), yalnızca adaptör fp16 eğitilir. Çıkarımda
adaptör fp16 base'e merge edildiği için **kuantizasyon değerlendirmeye taşınmaz**.

**Completion-only loss.** `DataCollatorForCompletionOnlyLM` ile kayıp yalnızca
asistan turunda hesaplanır; kullanıcı turundaki şema metni maskelenir. Aksi halde
model tablo şeması *üretmeyi* de öğrenir — çıkarımda hiç kullanılmayacak bir
beceriye kapasite harcar.

**Kaynak/format dengeli doğrulama bölümü.** Katmanlar çok dengesiz (şablon 344,
elle 27, hata-hedefli 10). Rastgele bölme azınlık katmanlardan hiç örnek
almayabilir ve eval loss yanıltıcı olur. Çok görevli sette katman ekseni
**format**: doğrulama setinde dört formattan da ~37'şer örnek bulunur.

**En iyi epoch seçimi.** İlk koşumda eval loss epoch 2'de dip yapıp 3'te yükseldi
(hafif aşırı öğrenme). `load_best_model_at_end` olmadan kaydedilen adaptör son
epoch'unki oluyordu. Bu ayarla fazladan bir epoch koşmak risk değil, bilgi olur.

**Greedy decoding.** Değerlendirme koşumlarında `temperature=0.0` — ölçümler
tekrar üretilebilir olsun.

### Compute

Kaggle (T4 16 GB) ve Colab (T4) üzerinde koşuldu. Kaggle'ın haftalık 30 saatlik
kotası açıkça görünür olduğu için tercih edildi.

| koşum | örnek | epoch | adım | süre (T4) |
|---|---|---|---|---|
| tek-çağrı | 381 | 4 | 156 | ~32 dk |
| çok görevli v1 | 1524 | 3 | 513 | ~94 dk |
| çok görevli v2 | 1884 | 3 | 636 | ~126 dk |

Bellek: 4-bit base ~2 GB + aktivasyonlar; T4'ün 16 GB'ında rahat. Her koşum
sonunda `run_report.json` yazılır: GPU adı, torch sürümü, örnek sayıları, süre,
loss eğrisi, hangi epoch'un kaydedildiği.

Paket sürümleri (`training/requirements-train.txt`): TRL/PEFT API'si minor
sürümler arasında değiştiği için sabitlenir. **bitsandbytes sabitlenmez**:
0.44.1 sabitlemesi Colab'ın CUDA 12.8'iyle çakışıp `triton.ops` import hatası
verdi. Space tarafında ayrıca `huggingface_hub<1.0` sınırı gerekir — hub 1.0'da
`use_auth_token` parametresi kaldırıldı, peft 0.13.2 onu hâlâ kullanıyor.

### Sonuçlar

| koşum | eval loss (epoch başına) | kaydedilen |
|---|---|---|
| tek-çağrı | 0.0803 → 0.0552 → **0.0497** → 0.0508 | epoch 3 |
| çok görevli v1 | 0.0651 → 0.0371 → **0.0351** | epoch 3 |
| çok görevli v2 | 0.0600 → 0.0307 → **0.0276** | epoch 3 |

**Tek-ajan karşılaştırması (dev split, 38 soru).** Üç konfigürasyon, aynı sorular.
B ve C **aynı ağırlıklardır**, yalnızca çıkarımdaki prompt farklıdır — bu, eğitim
etkisini prompt etkisinden ayırır.

| metrik | A base+uzun | B SFT+kısa | C SFT+uzun |
|---|---|---|---|
| şema geçerli | 100% | 100% | 100% |
| kolonlar mevcut | 97.4% | 100% | 97.4% |
| grafik-niyet uyumu | 94.7% | 100% | 100% |
| insight sayı içeriyor | 26.3% | 5.4% | 34.2% |
| bunlardan uydurma | 0% | 0% | 5.3% |
| saniye/soru | 5.9 | 5.0 | 5.0 |

Okunuşu:

1. **Kısa prompt aynı işi görüyor.** 1260 → ~250 token, üstelik daha hızlı.
2. **Grafik-niyet uyumu %100.** Anomali örneklerinin genişletilmesinin doğrudan
   sonucu.
3. **Uzun prompt eğitilmiş davranışı geri alıyor.** Dondurulmuş prompt'un
   few-shot örneklerindeki sayılı insight'lar taklit ediliyor. Dağıtım kuralı:
   **SFT modeli kısa promptla kullanılmalıdır.**

**Multi-agent duman testi (8 soru).** Çok görevli adaptörle **8/8 PASSED**, tüm
insight'lar `source=llm` (şablon yedeği yok). Aşama 1'e göre düzelenler:

- Insight JSON dökümü bitti (`_compact_stats` + ajan formatlı eğitim)
- `energy_006`: `day_of_week(date)` üretiyor → 7 gün grubu (önce camelCase
  `dayOfWeek` yazıp motoru şaşırtıyor, 138 takvim günü dönüyordu)
- `retail_013`: doğru kolonlar (discount/profit), r=-0.219, guardrail scatter→box
  (önce `order_id` seçip NaN korelasyon üretiyordu)

**Dev split taraması** (`evaluation/scan_dev_failures.py`, 38 soru):

| | v1 | v2 |
|---|---|---|
| zincir durması | 0 | 0 |
| Evaluation FAILED | 1 | **0** |
| guardrail düzeltmesi | 2 | 1 |
| şablona düşen insight | 0 | 0 |

### Ölçüm notları ve bilinen sınırlar

**Uydurma sayı metriği iki kez düzeltildi.** İlk hali insight'taki her rakamı
uydurma sayıyordu; ama sorunun kendi kısıtı ("2018") ya da şemadaki aralık ("yaş
18-70") uydurma değildir. Düzeltilmiş metrik soru, şema, filtre ve limitten
türeyen sayıları eler ve yuvarlamaya toleranslıdır. Düzeltmeden önce baseline
%26.3 hata oranıyla görünüyordu, sonrasında **%0** — yani baseline sayı
uydurmuyor, şema metadatasını bulgu gibi sunuyordu. İki metrik birlikte
raporlanır: `insight_has_numbers` (üslup) ve `insight_invented_numbers` (doğruluk).

**Halüsinasyon ölçümünün yeri tek-ajan değil.** Tek çağrıda model zaten veri
görmüyor; asıl sınav multi-agent modunda, hesaplanmış istatistiklerin yanında.

**Tek seed, 38 soru, greedy decoding.** Birkaç puanlık farklar tek soru demektir.
Çoklu seed ve bootstrap güven aralıkları final değerlendirmede.

**`has_transform` metriği yanıltıcı.** Relationship ve distribution soruları doğru
olarak transform'suzdur; bu metrik soru tipine koşullanmalı ya da kaldırılmalıdır.

### Adaptörler

| repo | ne |
|---|---|
| `berencarkci/qwen2.5-3b-va-sft-v5` | çok görevli, alan-dışı sağlamlık turu (3375 örnek) — **Space/app bunu kullanır** (`app/main.py`, `DEFAULT_ADAPTER`) |
| `berencarkci/qwen2.5-3b-va-sft-v4` | kategorik dağılım + olumsuzlama + nested share + istatistik sağlamlık düzeltmeleri |
| `berencarkci/qwen2.5-3b-va-sft-v3` | supervisor formatı + filter_aggregation tuzak bankası + diff/anomali genişletmesi |
| `berencarkci/qwen2.5-3b-va-sft-v2` | çok görevli, şema genişletmesi sonrası (1884 örnek) — DPO'nun tabanı |
| `berencarkci/qwen2.5-3b-va-sft` | çok görevli v1 (1524 örnek) |
| `berencarkci/qwen2.5-3b-va-sft-singlecall` | tek-çağrı (381 örnek) |
| `berencarkci/qwen2.5-3b-va-dpo` | v2 üzerine DPO (tüm çiftler) |
| `berencarkci/qwen2.5-3b-va-dpo-real` | v2 üzerine DPO (yalnız gözlenen hatalar) |

Hepsi ayrı tutulur; final değerlendirmede base / SFT / DPO yan yana koşturulur.

## DPO

### Tercih çiftleri

**430 çift, 325 farklı soru**, üç kaynaktan:

| kaynak | çift | ne |
|---|---|---|
| synthetic | 270 | doğrulanmış hedefin bilinen hata modlarıyla bozulması |
| base | 129 | eğitilmemiş modelin kendi cevabı |
| sft_temp | 31 | SFT modeli, sıcaklık 0.9 |

| format | çift |
|---|---|
| data_analyst | 183 |
| supervisor | 157 |
| visualization | 90 |

Neden çoğunluğu sentetik: dev split taraması SFT sonrası modelin 38 sorunun
yalnızca 1-3'ünde hata yaptığını gösterdi. Birkaç yüz çift üretecek kadar gerçek
hata yok, negatiflerin çoğu üretilmek zorunda.

**Bulgu.** `sft_temp` kaynağında 460 soru tarandı ve çıkan 31 çiftin **hepsi
supervisor formatında**. v2 modeli sıcaklık 0.9'da bile Data Analyst ve
Visualization formatlarında yakalanabilir hata üretmiyor. DPO'nun kazanç alanı
büyük ölçüde niyet sınıflandırma; plan kalitesinde tavan yapılmış olabilir.

**Dürüst kayıt.** Visualization çiftlerinin tamamı sentetik. O formatta gerçek
model hatası gözlenmedi, yani DPO sinyali gözleme değil kurguya dayanıyor.

Sentetik bozmalar, projede gözlenmiş hata modlarını taklit eder: yanlış kolon,
düşen filtre, bozulan granülerlik, agregasyon karışıklığı, camelCase sözcük
uyuşmazlığı, izinsiz grafik, tek gruba filtreleme, `agg` slotuna ifade koyma,
ters sıralama yönü, türetilmiş ölçüyü ikiye bölme.

### Rubrik

Altı boyut, format-farkındalıklı, her biri 0/1/2:

| boyut | supervisor | data_analyst | visualization | insight | single_call |
|---|---|---|---|---|---|
| schema_validity | | x | | | x |
| column_selection | | x | | | x |
| transform_correctness | | x | | | x |
| chart_appropriateness | | | x | | x |
| groundedness | | | | x | x |
| clarity | | | x | x | x |
| intent_correctness | x | | | | |

Sert kapılar: olmayan kolon, izinli listede olmayan grafik, `agg`/`sort`
slotunda geçersiz değer, ters sıralama yönü → 0. Gerekçe: sistem bu cevapları
zaten reddederdi.

Ters sıralama neden sunum detayı değil: "hangi şehirler en çok zarar ediyor"
sorusuna `value_desc` cevabı, doğru sıralamanın **yanlış ucunu** gösteriyor.
Eksik sıralama hafif kusur, ters sıralama yanlış cevap.

Karşılaştırma ham puan üzerinden yapılır; normalize skor formatlar arası farklı
ölçekte (3 boyutlu formatta 1 puan = 16.7, 2 boyutluda 25). Bir ham puanlık fark
"belirsiz" sayılıp eğitim dışı bırakılır — zayıf sıralama gürültülü sinyal demek.

**Kalibrasyon** (`evaluation/calibrate_rubric.py`): projede gözlenmiş 10 hata
modu, 10/10 hakem yargısıyla uyumlu. Vakalar spesifikasyon niteliğinde — rubrik
hakemle çelişirse rubrik düzeltilir, tersi değil.

Kalibrasyon bir bug da yakaladı: `verify_grounded` mutlak 0.5 tolerans
kullanıyordu, yani korelasyon katsayılarında (r ∈ [-1, 1]) groundedness kontrolü
hiç çalışmıyormuş. Model `r=-0.219` yerine `r=-0.48` dese bile geçiyordu.

### Hiperparametreler

```yaml
sft_adapter: berencarkci/qwen2.5-3b-va-sft-v2   # referans ve başlangıç noktası
beta: 0.1
learning_rate: 5.0e-6      # SFT'nin kırkta biri
num_train_epochs: 2
```

**Ayrı referans model yüklenmiyor.** SFT adaptörü eğitilebilir olarak yükleniyor
ve TRL onu devre dışı bırakarak referans log-olasılıklarını alıyor. İkinci bir 3B
kopya T4'e sığmaz, ve referans zaten "bu koşumdan önceki model".

**Öğrenme oranı SFT'den kırk kat düşük.** Tercih verisi bir düzeltme, müfredat
değil: 430 çift 2e-4 ile 1884 SFT örneğinin kurduğu davranışın üstüne yazardı.

**beta 0.1** modelin referanstan ne kadar uzaklaşabileceğini sınırlar. Düşürmek
daha çok değişim ama sapma riski, yükseltmek koşumu neredeyse etkisiz kılar.

Çıkan adaptör SFT ve DPO'yu birlikte taşıyor; iki etki değerlendirmede yine
ayrılabilir, iki adaptör yan yana koşturularak.

**İzlenecek asıl gösterge `rewards/accuracies`** — modelin chosen'ı rejected'tan
yüksek puanladığı oran. Başlangıçta ~0.5 (rastgele) olup yükselmeli; loss
düşüşünden daha bilgilendirici.
