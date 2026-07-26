# Eğitim Metodolojisi

Veri setinin içeriği için [data.md](data.md), sistemin çalışma mantığı için
[backend.md](backend.md).

## Model Seçimi

Qwen2.5-3B-Instruct seçildi. Baseline testlerinde (Task 1.3) grafik ve kolon
seçiminde ilk denemede geçerli JSON üretti, muğlak soruları da doğru çözdü
(ör. "split across segments" → pie + count_distinct). Bu yüzden daha küçük
varyantlara (1.5B) inme ihtiyacı görülmedi. Cihaz: yerelde MacBook (MPS),
final koşumlarda Colab/Kaggle (CUDA).

> Not: `HFClient` varsayılanı bir süre yanlışlıkla 1.5B'de kalmıştı ve Space ile
> tüm multi-agent koşumları 1.5B ile çalışıyordu (B3/T4'te fark edilip
> düzeltildi). Bundan önce kataloglanan hata modları 1.5B'nin hatalarıdır.

## Neden eğitim

Baseline, dondurulmuş 1260 token'lık few-shot prompt kullanıyor. Hedef, format
bilgisini prompt'tan **ağırlıklara taşımak**: kısa prompt, daha az token, daha
hızlı çıkarım ve prompt mühendisliğine bağımlı olmayan davranış.

## SFT (Task 3.2–3.4)

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

### Hiperparametreler

Tüm ayarlar config dosyalarında; bir koşum "config + seed" ile tam tanımlıdır.

| | tek-çağrı | çok görevli |
|---|---|---|
| config | `training/config_sft.yaml` | `training/config_sft_agents.yaml` |
| veri | `sft_train.jsonl` (381) | `sft_agents_train.jsonl` (1524) |
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
| çok görevli | 1524 | 3 | 513 | ~94 dk (epoch ≈ 31 dk) |

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
| çok görevli | 0.0651 → 0.0371 → **0.0351** | epoch 3 |

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
| `berencarkci/qwen2.5-3b-va-sft` | çok görevli (1524 örnek) — Space bunu kullanır |
| `berencarkci/qwen2.5-3b-va-sft-singlecall` | tek-çağrı (381 örnek) |

İkisi ayrı tutulur; final değerlendirmede base / tek-çağrı SFT / çok görevli SFT /
DPO yan yana koşturulacak.

## DPO (Task 4.3)

_B4'te doldurulacak._

Tercih çiftleri gerçek hata kataloğundan üretilecek. Eldeki adaylar:
Visualization'ın relationship sorusuna `bar` önermesi (guardrail düzeltiyor ama
model yanlış), Data Analyst'in kategorik kolonu Pearson'a sokması, guardrail
grafiği değiştirdiğinde `reason` metninin eski grafikten kalması.