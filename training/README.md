# SFT + QLoRA Eğitimi (B3/T4)

`data/sft_train.jsonl` üzerinde LoRA adaptörü eğitir. Amaç: modelin kısa
prompt'la geçerli grafik önerisi üretmesi (baseline'ın 1260 token'lık few-shot
prompt'una ihtiyaç kalmadan).

## Dosyalar

| Dosya | İş |
|---|---|
| `config_sft.yaml` | Tüm hiperparametreler + seed. Bir koşum bu dosya + seed ile tam tanımlı. |
| `train_sft.py` | TRL SFTTrainer + PEFT QLoRA, completion-only loss, süre/compute raporu. |
| `../evaluation/sanity_check_sft.py` | 10 dev sorusu, gözle kontrol. |
| `../evaluation/compare_prompt_vs_sft.py` | Üçlü ön karşılaştırma (A/B/C). |

## Colab'da koşum

Runtime > Change runtime type > **T4 GPU** seçili olmalı.

```python
# 1) Repo
!git clone https://github.com/berencarkci/visual-analytics-multiagent.git
%cd visual-analytics-multiagent

# 2) Paketler (sürümler sabit: TRL/PEFT API'si sık değişiyor)
!pip -q install -r training/requirements-train.txt

# 3) (Hub'a push edecekseniz) token
from huggingface_hub import login
login()          # veya: Colab secrets > HF_TOKEN

# 4) Eğitim
!python training/train_sft.py --push
```

Koşum bitince:
- Adaptör → `outputs/sft-qwen2.5-3b/` (yerel yedek) ve Hub'daki model reposu
- Süre/compute notu → `outputs/sft-qwen2.5-3b/run_report.json`

**Yerel yedeği indirmeyi unutmayın** — Colab oturumu kapanınca `outputs/` silinir:
```python
!zip -qr sft_adapter.zip outputs/sft-qwen2.5-3b
from google.colab import files; files.download("sft_adapter.zip")
```

## Beklenen ölçüler

323 örnek, %10 doğrulama, 3 epoch, efektif batch 8 → **~110 optimizer adımı**.
T4'te tahminen **25-40 dakika** (gerçek süre `run_report.json`'a yazılır).
Bellek: 4-bit base ~2 GB + aktivasyonlar; T4'ün 16 GB'ında rahat.

## Eğitim sonrası

```bash
# Gözle kontrol (10 soru)
python evaluation/sanity_check_sft.py --adapter berencarkci/qwen2.5-3b-va-sft

# Ön karşılaştırma (dev split, 3 konfigürasyon)
python evaluation/compare_prompt_vs_sft.py --adapter berencarkci/qwen2.5-3b-va-sft
```

Karşılaştırmanın üç konfigürasyonu:
- **A** base + dondurulmuş uzun prompt (mevcut baseline)
- **B** SFT + kısa prompt (yeni sistem)
- **C** SFT + uzun prompt (farkın ne kadarı eğitimden, ne kadarı prompt'tan)

## Tasarım notları

**Completion-only loss.** Kayıp yalnızca asistan JSON'unda hesaplanır; kullanıcı
turundaki şema metni maskelenir. Aksi halde model şema *üretmeyi* de öğrenir ve
kapasitesini hiç kullanılmayacak bir beceriye harcar.

**4-bit base, fp16 adaptör.** Ücretsiz T4/P100 bf16 desteklemez. Base NF4'e
kuantize edilir (~6 GB → ~2 GB), sadece adaptör fp16 eğitilir. Çıkarımda adaptör
fp16 base'e merge edilir, yani kuantizasyon değerlendirmeye taşınmaz.

**Kaynak-dengeli doğrulama bölümü.** %10 doğrulama, üç kaynaktan (template /
handwritten / failure_targeted) orantılı ayrılır. Rastgele bölme, azınlık
kaynaklardan hiç örnek almayabilir ve eval loss yanıltıcı olur. Bu bölüm
benchmark'tan tamamen ayrıdır; 60 soruluk sete burada dokunulmaz.

**Greedy decoding.** Değerlendirme koşumları `temperature=0.0` ile yapılır ki
karşılaştırmalar tekrar üretilebilir olsun.

## Sorun giderme

| Belirti | Sebep / çözüm |
|---|---|
| `no CUDA device found` | Colab'da GPU runtime seçili değil. |
| `NaN loss` ilk adımlarda | fp16 kararsızlığı; `learning_rate`'i 1e-4'e düşürün. |
| `CUDA out of memory` | `per_device_train_batch_size: 1`, `gradient_accumulation_steps: 8`. |
| `DataCollatorForCompletionOnlyLM` import hatası | TRL sürümü uyuşmuyor; `requirements-train.txt` sürümlerini kullanın. |
| Adaptör Hub'a push olmuyor | `login()` çalıştırılmamış veya token'da yazma izni yok. |