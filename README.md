# Tercih Optimizasyonlu Multi-Agent Görsel Analitik Asistanı

Kullanıcının tablo veri (CSV/Excel) yükleyip doğal dilde soru sorabildiği, uygun grafiği ve veriye dayalı içgörüyü üreten multi-agent bir Hugging Face Space. Sistemin grafik öneri modeli **prompt-only → SFT+LoRA → DPO** hattıyla iyileştirilir ve aynı benchmark üzerinde karşılaştırılır.

Model: **Qwen2.5-3B-Instruct**, transformers ile lokal (API değil — ağırlıklara dokunup eğitiyoruz).

## Araştırma Soruları
1. Multi-agent iş akışı, tek-ajan baseline'a göre daha doğru ve şeffaf görsel analiz cevapları üretir mi?
2. İnsan tercih verisi (DPO), küçük açık kaynak bir modeli prompt-only ve SFT baseline'ların ötesine taşır mı?

## Depo Yapısı
```
app/         Gradio Hugging Face Space uygulaması
agents/      Supervisor, Data Analyst, Visualization, Insight, Evaluation ajanları
training/    SFT ve DPO eğitim scriptleri, config'ler
evaluation/  Benchmark, metrik scriptleri, sonuçlar
data/        Veri setleri (örneklem), SFT ve tercih verileri
docs/        charter, risks, datasets_and_models, architecture, backend, frontend, training, data, evaluation, error_analysis
notebooks/   Colab/Kaggle eğitim ve keşif defterleri
```

## Kurulum
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

İlk çalıştırmada base model (~6 GB) ve LoRA adaptörü (~120 MB) Hugging Face
Hub'dan iner. GPU yoksa CPU'da çalışır ama yavaştır.

## Çalıştırma

### Arayüz
```bash
python app/main.py   # Gradio arayüzü (yerel)
```

### Testler
```bash
python test_baseline.py                                             # tek-ajan baseline
python test_multiagent.py                                           # multi-agent, base model
python test_multiagent.py --adapter berencarkci/qwen2.5-3b-va-sft-v5    # multi-agent, SFT adaptörü
python test_eval_injection.py                                       # Evaluation birim testleri
```

`test_eval_injection.py` model gerektirmez (MockClient), saniyeler sürer,
`12/12 checks passed` vermelidir. `test_multiagent.py` sonuçları
`evaluation/results/smoke_multiagent.json`'a yazar.

### Veri üretimi
```bash
python evaluation/make_split.py             # benchmark dev/test bölünmesi (seed 42)
python evaluation/make_sft_data.py          # tek-çağrı SFT seti (381 örnek)
python evaluation/make_agent_sft_data.py    # ajan formatlı set (1524 örnek)
```

`make_sft_data.py` yazmadan önce **benchmark kesişim kontrolü** yapar (birebir +
fuzzy). Bir soru benchmark'a çok benziyorsa üretim durur — eğitim setine
benchmark sızmasını engeller. Test split mühürlüdür, geliştirme sırasında
kullanılmaz.

### Eğitim
Colab/Kaggle (T4 GPU) üzerinde koşulur. Adım adım: [training/README.md](training/README.md)

```bash
pip install -r training/requirements-train.txt

python training/train_sft.py                                            # tek-çağrı, ~32 dk
python training/train_sft.py --config training/config_sft_agents.yaml   # çok görevli, ~94 dk
```

Her koşum `run_report.json` yazar: GPU, süre, loss eğrisi, hangi epoch kaydedildi.

### Değerlendirme
```bash
python evaluation/sanity_check_sft.py --adapter berencarkci/qwen2.5-3b-va-sft-v5      # 10 soru, gözle kontrol
python evaluation/compare_prompt_vs_sft.py --adapter berencarkci/qwen2.5-3b-va-sft-v5  # 3 konfigürasyon, dev split
```

Sonuçlar `evaluation/results/` altına JSON olarak yazılır.

## Tekrar Üretilebilirlik
- Tüm rastgelelik `seed: 42` ile sabitlenir (veri bölünmesi, eğitim, örnekleme)
- Çıkarım `temperature=0.0` (greedy) — aynı girdi aynı çıktıyı verir
- Bir eğitim koşumu "config dosyası + seed" ile tam tanımlıdır
- Üretilmiş eğitim setleri repoda commit'lidir; hangi adaptörün hangi veriyle
  eğitildiği izlenebilir

## Durum
Hat tamamlandı: SFT (tek-çağrı + çok görevli ajan formatları, v1→v5) ve DPO
(tüm çiftler + yalnız-gözlenen-hatalar varyantı). Final değerlendirme
base / SFT / DPO × tek-ajan / multi-agent matrisinde koşuldu; donuk test split
sonuçları `evaluation/results/test_split_results.md`, kol özetleri
`evaluation/results/arm_comparison.json`. Canlı Space v5 adaptörünü servis eder.
Ayrıntı: [Değerlendirme](docs/evaluation.md).

## Dokümanlar
- [Proje Charter](docs/charter.md)
- [Risk Kaydı](docs/risks.md)
- [Veri Seti ve Model Araştırması](docs/datasets_and_models.md)
- [Mimari](docs/architecture.md) — akış şeması ve sıralı diyagram
- [Backend](docs/backend.md) — orkestrasyon kararları, ajan sözleşmeleri, evaluator, retry/blame
- [Frontend](docs/frontend.md) — Gradio Space, sekmeler, çalışma modu
- [Eğitim](docs/training.md) — SFT verisi, hiperparametreler, compute, sonuçlar
- [Veri Kartı](docs/data.md) — kaynaklar, dağılımlar, üretim kuralları
- [Değerlendirme](docs/evaluation.md) — benchmark, dev/test split, kollar, rubrik, problar
- [Hata Analizi](docs/error_analysis.md) — gözlenen hata modları ve kök nedenleri