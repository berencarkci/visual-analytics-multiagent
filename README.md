# Tercih Optimizasyonlu Multi-Agent Görsel Analitik Asistanı

Kullanıcının tablo veri (CSV/Excel) yükleyip doğal dilde soru sorabildiği, uygun grafiği ve veriye dayalı içgörüyü üreten multi-agent bir Hugging Face Space. Sistemin grafik öneri modeli **prompt-only → SFT+LoRA → DPO** hattıyla iyileştirilir ve aynı benchmark üzerinde karşılaştırılır.

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
docs/        charter, risks, datasets_and_models, frontend, backend, training, data, evaluation
notebooks/   Colab/Kaggle eğitim ve keşif defterleri
```

## Kurulum
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Çalıştırma
```bash
python app/main.py   # Gradio arayüzü (yerel)
```
(Eğitim ve değerlendirme komutları ilgili bloklar tamamlandıkça buraya eklenecek.)

## Dokümanlar
- [Proje Charter](docs/charter.md)
- [Risk Kaydı](docs/risks.md)
- [Veri Seti ve Model Araştırması](docs/datasets_and_models.md)
