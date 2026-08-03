# Proje Charter — Tercih Optimizasyonlu Multi-Agent Görsel Analitik Asistanı

## 1. Problem Tanımı

Kullanıcılar bir veri seti hakkında sormak istedikleri soruyu bilir, ancak hangi kolonların, dönüşümlerin ve grafik tipinin uygun olduğunu çoğu zaman bilmez. Küçük açık kaynak dil modelleri de bu görevde güvenilir değildir: yanlış grafik ailesi seçer, yanlış kolonları kullanır veya veride desteklenmeyen "içgörüler" üretir.

Bu proje, doğal dildeki bir analitik isteği şeffaf bir analiz akışına çevirip **uygun grafik + veriye dayalı içgörü** döndüren bir sistem kurar ve bu sistemin model bileşenini insan tercih verisiyle iyileştirir.

## 2. Araştırma Soruları

1. **Mimari:** Multi-agent iş akışı (Supervisor → Data Analyst → Visualization → Insight → Evaluation), tek-ajan baseline'a göre daha doğru, şeffaf ve sağlam görsel analiz cevapları üretir mi?
2. **Öğrenme:** İnsan tercih verisi (DPO), küçük açık kaynak bir modelin grafik öneri davranışını prompt-only ve SFT baseline'ların ötesine taşır mı?

## 3. Kapsam

- CSV/Excel yükleme, veri profili çıkarma, şema özetleme
- Doğal dil sorusu → yapılandırılmış JSON grafik önerisi (`chart_type, x_axis, y_axis, transform, reason, insight`)
- 4 temel grafik tipi: bar, line, scatter, pie (kural korumalı)
- Multi-agent mimari: Supervisor, Data Analyst, Visualization, Insight, Evaluation ajanları; yapılandırılmış ajanlar arası mesajlar; güvenli execution trace
- Model hattı: prompt-only → SFT+LoRA → DPO+LoRA (aynı benchmark, aynı held-out split)
- Tercih verisi: rubrik tanımlı pairwise etiketleme (A / B / berabere / ikisi de kötü), 150–300 kullanılabilir çift hedefi
- Benchmark: ≥3 açık veri seti (perakende satış, enerji tüketimi, müşteri analitiği), 45–60 soru, dondurulmuş held-out split
- Değerlendirme: grafik/kolon/dönüşüm doğruluğu, şema geçerliliği, groundedness, kör insan tercih kazanma oranı, gecikme/compute
- Hugging Face Space (ücretsiz altyapı): Ask Your Data, Agent Trace, Preference Labeling, Model & Mimari Karşılaştırma, Benchmark Results, Methodology sekmeleri
- Reproducible GitHub reposu, bileşen bazlı .md dokümantasyonu, teknik rapor, sunum + demo

## 4. Kapsam Dışı

- Tam bir iş zekâsı platformu; kullanıcı yönetimi, kalıcı depolama, üretim altyapısı
- Ücretli API bağımlılığı (çekirdek demo tamamen açık/ücretsiz bileşenlerle çalışır)
- Ham chain-of-thought'un arayüzde gösterilmesi (yalnızca güvenli, yapılandırılmış trace)
- Reward model + PPO eğitimi → **stretch goal** (yalnızca zorunlu hat tamamlanırsa)
- Histogram/box plot, ileri UI tasarımı, otomatik grafik iyileştirme → **stretch goal**

## 5. Başarı Kriterleri

| # | Kriter |
|---|---|
| 1 | Space, yüklenen tablo veri üzerinde temsili soruları cevaplayıp göreve uygun grafik üretiyor |
| 2 | Multi-agent implementasyon yapılandırılmış ajanlar arası çıktı kullanıyor (serbest prompt zinciri değil) |
| 3 | Evaluation ajanı kasıtlı enjekte edilen hataların en azından bir kısmını yakalıyor |
| 4 | Gerçek pairwise tercih verisi DPO eğitiminde kullanılmış (yalnızca teoride anlatılmamış) |
| 5 | Prompt-only / SFT / DPO aynı held-out sette karşılaştırılmış; kör tercih değerlendirmesi yapılmış |
| 6 | Benchmark: ≥3 veri seti, ≥45 soru, ground truth'lu, dondurulmuş held-out split |
| 7 | Sonuçlar nicel metrik tabloları + kategorize hata analizi (≥5 vaka) ile raporlanmış |
| 8 | Tüm çekirdek işlev ücretli API olmadan, repo talimatlarından tekrar üretilebilir |
| 9 | Bileşen bazlı .md dokümanları (frontend, backend, training, data, evaluation) tamam |
| 10 | 3 günde bir düzenli ilerleme çıktısı + PR akışı işletilmiş |

> Başarı, model karmaşıklığını maksimize etmek değil; tercih optimizasyonunun hedef davranışı iyileştirip iyileştirmediğini gösteren **güvenilir ve tekrar üretilebilir bir karşılaştırma** sunmaktır.

## 6. Opsiyonel hedefler

1. Reward model + PPO karşılaştırması
2. Histogram / box plot desteği ve ilgili benchmark soruları
3. Etiketleyiciler arası anlaşma analizi (Cohen's kappa)
4. MLflow ile deney takibi
