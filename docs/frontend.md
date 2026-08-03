# Frontend — Gradio Space

Uygulama tek bir Hugging Face Space; giriş noktası `app/main.py`. Amaç bir demo
değil, projenin iki iddiasını (multi-agent > tek-ajan, DPO > SFT) **canlı ve
ölçümle** gösterebilmek: hem soruyu uçtan uca koşturur, hem de dondurulmuş
benchmark sonuçlarını okur.

## Çalışma modu

İki ortam değişkeni davranışı belirler (`app/main.py`):

| değişken | değer | etki |
|---|---|---|
| `APP_MODE` | `live` (varsayılan) / `mock` | `mock` gerçek model yüklemeden hazır cevap döner — UI testi için |
| `VA_ADAPTER` | Hub id / boş | Servis edilen LoRA adaptörü; boş string = base model. Varsayılan `DEFAULT_ADAPTER = berencarkci/qwen2.5-3b-va-sft-v5` |

`mock` modu dört ajanı (intent, plan, chart, insight) sırayla ayrı cevaplarla
beslemek zorundadır; tek-çağrı formatı tek başına verilirse multi-agent zinciri
Data Analyst'te boş planla durur.

## Sekmeler

Beş sekme (`gr.Tab`):

1. **Ask Your Data.** Örnek veri seti seçilir (Retail/Superstore, Mall, Energy
   hourly) veya kendi CSV/Excel'in yüklenir. Modelin gördüğü **şema özeti**
   gösterilir. Soru yazılır, **Mode** radyosu ile `Single agent` / `Multi agent`
   seçilir (varsayılan Multi). Çıktı akış halinde gelir (grafik + içgörü);
   başarısızlıkta ham model çıktısı bir accordion'da açılır.
   - **Single** = tek model çağrısı, veriye erişim yok → içgörü *bilinçli olarak
     gizlenir* (doğrulanamaz ve sık uydurulur; ham JSON yine görünür).
   - **Multi** = altı çağrıya kadar tam zincir (Supervisor → Data Analyst →
     Visualization → Insight → Evaluation, gerekirse retry).

2. **Agent Trace.** En son multi-agent sorusunun adım adım yürütmesini gösterir
   (her ajanın girdi/çıktısı, guardrail'ler, blame/retry). "Show latest trace".

3. **Model Comparison.** `app/comparison.py`, önceden hesaplanmış
   `results/arm_comparison.json`'dan kol başına metrik özetlerini basar. Canlı
   değerlendirme değil, okuyucudur.

4. **Benchmark Results.** Aynı önhesaplı kaynaktan (`comparison_cache.json`) her
   kolun her soruya cevabını gösterir.

5. **Preference Labeling.** `app/labeling.py`; aynı soruya iki aday cevap üretir,
   hangisinin hangi model olduğunu söylemeden yan yana gösterir, seçimi kaydeder.
   İki önyargı kontrolü: adaylar karıştırılır (konum önyargısı), rubriğin kararı
   seçimden önce gizlenir (çıpalama). Etiketler anında yerel JSONL'e eklenir ve
   istenirse HF Dataset'e push edilir (`HF_TOKEN` Space secret'ı gerekir) — Space
   uyarısız yeniden başlar, diskte kalmayan etiket kaybolur.

## Neden ölçüm offline

Dokuz konfigürasyonu benchmark üzerinde koşmak T4'te ~1 saat sürer; ücretsiz
Space bunu anlık yapamaz. Ölçüm bu yüzden geliştirme reposunda offline koşulur ve
JSON olarak shippenir; Comparison/Benchmark sekmeleri bu dosyaların okuyucusudur.

**Servis edilen adaptör (v5) bilinçli olarak ölçümden yenidir.** Benchmark koşumu
dondurulmuştur: sonraki düzeltmelerden sonra yeniden ölçmek, held-out
değerlendirmeyi model seçimine çevirirdi — held-out split'in tam da engellemek
için var olduğu şey.

Ayrıntılı metodoloji ve sonuçlar için [evaluation.md](evaluation.md),
sistemin çalışma mantığı için [backend.md](backend.md).
