# Backend — Orkestrasyon Kararları ve Çalışma Mantığı

Görsel akış için [architecture.md](architecture.md), eğitim tarafı için
[training.md](training.md), veri seti için [data.md](data.md).

## Bölüm 1 — Tasarım Kararları

### Karar: Özel Python orkestrasyonu (framework yok)

Multi agent iş akışı için smolagents/LangGraph gibi bir ajan framework'ü yerine
kendi hafif orkestrasyon katmanımızı yazıyoruz. Gerekçeler:

1. **İş akışımız deterministik bir boru hattı.** Profil → niyet → dönüşüm →
grafik → insight → değerlendirme sırası sabittir, adımların içleri LLM kullanır
ama sırası llm tarafından seçilmez. Ajan framework'lerinin katma değeri, modelin
hangi aracı çağıracağına kendisinin karar verdiği açık uçlu döngülerdedir ve
bizim problemimiz bu sınıfta değil.

2. **Agent Trace birinci sınıf ürün gereksinimi.** Space'teki Agent Trace sekmesi
her adımın yapılandırılmış girdi/çıktısını gösterecek. Kendi orkestrasyonumuzda
trace, mesaj zarfının doğal yan ürünü, framework'te ise framework'ün log biçimine
bağımlı kalırdık.

3. **Benchmark tekrar üretilebilirliği.** Değerlendirme aynı soruya aynı adım
dizisinin uygulanmasını gerektirir. LLM framework'leri adım dizisini değiştirebilir
ve bu metriklere gürültü katar.

4. **Anlaşılabilirlik ve hata ayıklama.** ~300 satırlık kendi kodumuz, katmanlı
framework soyutlamalarından daha kolay anlaşılır ve debug edilir.

### Mesajlaşma: pydantic şemaları, serbest metin değil

Ajanlar arası her aktarım `AgentMessage` zarfı içinde tipli bir payload taşır
(`agents/messages.py`). Ham düşünme süreci veya serbest metin ajanlar arasında
dolaşmaz, her adımın çıktısı doğrulanabilir bir sözleşmedir.

### Niyet sınıflandırma: LLM + kural yedeği (hibrit)

Supervisor niyeti küçük bir LLM çağrısıyla sınıflandırır (tek etiket, pydantic
doğrulamalı). Başarısız çıktıda anahtar kelime tabanlı kural yedeği devreye girer
sistem asla sınıflandırma yüzünden durmaz. Niyet taksonomisi benchmark soru
tipleriyle birebir aynıdır (7 etiket) böylece B5'te "Supervisor niyeti doğru
sınıflandırdı mı" analizi ground truth ile doğrudan eşlenir.

### Loglama: JSONL trace + bellek içi oturum

Her AgentMessage hem `logs/` altına zaman damgalı JSONL olarak yazılır hem de
oturum içi listede tutulur. Agent Trace sekmesi bellek içi listeden beslenir,
JSONL dosyaları hata ayıklama ve rapor örnekleri içindir. `verbose=True` ile
trace terminale de canlı akar: `[saat] step N ajan -> tip | özet`.

### İki temel ilke

Yukarıdaki kararların altında iki ilke yatar:

1. **Sayılar LLM'den gelmez.** Dönüşüm pandas ile çalıştırılır, istatistikler
   deterministik hesaplanır. LLM yalnızca *plan kurar* ve *cümle yazar*.
2. **Son denetçi LLM değildir.** Evaluation Agent tamamen kural tabanlıdır;
   denetleyen katmanın kendisi halüsinasyon yapamamalıdır.

## Bölüm 2 — Orkestrasyon

Ajanlar birbirini doğrudan çağırmaz. Her ajan tipli bir mesaj döndürür,
orkestratör (`agents/orchestrator.py`) bunları toplar ve bir sonraki ajana
yalnızca ihtiyacı olan parçayı verir.

```
run_workflow(client, df, profile, question) -> WorkflowResult
```

| # | Ajan | Girdi | Çıktı |
|---|---|---|---|
| 1 | Supervisor | soru | `IntentResult` |
| 2 | Supervisor | intent | `WorkflowPlan` |
| 3 | Data Analyst | şema + soru + intent | `TransformPlan` (+ hazırlanmış veri) |
| 4 | Visualization | soru + intent + veri özeti + izinli liste | `ChartDecision` |
| 5 | Insight | soru + hesaplanmış istatistikler | `InsightResult` |
| 6 | Evaluation | yukarıdakilerin tümü | `EvalVerdict` |

`run_workflow` **hiç exception fırlatmaz**. Data Analyst planı üretemezse zincir
`StepError` ile durur ve trace yine döner; diğer her durumda cevap üretilir ve
varsa uyarı bayrağıyla teslim edilir.

## Bölüm 3 — Ajan Sözleşmeleri

Mesaj tipleri `agents/messages.py`, grafik önerisi şeması `agents/schemas.py`,
promptların tamamı `agents/prompts.py`'de.

### Supervisor (`agents/supervisor.py`)

```python
IntentResult(intent: str, source: "llm" | "rule_fallback")
WorkflowPlan(intent: str, steps: list[str], insight_focus: str)
```

Niyet, içgörü odağını belirler:

| niyet | insight_focus |
|---|---|
| trend | `trend_stats` |
| comparison | `group_stats` |
| composition | `share_stats` |
| relationship | `correlation` |
| distribution | `distribution_stats` |
| filter_aggregation | `group_stats` |
| anomaly | `outlier_detection` |

### Data Analyst (`agents/data_analyst.py`)

İki fazlı: önce LLM plan kurar, sonra plan **gerçek veride çalıştırılır**.

```python
TransformPlan(transform: Transform, target_columns: list[str],
              result_rows: int, summary_stats: dict, notes: list[str])
```

`Transform` alanları: `groupby`, `agg` (`sum`/`mean`/`count`/`count_distinct`),
`filter` (pandas query), `sort` (`date_asc`/`value_desc`), `limit`.

`groupby` türetilmiş ifade alabilir: `month(col)`, `quarter(col)`, `week(col)`,
`day(col)`, `year(col)`, `hour_of_day(col)`, `day_of_week(col)`,
`weekend_flag(col)`, `bins(col)`.

Sağlamlaştırmalar:
- **Format hatasında alan hatırlatması** (`_FIELD_HINT`) retry prompt'una eklenir.
  Küçük modeller değeri yanlış slota koyabiliyor (ör. `agg="date_asc"`).
- **Geçersiz sort reddedilmez, düşürülür** ve `notes`'a yazılır — plan tamamen
  kaybolmaktansa kısmen uygulanır.
- Tanınmayan türetilmiş ifade groupby'ı atlar, `notes`'a not düşer.

`summary_stats`, `insight_focus`'a göre hesaplanır ve Insight ajanının tek bilgi
kaynağıdır.

### Visualization (`agents/visualization.py`)

LLM yalnızca **izinli listeden** seçer, sonra guardrail'ler veriye karşı doğrular.

```python
ChartDecision(recommendation: ChartRecommendation, guardrails_applied: list[str])
```

| niyet | izinli grafikler |
|---|---|
| trend | line, bar |
| comparison | bar, box |
| composition | pie, bar |
| relationship | scatter, box |
| distribution | histogram, box |
| filter_aggregation | bar, line |
| anomaly | line, box |

Guardrail kuralları (sırayla uygulanır, her düzeltme `guardrails_applied`'a
yazılır — modelin ham tercihi gizlenmez):

1. Grafik izinli listede değilse listenin ilkine çevrilir.
2. `pie` + 5'ten fazla kategori → `bar` (pasta okunmaz olur).
3. `pie` + negatif değer → `bar` (negatif dilim anlamsız).
4. `scatter` + x az sayıda kesikli sayısal değer → `box` (overplotting).

### Insight (`agents/insight.py`)

```python
InsightResult(insight: str, source: "llm" | "template_fallback",
              supporting_stats: dict)
```

`verify_grounded(text, stats)` metindeki her sayının istatistiklerde bulunduğunu
kontrol eder (yuvarlama ve işaret toleranslı). Bir sayı bulunamazsa cümle
reddedilir ve deterministik şablona düşülür — **kullanıcıya doğrulanmamış sayı
gitmez**. NaN/inf değerler karşılaştırmaya girmez; kategorik bir kolon Pearson'a
ulaştığında istatistikler NaN olabiliyor.

`_compact_stats` prompt katmanında büyük grup sözlüklerini kırpar (en büyük 8
grup + `{key}_omitted` sayacı). 48 aylık ya da 138 günlük bir sözlük 3B model
için bilgi değil gürültüdür ve modelin yığını "içgörü" diye geri kusmasına yol
açıyordu. Kırpma yalnızca prompt'ta; trace ve doğrulama tam istatistikle çalışır.

### Evaluation (`agents/evaluation.py`)

LLM kullanmayan son kapı. Sekiz kural:

| kural | ne kontrol eder |
|---|---|
| `schema_valid` | kolonlar gerçekten tabloda var mı |
| `execution_ok` | dönüşüm sonuç üretti mi |
| `chart_intent_fit` | grafik niyetin izinli listesinde mi |
| `insight_grounded` | içgörüdeki sayılar istatistiklerde var mı |
| `stats_health` | NaN korelasyon / n=0 durumu var mı |
| `composition_integrity` | pay sorusunda tek gruba düşülmüş mü |
| `wording_consistency` | yön/şiddet kelimeleri r ile tutarlı mı |
| `insight_informative` | içgörü jenerik şablona mı düşmüş |

```python
EvalVerdict(passed: bool, issues: list[str], warnings: list[str],
            checks: dict, retried_step: str | None, retry_helped: bool | None)
```

**issue vs warning ayrımı:** issue retry tetikler ve `passed=False` yapar; warning
cevapla birlikte teslim edilir. Örnek: `|r|<0.1` ise ifade ilişkiyi abartıyor
olabilir → warning, çünkü cevap yanlış değil, sadece dikkat gerektiriyor.

## Bölüm 4 — Hata Yönetimi ve Hedefli Retry

İki seviye:

**Ajan içi retry.** LLM çıktısı parse edilemez ya da şemaya uymazsa, ajan hata
mesajını modele geri besleyerek bir kez yeniden dener.

**Zincir seviyesi hedefli retry.** Evaluation FAILED verdiğinde **tüm zincir
baştan koşmaz**. `_blame` haritası her kural ihlalini sorumlu ajana eşler:

```python
_BLAME_ORDER = [
    ("data_analyst",  ("execution_ok", "stats_health",
                       "composition_integrity", "schema_valid")),
    ("visualization", ("chart_intent_fit",)),
    ("insight",       ("insight_grounded", "wording_consistency")),
]
```

Sıra önemlidir: aynı anda birden fazla kural düşmüşse en aşağıdaki ajan (Data
Analyst) suçlanır, çünkü yukarıdakiler onun çıktısına bağlıdır.

Retry **yalnızca suçlu ajandan itibaren** koşar, ama bağımlılık gözetilir:

- suçlu `data_analyst` → plan değişeceği için Visualization ve Insight da yeniden
- suçlu `visualization` → yalnızca grafik seçimi yeniden
- suçlu `insight` → yalnızca cümle yeniden, doğru adımların çıktısı korunur

**Reddedilme gerekçesi prompt'a geri beslenir.** Çıkarım `temperature=0.0` ile
yapıldığı için aynı prompt aynı çıktıyı üretir; geri bildirim olmadan retry
maliyetli bir no-op olurdu. `REVIEW_FEEDBACK` şablonu denetçinin gerekçesini
taşır, `RETRY_HINTS` ise kural bazlı düzeltici ipucu ekler (ör. `stats_health`
için: "korelasyon iki sayısal kolon ister; biri kategorikse gruplara göre
karşılaştır"). 3B model tek başına "ne yapmalıyım" çıkarımını yapamıyor.

Tek retry hakkı vardır. İkinci denetim de FAILED derse cevap yine teslim edilir —
ama verdict ve sorun listesiyle birlikte. **Sistem hatayı gizlemek yerine
işaretler.** `retried_step` ve `retry_helped` trace'e yazılır, yani mekanizmanın
kendisi de ölçülebilir.

## Bölüm 5 — Model İstemcisi

`agents/model_client.py` iki uygulama sunar:

- `HFClient` — Qwen2.5-3B-Instruct, transformers ile lokal. `adapter` parametresi
  verilirse LoRA adaptörü `PeftModel` ile yüklenip `merge_and_unload()` ile base'e
  katılır; çıkarım hızı base modelle aynı kalır. Bu tek parametre sayesinde **tüm
  zincir** değişiklik gerektirmeden eğitilmiş modeli kullanır.
- `MockClient` — testler için, model gerektirmez.

Çıkarımda `temperature=0.0` (greedy): ölçümler tekrar üretilebilir olsun.