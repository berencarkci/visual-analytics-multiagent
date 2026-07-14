# Backend Mimarisi — Orkestrasyon Kararları

## Karar: Özel Python orkestrasyonu (framework yok)

Multi agent iş akışı için smolagents/LangGraph gibi bir ajan framework'ü yerine
kendi hafif orkestrasyon katmanımızı yazıyoruz. Gerekçeler:

1. **İş akışımız deterministik bir boru hattı.** Profil → niyet → dönüşüm → grafik → insight → değerlendirme sırası sabittir, adımların içleri LLM kullanır ama sırası llm tarafından seçilmez. 
Ajan framework'lerinin katma değeri, modelin hangi aracı çağıracağına kendisinin karar verdiği açık uçlu döngülerdedir ve bizim problemimiz bu sınıfta değil.

2. **Agent Trace birinci sınıf ürün gereksinimi.** Space'teki Agent Trace sekmesi her adımın yapılandırılmış girdi/çıktısını gösterecek. 
Kendi orkestrasyonumuzda trace, mesaj zarfının doğal yan ürünü, framework'te ise framework'ün log biçimine bağımlı kalırdık.

3. **Benchmark tekrar üretilebilirliği.** Değerlendirme aynı soruya aynı adım dizisinin uygulanmasını gerektirir. LLM framework'leri adım dizisini değiştirebilir ve bu metriklere gürültü katar.

4. **Anlaşılabilirlik ve hata ayıklama.** ~300 satırlık kendi kodumuz, katmanlı framework soyutlamalarından daha kolay anlaşılır ve debug edilir.

## Mesajlaşma: pydantic şemaları, serbest metin değil

Ajanlar arası her aktarım 'AgentMessage' zarfı içinde tipli bir payload taşır ('agents/messages.py'). Ham düşünme süreci veya serbest metin ajanlar arasında dolaşmaz, her adımın çıktısı doğrulanabilir bir sözleşmedir.

## Niyet sınıflandırma: LLM + kural yedeği (hibrit)

Supervisor niyeti küçük bir LLM çağrısıyla sınıflandırır (tek etiket, pydantic doğrulamalı). Başarısız çıktıda anahtar kelime tabanlı kural yedeği devreye girer sistem asla sınıflandırma yüzünden durmaz. Niyet taksonomisi benchmark soru tipleriyle birebir aynıdır (7 etiket) böylece B5'te "Supervisor niyeti doğru sınıflandırdı mı" analizi ground truth ile doğrudan eşlenir.

## Loglama: JSONL trace + bellek içi oturum

Her AgentMessage hem 'logs/' altına zaman damgalı JSONL olarak yazılır hem de oturum içi listede tutulur. Agent Trace sekmesi bellek içi listeden beslenir, JSONL dosyaları hata ayıklama ve rapor örnekleri içindir.