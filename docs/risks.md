# Risk Kaydı

Durum kodları: 🟢 kontrol altında · 🟡 izleniyor · 🔴 gerçekleşti/aksiyon gerekli

| # | Risk | Olasılık | Etki | Önlem | Tetik göstergesi | Durum |
|---|---|---|---|---|---|---|
| R1 | Tercih etiketleri gürültülü/tutarsız | Orta | Yüksek | Rubrik etiketlemeden önce tanımlı (Gün 13 sabah); 5–10 kalibrasyon örneği; berabere/ikisi-kötü seçenekleri; mümkünse kesişim setiyle anlaşma oranı | Kalibrasyon örneklerinde kararsızlık; anlaşma < ~%70 | 🟢 |
| R2 | Aday çıktılar birbirine çok benziyor → anlamsız tercih çiftleri | Orta | Yüksek | Kontrollü aday üretimi: farklı temperature/prompt varyantları, base+SFT model karışımı | Etiketlemede "berabere" oranı > ~%40 | 🟢 |
| R3 | Yetersiz compute (ücretsiz GPU kotaları) | Orta | Orta | 1–3B model + QLoRA; kısa sequence; sınırlı veri; sık checkpoint; Colab ↔ Kaggle yedekli plan | Tek epoch > 2 saat; kota uyarısı | 🟢 |
| R4 | Değerlendirme sızıntısı (test verisinin eğitime karışması) | Düşük | Yüksek | Held-out split Gün 5 sonunda dondurulur; otomatik kesişim kontrol scripti her veri üretiminde koşar | Kesişim scripti > 0 eşleşme | 🟢 |
| R5 | Kapsam şişmesi | Yüksek | Yüksek | PPO+RM, histogram/box plot, ileri UI = stretch; her blok sonunda kapsam gözden geçirme; "asgari çıktı + iyileştirme" mantığı | Blok çıktısı gün sonunda hazır değil | 🟢 |
| R6 | Sentetik eğitim verisi kalitesi düşük | Orta | Orta | Her örnekte kaynak etiketi (human/synthetic/edited); sentetik örneklerin rastgele %10'unu elle denetle; genelleme iddialarında ölçülülük | Denetlenen örneklerde hata > %15 | 🟢 |
| R7 | Gün kaybı (İSG ve gezi planda; ek beklenmedik kesintiler) | Orta | Orta | Eğitim aşamalarında esnek tampon; eğitim koşuları geceye/paralele alınır | Herhangi bir blokta 1+ gün kayıp | 🟡 (Gün 2–3 biliniyor) |
| R8 | Ücretsiz HF Space kısıtları (CPU, cold start, bellek) | Orta | Düşük | Önceden hesaplanmış sonuçlar + hafif inference; eğitim Space dışında; büyük dosyalar repoda değil | Space yanıt süresi > ~30 sn | 🟢 |
| R9 | Küçük modelin JSON şema uyumu düşük | Orta | Orta | Katı şema + pydantic doğrulama + yeniden deneme; SFT verisinde şemaya sıkı bağlılık; gerekirse constrained decoding | Prompt-only şema geçerliliği < ~%70 | 🟢 |
| R10 | İkinci etiketleyici desteği alınamaması | Düşük | Düşük | Gün 13'te önceden haber; tek etiketleyiciyle de proje geçerli, anlaşma analizi stretch'e düşer | Gün 14 sonunda ikinci etiketleyici yok | 🟢 |

**Gözden geçirme ritmi:** Her blok kapanışında (Gün 6, 9, 12, 15, 18) durum kolonu güncellenir; 🔴 olan riskler ilerleme raporuna eklenir.
