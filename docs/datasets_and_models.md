# Veri Seti ve Model Ön Araştırması

## 1. Aday Veri Setleri

Hedef: 3 farklı analitik alan → perakende satış, enerji tüketimi, müşteri analitiği. Her biri farklı soru tiplerini (trend, karşılaştırma, dağılım, ilişki) doğal olarak destekler.

| Tema | Aday | Kaynak | Neden uygun | Lisans |
|---|---|---|---|---|
| **Perakende satış** | Superstore Sales | Tableau örnek verisi / Kaggle kopyaları | Kategori, alt kategori, bölge, tarih, satış, kâr kolonları → karşılaştırma + trend + kompozisyon soruları; "beyaz eşya kategorisi" senaryosuna kolay uyarlanır | Serbest örnek veri; Kaggle sürümünün lisans sayfasını kontrol et |
| Perakende (alternatif) | Online Retail II | UCI ML Repository | Gerçek işlem verisi, tarih+ülke+ürün → agregasyon ve filtreleme soruları | UCI: CC BY 4.0 (doğrula) |
| **Enerji tüketimi** | Individual Household Electric Power Consumption | UCI ML Repository | Zaman serisi güç tüketimi → trend + anomali soruları; **Beko ile en doğal bağ** (cihaz enerji profili anlatısı) | UCI: CC BY 4.0 (doğrula) |
| Enerji (alternatif) | Appliances Energy Prediction | UCI ML Repository | Cihaz bazlı enerji + ortam sensörleri → ilişki soruları | UCI: CC BY 4.0 (doğrula) |
| **Müşteri analitiği** | Mall Customers | Kaggle | Yaş, gelir, harcama skoru → ilişki + dağılım + segmentasyon soruları (Brief 1'deki "yaş ↔ yıllık harcama" örneğiyle birebir) | Kaggle lisans sayfasını kontrol et |
| Müşteri (alternatif) | Marketing Campaign / Customer Personality | Kaggle | Daha zengin kolon seti; kampanya-yanıt soruları | Kaggle lisans sayfasını kontrol et |

**Seçim önerisi:** Superstore + UCI Household Power + Mall Customers. Üçü de küçük/orta boyutlu (ücretsiz altyapıya uygun), temiz ve yaygın bilinen setler; benchmark'ı savunmak kolay.

**Beko uyarlaması:** Veri setlerine dokunulmaz (provenance korunur); Beko bağlamı **soru setinde** kurulur: "Buzdolabı kategorisinde son çeyrek satış trendi", "Enerji tüketiminin gün içi dağılımı", "Müşteri yaşı ile harcama ilişkisi" gibi. Gerekirse Superstore'un kategori adları senaryo gereği beyaz eşya kategorileriyle eşlenir ve bu eşleme `docs/data.md`'de açıkça belgelenir (overclaim yok).

## 2. Aday Modeller

| Model | Boyut | Artı | Eksi |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 1.5B | Güçlü JSON/instruction uyumu, hızlı iterasyon | Küçük boyutta muhakeme sınırlı |
| Qwen2.5-3B-Instruct | 3B | Kalite/maliyet dengesi iyi — **birincil aday** | QLoRA şart, eğitim biraz daha yavaş |
| Llama-3.2-3B-Instruct | 3B | Geniş ekosistem, TRL örnekleri bol | Lisans şartlarını kontrol et |
| Phi-3-mini (3.8B) | 3.8B | Küçük boyuta göre güçlü muhakeme | Ücretsiz GPU'da DPO sınırda |

**Karar planı:** Gün 5'teki prompt-only baseline'ı Qwen2.5-3B ile kur; şema geçerliliği çok düşükse (R9 riski) 1.5B ile karşılaştır ve kararı `docs/training.md`'ye gerekçesiyle yaz. Tüm hat (SFT → DPO) **tek modelde** koşar — model değiştirmek karşılaştırmayı bozar.

## 3. Ücretsiz Compute Planı

| Kaynak | Kota | Kullanım |
|---|---|---|
| Google Colab (ücretsiz) | T4, oturum başına ~birkaç saat, değişken | SFT ve DPO ana koşuları; checkpoint'ler Drive'a |
| Kaggle Notebooks | Haftalık ~30 saat GPU (P100/T4) | Yedek + uzun koşular (daha öngörülebilir kota) |
| HF Spaces (ücretsiz CPU) | Sürekli | Yalnızca demo/inference; önceden hesaplanmış sonuçlar |
| Yerel makine | — | Veri hazırlığı, değerlendirme scriptleri, Gradio geliştirme |

**Kurallar:** (1) Eğitim asla Space'te koşmaz. (2) Her koşu config + seed dosyasıyla başlar, checkpoint'ler oturum kapanmadan dışa alınır. (3) Uzun koşular gün sonuna bırakılır (Gün 11 ve 14 akşamları — plana işli).

## 4. Gün 4'e Devir Listesi

- [ ] Seçilen 3 setin lisans sayfalarını doğrula, `docs/data.md`'ye linkle
- [ ] Setleri indir, boyut/eksik değer hızlı kontrolü
- [ ] Qwen2.5-3B-Instruct'ı Colab'da yükleyip 1 örnek inference ile doğrula
