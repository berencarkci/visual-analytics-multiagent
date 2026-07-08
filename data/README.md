# Veri Setleri

Üç açık veri seti, üç farklı analitik alanı temsil eder. Tam sürümler bu klasörde, 500 satırlık örnekler `samples/` altındadır. Benchmark soruları (B1) bu setler üzerine yazılır.

| Dosya | Tema | Boyut | Orijinal kaynak |
|---|---|---|---|
| `retail_sales_superstore.csv` | Perakende satış | 9.994 satır × 19 kolon | Tableau "Sample - Superstore" örnek verisi |
| `customer_analytics_mall.csv` | Müşteri analitiği | 200 satır × 5 kolon | Mall Customer Segmentation (Kaggle) |
| `energy_consumption_appliances.csv` | Enerji tüketimi (10 dk çözünürlük) | 19.735 satır × 27 kolon | UCI ML Repository: Appliances Energy Prediction |
| `energy_consumption_hourly.csv` | Enerji tüketimi (saatlik özet) | 3.290 satır × 27 kolon | Yukarıdakinden türetildi (saatlik ortalama) |

## İndirme kaynağı ve lisans notu

Dosyalar geliştirme kaggle'dan alındı:

- Superstore: `https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting?resource=download`
- Mall Customers: `https://www.kaggle.com/datasets/shwetabh123/mall-customers`
- Appliances Energy: `https://www.kaggle.com/datasets/gaganmaahi224/appliances-energy-consumption`

## Uygulanan temizlik adımları

**retail_sales_superstore.csv**
- Kolon adları `snake_case`'e çevrildi; `row_id` (satır indeksi) ve `country` (tek değer: United States) kolonları atıldı
- `order_date`, `ship_date` → datetime (kaynak format M/D/YYYY)
- Mirror'daki 806 tamamen boş satır atıldı → kalan 9.994 satır klasik Superstore ile birebir
- Bilinen 11 eksik `postal_code` (Burlington, Vermont satırları) 05401 ile dolduruldu; posta kodları başındaki sıfır korunarak string yapıldı
- Orijinal veride bulunan 1 duplike işlem satırı bilinçli olarak korundu (gerçek veri özelliği)

**customer_analytics_mall.csv**
- Kolonlar yeniden adlandırıldı: `customer_id, gender, age, annual_income_k_usd, spending_score`
- Eksik değer yok; başka müdahale yapılmadı

**energy_consumption_appliances.csv**
- Kolon adları `snake_case`; `date` → datetime
- Kaynaktaki string-sayı biçimleri (baştaki boşluklar, 15+ ondalık) sayısala çevrilip 3 haneye yuvarlandı
- `rv1`, `rv2` atıldı (veri setinin dokümantasyonunda "rastgele değişken" olarak tanımlı, analitik değeri yok)
- Saatlik özet (`energy_consumption_hourly.csv`) 10 dakikalık verinin saatlik ortalamasıdır — Space demosu için hafif sürüm

## Yeniden üretim

Temizlik adımları `notebooks/` altındaki hazırlık scriptiyle veya bu README'deki tanımla yeniden üretilebilir. Ham dosyalar repoya konmaz (`.gitignore`: `data/raw/`).
