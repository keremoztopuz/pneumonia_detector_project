# Pneumonia Detector (canlı servis): Proje planı

> Bu dosya projenin hafızasıdır. Kararlar, sistem haritası ve ilerleme
> burada tutulur. Her aşama bitince güncellenir.

## Aşama durumu
- [x] 1. Sorular ve proje özeti
- [x] 2. Benzer sistemler ve sistem haritası
- [x] 3. Adım sınıflandırması
- [~] 4. Dosya düzeni: kullanıcı kararıyla ertelendi, aynı repo, düzen dosya açtıkça kurulur
      (kural: yeni kod paket olarak açılır, mutlak import; eski `src/` dokunulmadan kalır)
- [ ] 5. Geliştirme (aşağıdaki adım tablosundan takip)

## Proje özeti
- **Problem (tek cümle):** Göğüs röntgenini NORMAL/PNEUMONIA olarak
  sınıflandıran ve kararı Grad-CAM ısı haritasıyla açıklayan bir modeli,
  tekrarlanabilir şekilde eğitip container olarak canlıya alan uçtan uca
  bir ML servisi.
- **Kullanıcı:** Portföy (mülakatçılar, işverenler). Ek hedef: freelance
  için tekrar kullanılabilir şablon.
- **"Bitti" tanımı:** Canlı URL + CI/CD + MLflow kaydı + README.
- **Girdi → çıktı:** Röntgen görseli (jpeg/png) → sınıf + olasılık +
  Grad-CAM görseli.
- **Veri:** Kaggle "Chest X-Ray Images (Pneumonia)" (pediatrik, tek
  merkez). Lisansı README'ye yazmadan önce doğrulanacak.
- **Deploy hedefi / bütçe:** Sıfır bütçe. Kubernetes lokal `kind` kümesinde
  (öğrenme), herkese açık demo Cloud Run ücretsiz katmanında (aynı image).
- **Zaman:** Hedef 3-4 gün (kullanıcı en iyi ihtimalle 2 gün dedi).
  Eğitim yeniden yapılacağı için "model hazır" varsayımı geçersiz.

## Kararlar
| Karar | Seçim | Neden | Varsayılan mı? |
|-------|-------|-------|----------------|
| Kitle | Portföy | Canlı demo + temiz repo en çok iş görür | Hayır, önerilenden seçildi |
| Gecikme | Birkaç sn, senkron API | ConvNeXt-Tiny CPU'da yeterli, kuyruk gereksiz | Evet |
| Veri saklama | Saklanmaz, bellekte işlenir | Sağlık görüntüsü kişisel veri; KVKK/GDPR yükü doğmaz | Evet |
| Drift/yeniden eğitim | Faz 2 | Önce sabit model, MLflow registry baştan kurulur | Hayır |
| Orkestrasyon | Kubernetes (kind, lokal) + demo için Cloud Run | Sıfır bütçede bulut k8s mümkün değil | Hayır (kullanıcı k8s seçti) |
| Bütçe | Sıfır | Ücretsiz katmanlar, CPU inference | Evet |
| Git akışı | Branch → PR → merge | Kullanıcı rahat; CI PR'larda çalışır | Hayır |
| Rol dağılımı | Mühendislik adımları kullanıcıda, amelelik Claude'da | Hedef öğrenme | Hayır |
| Eğitim ortamı | Kaggle GPU (kod pakette, notebook sadece çağırır) | Lokal CPU yavaş; GPU ücretsiz | Hayır |
| Repo | Aynı repo (`pneumonia_detector_project`), yeni repo yok | Tek geçmiş, tek yer; kullanıcı tercihi | Hayır |
| Dosya düzeni | Önceden kurulmaz, dosya açtıkça yerleşir | Kullanıcı tercihi. Risk not edildi: sonradan taşıma import/Docker/test yollarını kırar | Hayır |
| Model ağırlığı | `models/best_model.pth`, Git LFS; `model_meta.json` ve `metrics.json` yanında (normal git) | Kullanıcı kararı. Risk: LFS ücretsiz bant kotası 1 GB/ay, CI'da `lfs: true` her job'a konmaz | Hayır |
| Ağırlığın kaynağı | Yeniden eğitim (11 Eki 2026): holdout %97.1, resmi Kaggle test %88.5, NORMAL recall 0.71 | README'deki eski sayılar (%97.75) bu ağırlığa ait değil | Hayır |

## Referans sistemler
Gerçek bir "pnömoni servisi, canlıda ve bakımlı" proje bulunamadı. Bulunan
pnömoni repoları terk edilmiş hobi projeleri (0 yıldız, son push 2021/2025).
Bu yüzden sistemin parçalarını ayrı ayrı canlı projelerden alıyoruz.

| Proje | Link | Neden benzer | Aldığımız fikir | Almadığımız ve neden |
|-------|------|--------------|-----------------|----------------------|
| Made-With-ML (49k★, push 2026-03, MIT) | https://github.com/GokuMohandas/Made-With-ML | Eğitim + değerlendirme + servis + CI/CD'li uçtan uca ML projesi | Kütüphane kodunun paket olarak ayrılması (`madewithml/`), `deploy/` klasörü, workflow'ların iş başına ayrılması, `Makefile` ile `style` hedefi | Anyscale/Ray ve AWS'ye bağımlı deploy; bizim ölçeğimizde ağır ve ücretli |
| KServe (6k★, push bugün, Apache-2.0) | https://github.com/kserve/kserve | Kubernetes üzerinde model servisi referansı | Çok aşamalı, venv'li, `uv` kullanan model Dockerfile'ları; her image için ayrı publish workflow'u; Helm + kustomize ile kurulum | Controller/CRD (InferenceService) katmanı; tek model için aşırı karmaşıklık |
| MLflow (28k★, push bugün, Apache-2.0) | https://github.com/mlflow/mlflow | Deney takibi + model registry standardı | Run/param/metrik/artefakt kavramları, registry'de sürümleme | Tracking server'ı bulutta barındırmak (sıfır bütçe) |
| TorchXRayVision (1.2k★, push 2026-10-01) | https://github.com/mlmed/torchxrayvision | Göğüs röntgeni modelleri için canlı, bakımlı kütüphane | Görüntü ön işleme/normalizasyonun kütüphanede tek yerde olması; DICOM'lu testler; `tests/` + `pep8.sh` | Çoklu hastalık modelleri ve veri seti sarmalayıcıları; kapsamımız dışında |
| ml-deployment-k8s-fastapi (223★, push 2022, Apache-2.0) | https://github.com/sayakpaul/ml-deployment-k8s-fastapi | FastAPI + Docker + Kubernetes + GitHub Actions ile görüntü sınıflandırma servisi | `.kube/` altında base + kustomize overlay yapısı ve CI ile GKE'ye deploy fikri | Bakımı bitmiş (2022), GKE ücretli. Sadece okunabilir desen referansı, "canlı" sayılmaz |

## Sistem haritası
```
 Kaggle veri seti              Eğitim (Kaggle GPU)             Kayıt
 chest-xray-pneumonia  ───▶  pneumonia/ paketi (data,    ───▶  MLflow: run, metrik,
 hasta bazlı bölme           model, training, evaluation)       artefakt, registry
 (ref: Made-With-ML)         tek ön işleme: CLAHE+crop          (ref: MLflow)
                                      │
                                      ▼  best_model.pth + model_meta.json
                              Servis (FastAPI, CPU)
                              POST /predictions  GET /health
                              bellekte işle, Grad-CAM, saklama yok
                              (ref: Made-With-ML serve.py)
                                      │
                                      ▼
                              Docker image (multi-stage, non-root)
                              (ref: KServe Dockerfile'ları)
                                      │
        Git push / PR  ───▶  CI: flake8 → pytest → docker build
        (GitHub Actions)              │ main'e merge
                                      ▼
                         GHCR'a image push (sürüm etiketli)
                          │                          │
                          ▼                          ▼
              Cloud Run (herkese açık URL)   kind kümesi (lokal k8s)
              ücretsiz katman                Deployment + Service + probe
                                             + ConfigMap + HPA (kustomize)
                                             (ref: KServe, ml-deployment-k8s-fastapi)
                                      │
                                      ▼
              İzleme: yapılandırılmış log, /metrics (Faz 2: drift)
```

## API sözleşmesi (karar: 13 Eki 2026)
| Endpoint | Girdi | Başarı | Hatalar |
|----------|-------|--------|---------|
| `GET /health` | yok | 200 `{"status": "ok"}` | model yüklü değilse 503 |
| `POST /predictions` | `multipart/form-data`, alan adı `file` (jpeg/png, ham görsel) | 200: sınıf, olasılık, Grad-CAM (base64 PNG) | 413 dosya çok büyük, 415 desteklenmeyen tür, 422 çözülemeyen/bozuk görsel, 500/503 sunucu hatası |

- Ön işleme sunucuda yapılır (`pneumonia/preprocessing.py`), istemci ham görsel yükler.
- Başarıda 200 (201 değil): hiçbir şey saklanmıyor, kaynak oluşmuyor.
- Grad-CAM tahminin doğruluğunu değil, modelin nereye baktığını gösterir.
  Arayüz ve README bunu belirtmeli.
- Model yüklenemezse hata sunucudadır: 5xx, 4xx değil.

## Dosya düzeni
(Aşama 4'te onaylanacak.)

## Adımlar
Etiket: 🧱 amelelik · 🔁 bir kez öğren · 🧠 mühendislik · ⭐ mülakatta anlatılır

| # | Adım | Etiket | Ne öğretir | Kim yapar | Durum |
|---|------|--------|------------|-----------|-------|
| 1 | Klasör ağacı ve paket yapısı | 🧠 | Eğitim, servis ve scriptleri sorumluluğa göre ayırmak | Sen, ben sorarım | ⏳ |
| 2 | `.gitignore`, `.dockerignore`, `__init__.py`, `pyproject.toml` | 🧱 | — | Ben | ⏳ |
| 3 | Config modülü (pydantic-settings) + `.env.example` | 🔁 | Ayarı koddan ayırmak (12-factor) | Sen | ⏳ |
| 4 | Eğitim kodunu pakete taşı, ön işlemeyi tek yere al | 🧠 ⭐ | Train/serve skew: eğitimde CLAHE+crop, Grad-CAM'de yoktu | Sen | ⏳ |
| 5 | MLflow tracking: parametre, metrik, artefakt | 🔁 | Deney takibi, run karşılaştırma | Sen | ⏳ |
| 6 | Eğitimi çalıştır, iki test setinde doğrula, karar eşiğini seç | 🧠 ⭐ | Dış test %88.6 vs iç test %98.5; NORMAL recall 0.71 | Sen | ⏳ |
| 7 | Model artefaktı nerede yaşar? (registry, saklama, sürümleme) | 🧠 | Ağırlığın git dışında nasıl taşınacağı | Sen | ⏳ |
| 8 | API sözleşmesi: endpoint'ler, şemalar, hata kodları | 🧠 ⭐ | `POST /predictions` tasarımı, 4xx/5xx ayrımı | Sen | ⏳ |
| 9 | Servis katmanı: model yükleme (lifespan), tahmin, Grad-CAM | 🧠 | İnce route, kalın servis; her istekte model yüklememek | Sen | ⏳ |
| 10 | Girdi doğrulama ve güvenlik: boyut, MIME, rate limit | 🧠 | Herkese açık servise gelen kötü girdi | Sen | ⏳ |
| 11 | Testler: birim, API, hata yolları | 🧠 | Neyin test edileceğine karar vermek | Sen (fixture'ı ben) | ⏳ |
| 12 | flake8 + pre-commit | 🧱 | — | Ben | ⏳ |
| 13 | Dockerfile: multi-stage, non-root, CPU torch, image boyutu | 🔁 | Katman önbelleği, küçük image, güvenlik | Sen (daha önce Docker yaptın, hızlı geç) | ⏳ |
| 14 | docker compose (api + mlflow) | 🔁 | Çok servisli yerel ortam | Sen | ⏳ |
| 15 | CI: flake8 → pytest → docker build | 🔁 | PR'da otomatik kalite kapısı | Sen (Actions biliyorsun) | ⏳ |
| 16 | GHCR'a push ve image sürümleme | 🔁 | Sürüm etiketi, geri alınabilirlik | Sen | ⏳ |
| 17 | Cloud Run CD (en az yetki, Workload Identity) | 🔁 | Anahtarsız bulut kimlik doğrulama | Sen (bulut deploy yaptın) | ⏳ |
| 18 | K8s manifest'leri: Deployment, Service, probe, limitler, ConfigMap, HPA | 🧠 ⭐ | Orkestrasyon temelleri, kaynak limitleri | Sen, ben sorarım | ⏳ |
| 19 | kind kümesi, kustomize/Helm kararı, yerel deploy | 🔁 | Kümeyi ayağa kaldırmak, `kubectl` akışı | Sen | ⏳ |
| 20 | CI'da kind ile smoke test | 🧠 | Manifest'leri merge'den önce doğrulamak | Sen | ⏳ |
| 21 | Gözlemlenebilirlik: yapılandırılmış log, `/metrics` | 🧠 | Neyin izlenip alarm olacağı | Sen | ⏳ |
| 22 | Model kartı, tıbbi uyarı, README | 🧠 (içerik) 🧱 (format) | Sorumlu yayın, dürüst raporlama | Sen yaz, ben biçim | ⏳ |
| 23 | Mimari diyagram ve mülakat hikayesi | 🧱 | — | Ben, sen onayla | ⏳ |

## Öğrenilenler
- (🔁 adımlardan tamamlananlar ve mülakatta anlatılacak hikayeler buraya)

## Gelecek iş
- Faz 2: otomatik yeniden eğitim hattı, MLflow registry'de staging → production terfisi
- Faz 2: drift izleme (girdi ve tahmin dağılımı), alarm
- Faz 2: ONNX'e dönüştürüp gecikmeyi ölçmek
- Faz 2: bulutta yönetilen Kubernetes (ücretli, bütçe gelirse)
- Dış kaynaklı bir doğrulama seti ile genelleme testi (farklı hastane/cihaz)
