# Ollama ile Yerel Yapay Zekâ Uygulaması

Bu proje, tarayıcıdan prompt gönderip bilgisayarınızda çalışan Ollama modelinden cevap alabileceğiniz küçük bir sohbet uygulamasıdır. Python dışında ek bir paket gerektirmez.

## 1. Ollama nedir?

Ollama, büyük dil modellerini bilgisayarınızda indirmenizi, çalıştırmanızı ve uygulamalara bağlamanızı kolaylaştıran bir model çalıştırıcısıdır. Ollama cevapların kendisi değildir: `gemma3`, `qwen3` veya benzeri bir model cevapları üretir; Ollama modeli yönetir ve ona terminal/API üzerinden erişim sağlar.

Temel akış şöyledir:

```text
Kullanıcı → Tarayıcı arayüzü → Python sunucusu → Ollama API → Yerel model
```

## 2. Neden kullanılır?

- Verilerinizin yerel bilgisayarda kalmasını istediğiniz projelerde
- İnternet bağlantısı olmadan çalışan bir asistan gerektiğinde
- Her API isteği için ücret ödemeden prototip geliştirirken
- Model, sistem promptu ve uygulama davranışı üzerinde kontrol istediğinizde
- Özetleme, soru-cevap, metin üretme ve kod yardımı gibi işler için

Yerel kullanımın bedeli, hesaplamanın sizin cihazınızda yapılmasıdır. Daha büyük modeller daha fazla RAM, disk alanı ve işlem gücü ister. Ayrıca yerel çalışma tek başına doğruluk garantisi vermez; model hatalı bilgi üretebilir.

## 3. Nerelerde kullanılır?

- Terminal sohbetleri
- Web ve masaüstü sohbet arayüzleri
- Kurum içi belge soru-cevap sistemleri
- Kodlama yardımcıları
- Metin sınıflandırma, özetleme ve veri çıkarma servisleri
- Python/JavaScript uygulamalarının arka ucunda yerel AI API'si

## 4. Ön koşulları kontrol edin

Terminal açıp çalıştırın:

```bash
ollama --version
ollama list
python3 --version
```

Ollama yanıt vermiyorsa macOS'ta Ollama uygulamasını Applications klasöründen açın. Linux'ta gerekirse `ollama serve` çalıştırın.

## 5. İlk modeli indirin

Küçük ve hızlı başlangıç modeli:

```bash
ollama pull gemma3:1b
```

İndirmeden sonra terminalde test edin:

```bash
ollama run gemma3:1b "Türkiye hakkında üç ilginç bilgi ver."
```

Etkileşimli sohbet için:

```bash
ollama run gemma3:1b
```

Çıkmak için `/bye` yazın.

## 6. Bu uygulamayı çalıştırın

Terminalde bu proje klasörüne geçin:

```bash
cd "/Users/iremsupalaa/Documents/Codex/2026-08-07/ollamay-kurman-ve-kurcalaman-istiyorum-herhangi/outputs/ollama-chat-app"
python3 app.py
```

Tarayıcıda şu adresi açın:

```text
http://127.0.0.1:8080
```

Metin kutusuna sorunuzu yazıp **Gönder** düğmesine basın. Enter gönderir; Shift+Enter yeni satır oluşturur. Uygulamayı durdurmak için terminalde `Ctrl+C` kullanın.

## 7. Kod nasıl çalışıyor?

1. `index.html` mesajı toplar ve `/api/chat` adresine yollar.
2. `app.py` mesajı yerel Ollama API'sindeki `http://127.0.0.1:11434/api/chat` adresine iletir.
3. Ollama, seçili modeli çalıştırır.
4. Modelin cevabı Python sunucusu üzerinden tarayıcıya döner.
5. Tarayıcı, konuşma geçmişini sonraki isteğe eklediği için model sohbetin bağlamını hatırlar.

Tarayıcının Ollama'ya doğrudan bağlanması yerine küçük bir arka uç kullanmak, ileride doğrulama, erişim kontrolü, kayıt tutma veya farklı model sağlayıcıları eklemeyi kolaylaştırır.

## 8. Başka model kullanmak

Örneğin daha güçlü fakat daha büyük `gemma3:4b` modelini indirin:

```bash
ollama pull gemma3:4b
```

Arayüzdeki **Model** alanını `gemma3:4b` olarak değiştirin. Sadece `ollama list` çıktısında görünen bir model adı kullanın.

## 9. API'yi doğrudan test etmek

```bash
curl http://127.0.0.1:11434/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma3:1b",
    "messages": [{"role": "user", "content": "Merhaba! Kısaca kendini tanıt."}],
    "stream": false
  }'
```

## 10. Sık karşılaşılan sorunlar

**`Ollama'ya ulaşılamadı`**  
Ollama uygulamasını açın veya uygun sistemde `ollama serve` çalıştırın.

**`model not found`**  
Önce `ollama pull gemma3:1b` komutunu çalıştırın ve arayüzde aynı adı kullanın.

**Cevap çok yavaş**  
Daha küçük bir model seçin; ilk cevapta modelin belleğe yüklenmesi de zaman alabilir.

**8080 portu kullanımda**  
Başka port seçin:

```bash
APP_PORT=8081 python3 app.py
```

Sonra `http://127.0.0.1:8081` adresini açın.

## 11. Sonraki geliştirmeler

- Sistem promptu/persona ayarı
- Sohbetleri dosyaya veya veritabanına kaydetme
- Yanıtı kelime kelime göstermek için streaming
- Belge yükleyip içerik üzerinde soru-cevap (RAG)
- Kullanıcı girişi ve çoklu kullanıcı desteği
- Docker ile paketleme

Uygulama yalnızca `127.0.0.1` üzerinde dinlediği için yerel makinenizden erişilebilir. İnternete açmadan önce kimlik doğrulama, HTTPS, oran sınırlama ve girdi kontrolleri ekleyin.
# ollama-chat-app
