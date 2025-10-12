<div align="center">

# 🎨 GAN-Based Manga Colorization Project

[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Hugging Face](https://img.shields.io/badge/Dataset-Hugging%20Face-yellow.svg)](https://huggingface.co/datasets/MichaelP84/manga-colorization-dataset)
[![Colab](https://img.shields.io/badge/Google-Colab-orange.svg)](https://colab.research.google.com/)

## 📎 Live Demo - Canlı Demo

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Demo-yellow?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/spaces/koesan/mangaspaces)

---

🇬🇧[English](#english) | 🇹🇷[Türkçe](#türkçe)

---

</div>

## English

### 🇬🇧

## 📖 Project Overview

This project implements three state-of-the-art Generative Adversarial Network (GAN) architectures for automatic colorization of black and white manga images. Through systematic implementation and comparative analysis, we demonstrate the effectiveness of different GAN approaches in image-to-image translation, specifically addressing the unique challenges of manga colorization including line art preservation, style consistency, and culturally appropriate color application.

The project provides comprehensive implementations of Pix2Pix, Pix2PixHD, and CycleGAN models, each offering distinct advantages for the manga colorization task. By leveraging these architectures, we've created robust pipelines capable of generating high-quality, aesthetically pleasing colorizations while maintaining the artistic integrity of the original work.

### 🎯 Core Objectives
- **Comparative Analysis**: Systematically evaluate performance across Pix2Pix, Pix2PixHD, and CycleGAN architectures for manga colorization
- **Technical Implementation**: Develop optimized image-to-image translation pipelines with reproducible training workflows
- **Quality Generation**: Produce visually compelling, culturally appropriate colorizations that preserve artistic intent and line work integrity
- **Research Contribution**: Advance understanding of GAN applications in digital art restoration and enhancement

## 🏗️ Architectural Framework

### 🔬 Model Implementations

#### 1. **Pix2Pix: Supervised Learning Foundation**
Built upon the conditional GAN framework, Pix2Pix utilizes a U-Net generator architecture paired with a PatchGAN discriminator. This supervised approach leverages paired training data to establish direct mappings between grayscale and color representations. The model excels in maintaining structural consistency through its encoder-decoder structure with skip connections, making it an ideal baseline for comparative analysis. Its primary strength lies in stable convergence when paired training data is available, though it may struggle with artistic variations not represented in the training set.

#### 2. **Pix2PixHD: High-Resolution Enhancement**
Developed by NVIDIA Research, Pix2PixHD introduces a multi-scale generator architecture capable of producing high-resolution outputs (up to 2048×1024). The model employs a coarse-to-fine generation strategy, first establishing basic color relationships before refining intricate details. This architecture significantly improves texture preservation and color consistency in complex scenes. The implementation includes instance-level feature normalization and multi-scale discriminators that evaluate image quality at different resolutions, resulting in photorealistic colorizations with exceptional detail preservation.

#### 3. **CycleGAN: Unsupervised Domain Adaptation**
CycleGAN revolutionizes the colorization process by eliminating the need for paired training data through its innovative cycle consistency mechanism. The dual-generator framework learns bidirectional mappings between grayscale and color domains, enforcing consistency through forward and backward translation cycles. This unsupervised approach enables training on unpaired datasets, significantly expanding the potential training data pool. The model's strength lies in its ability to learn stylistic adaptations without explicit supervision, though it requires careful balancing of adversarial and cycle consistency losses to prevent mode collapse.

## 🗂️ Project Structure
```

Manga_Coloring-main/
├── 📓 CycleGan.ipynb # CycleGAN implementation notebook
├── 📓 Pix2Pix.ipynb # Pix2Pix implementation notebook  
├── 📓 pix2pixHD.ipynb # Pix2PixHD implementation notebook
└── 📖 README.md # Project documentation

````

### 📊 Dataset Management
The project utilizes the **MichaelP84/manga-colorization-dataset** from Hugging Face Hub, containing paired grayscale and color manga images. The dataset undergoes systematic preprocessing including:

## 📚 References

### Academic Papers

- Isola, P., et al. (2017). "Image-to-Image Translation with Conditional Adversarial Networks"
- Wang, T. C., et al. (2018). "High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs"
- Zhu, J. Y., et al. (2017). "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"

### Implementation References

- [PyTorch CycleGAN and Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [NVIDIA Pix2PixHD](https://github.com/NVIDIA/pix2pixHD)


---

## Türkçe

### 🇹🇷 

## 📖 Proje Genel Bakış

Bu proje, siyah-beyaz manga resimlerini otomatik olarak renklendirmek için üç farklı son teknoloji Üretken Çekişmeli Ağ (GAN) mimarisini uygular. Sistematik uygulama ve karşılaştırmalı analiz yoluyla, görüntüden görüntüye çeviri alanında farklı GAN yaklaşımlarının etkinliğini, özellikle çizim sanatı korunumu, stil tutarlılığı ve kültürel olarak uygun renk uygulaması gibi manga renklendirmenin benzersiz zorlarını ele alarak gösteriyoruz.

Proje, manga renklendirme görevi için farklı avantajlar sunan Pix2Pix, Pix2PixHD ve CycleGAN modellerinin kapsamlı uygulamalarını sağlıyor. Bu mimarilerden yararlanarak, orijinal çalışmanın sanatsal bütünlüğünü korurken yüksek kaliteli, estetik olarak çekici renklendirmeler üretebilen sağlam işlem hatları oluşturduk.

### 🎯 Temel Hedefler

- **Karşılaştırmalı Analiz**: Manga renklendirme için Pix2Pix, Pix2PixHD ve CycleGAN mimarilerindeki performansı sistematik olarak değerlendirmek
- **Teknik Uygulama**: Tekrarlanabilir eğitim iş akışları ile optimize edilmiş görüntüden görüntüye çeviri işlem hatları geliştirmek
- **Kalite Üretimi**: Sanatsal niyeti ve çizim işi bütünlüğünü koruyan, görsel olarak etkileyici ve kültürel olarak uygun renklendirmeler üretmek
- **Araştırma Katkısı**: Dijital sanat restorasyonu ve geliştirmede GAN uygulamalarının anlaşılmasını ilerletmek

## 🏗️ Mimari Çerçeve

### 🔬 Model Uygulamaları

#### 1. **Pix2Pix: Denetimli Öğrenme Temeli**

Koşullu GAN çerçevesi üzerine inşa edilen Pix2Pix, U-Net üreteci mimarisini PatchGAN ayırıcısı ile birleştirir. Bu denetimli yaklaşım, eşleşmiş eğitim verilerini kullanarak gri tonlamalı ve renkli temsiller arasında doğrudan eşlemeler kurar. Model, atlama bağlantılarına sahip kodlayıcı-kod çözücü yapısı aracılığıyla yapısal tutarlılığı korumada mükemmeldir, bu da onu karşılaştırmalı analiz için ideal bir temel yapar. Ana gücü, eşleşmiş eğitim verisi mevcut olduğunda kararlı yakınsamasında yatar, ancak eğitim setinde temsil edilmeyen sanatsal varyasyonlarla zorlanabilir.

#### 2. **Pix2PixHD: Yüksek Çözünürlüklü Geliştirme**

NVIDIA Araştırma tarafından geliştirilen Pix2PixHD, yüksek çözünürlüklü çıktılar (2048×1024'ye kadar) üretebilen çok ölçekli bir üreteci mimarisi sunar. Model, önce temel renk ilişkilerini kurarak ardından karmaşık detayları hassaslaştıran kaba-ince bir üretim stratejisi kullanır. Bu mimari, karmaşık sahnelerde doku korunumunu ve renk tutarlılığını önemli ölçüde iyileştirir. Uygulama, farklı çözünürlüklerde görüntü kalitesini değerlendiren çok ölçekli ayırıcılar ve örnek seviyesi özellik normalizasyonu içerir, bu da istisnai detay korunumu ile fotorealistik renklendirmelerle sonuçlanır.

#### 3. **CycleGAN: Denetimsiz Alan Uyarlama**

CycleGAN, yenilikçi döngü tutarlılık mekanizması aracılığıyla eşleşmiş eğitim verisi ihtiyacını ortadan kaldırarak renklendirme sürecini devrim niteliğinde değiştirir. İkili üreteci çerçevesi, gri tonlamalı ve renkli alanlar arasında çift yönlü eşlemeler öğrenir, ileri ve geri çeviri döngüleri aracılığıyla tutarlılığı zorlar. Bu denetimsiz yaklaşım, eşleşmemiş veri setlerinde eğitimi mümkün kılar ve potansiyel eğitim veri havuzunu önemli ölçüde genişletir. Modelin gücü, açık denetim olmadan stilistik uyarlamaları öğrenme yeteneğinde yatar, ancak mod çökmesini önlemek için çekişmeli ve döngü tutarlılık kayıplarının dikkatli dengelenmesini gerektirir.

## 🗂️ Proje Yapısı

```
Manga_Coloring-main/
├── 📓 CycleGan.ipynb          # CycleGAN uygulama defteri
├── 📓 Pix2Pix.ipynb           # Pix2Pix uygulama defteri  
├── 📓 pix2pixHD.ipynb         # Pix2PixHD uygulama defteri
└── 📖 README.md               # Proje dokümantasyonu
```

### 📊 Veri Seti Yönetimi

Proje, Hugging Face Hub'dan **MichaelP84/manga-colorization-dataset** veri setini kullanır, eşleşmiş gri tonlamalı ve renkli manga resimleri içerir.

## 📚 Kaynaklar

### Akademik Makaleler

- Isola, P., ve diğ. (2017). "Image-to-Image Translation with Conditional Adversarial Networks"
- Wang, T. C., ve diğ. (2018). "High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs"
- Zhu, J. Y., ve diğ. (2017). "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"

### Uygulama Kaynakları

- [PyTorch CycleGAN ve Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [NVIDIA Pix2PixHD](https://github.com/NVIDIA/pix2pixHD)
