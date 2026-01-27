# ⚡ ComfyUI-Demucs-Pro 🎵

¡Bienvenido a **ComfyUI-Demucs-Pro**! Este es el nodo definitivo para la separación de fuentes de audio en ComfyUI, potenciado por la tecnología de vanguardia **Meta Demucs v4 (Hybrid Transformer)**. 🚀

Diseñado para profesionales y entusiastas del audio, este nodo permite extraer con una fidelidad asombrosa hasta 6 pistas individuales: **Voces, Batería, Bajo, Otros, Guitarra y Piano**.

## 🌟 Características Principales

- **Tecnología Demucs v4**: Acceso a los modelos más recientes, incluyendo `htdemucs`, `htdemucs_ft` y el potente `htdemucs_6s`.
- **Separación de 6 Stems**: No te conformes con 4. Obtén pistas separadas para Guitarra y Piano con modelos compatibles.
- **⚡ Optimización Ampere (RTX 3090/4090)**:
  - **Bfloat16 Precision**: Procesamiento más rápido con menor uso de memoria sin sacrificar calidad en arquitecturas modernas.
  - **Model Pinning**: Uso inteligente de la RAM para cambios de modelo instantáneos mediante memoria anclada (pinned memory).
  - **Aceleración CUDA**: Aprovecha al máximo los núcleos Tensor de tu GPU.
- **Gestión Inteligente de Memoria**: Sistema de `split` automático para procesar audios largos sin errores de memoria (OOM).
- **Resampleado Automático**: Integración con `torchaudio` para manejar cualquier frecuencia de muestreo de entrada de forma transparente.

## 🛠️ Instalación

### Opción 1: ComfyUI Manager (Recomendado)
1. Abre el **ComfyUI Manager**.
2. Busca `ComfyUI-Demucs-Pro`.
3. Haz clic en **Install**.
4. Reinicia ComfyUI y ¡listo! ⚡

### Opción 2: Instalación Manual
1. Navega a tu carpeta de `custom_nodes`:
   ```bash
   cd ComfyUI/custom_nodes
   ```
2. Clona el repositorio:
   ```bash
   git clone https://github.com/usuario/ComfyUI-Demucs-Pro
   ```
3. Instala las dependencias necesarias:
   ```bash
   pip install -r requirements.txt
   ```
4. Reinicia tu servidor ComfyUI.

## 🎮 Parámetros del Nodo

- **audio**: 🎵 Tu entrada de audio.
- **model**: Selecciona el cerebro del proceso. Recomendamos `htdemucs_6s` para máxima versatilidad (6 pistas).
- **device**: `cuda` para velocidad rayo ⚡ o `cpu` si prefieres ir con calma.
- **shifts**: Calidad vs. Velocidad. Valores entre 1-5 son ideales para la mayoría de los casos.
- **overlap**: Solapamiento de ventanas. 0.25 es el punto dulce recomendado.
- **split**: Actívalo para ahorrar VRAM en audios de larga duración. ¡Imprescindible para GPUs de menos de 8GB!

## 🚀 Optimización para Entusiastas

Este nodo ha sido calibrado específicamente para sistemas de alto rendimiento con **128GB de RAM** y GPUs **RTX 3090/4090**. Utilizamos un sistema de caché global (`_MODEL_CACHE`) para que el intercambio entre modelos sea prácticamente instantáneo una vez cargados por primera vez en la memoria del sistema.

## ✉️ Soporte y Contribuciones

¿Encontraste un bug? ¿Tienes una idea genial? 💡
Abre un *Issue* o un *Pull Request*. ¡Estamos construyendo el futuro del audio en ComfyUI juntos!

---
*Creado con ❤️ por un Senior Python Developer apasionado por la IA Multimedia. ¡Disfruta del silencio (o del sonido)!* ⚡
