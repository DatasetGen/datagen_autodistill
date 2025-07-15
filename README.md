

# 🧪 Datagen AutoDistill

**Datagen AutoDistill** es un microservicio basado en **FastAPI** que genera anotaciones sintéticas de imágenes utilizando modelos de segmentación como **Grounded SAM** y **Grounding DINO**, integrando la librería [Autodistill](https://github.com/roboflow/autodistill) desarrollada por Roboflow.

Este servicio forma parte de la suite **Datagen**, una plataforma modular para la **generación, anotación, visualización y gestión de datasets sintéticos** diseñados para el entrenamiento de modelos de aprendizaje automático.

---

## 🎯 ¿Qué hace este servicio?

* Recibe una imagen codificada en base64 y un prompt.
* Utiliza un modelo AutoDistill (por ejemplo, Grounded SAM) para detectar y segmentar objetos.
* Devuelve las anotaciones generadas en formato de polígonos o bounding boxes.
* Permite etiquetar automáticamente grandes volúmenes de imágenes.

---

## ⚙️ Características

* Compatible con modelos de segmentación como:

  * **Grounded SAM**
  * **Grounding DINO**
* Convierte automáticamente la salida del modelo en anotaciones estructuradas.
* Diseñado para integrarse con pipelines automatizados como los de **Datagen Orchestrator**.

---

## 🚀 Instalación local

1. Clona el repositorio y entra a la carpeta:

   ```bash
   git clone <repo-url>
   cd datagen-autodistill
   ```

2. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

3. Ejecuta el servidor FastAPI:

   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8001
   ```

El servicio estará disponible en:
`http://localhost:8001`

---

## 🐳 Uso con Docker

1. Construye la imagen:

   ```bash
   docker build -t datagen-autodistill .
   ```

2. Ejecuta el contenedor:

   ```bash
   docker run -p 8001:8001 datagen-autodistill
   ```

---

## 📡 Endpoint de la API

### `POST /generate_images/image_variants/`

Este endpoint acepta una imagen codificada en base64 junto con parámetros para la generación de anotaciones.

#### 📥 Cuerpo de la solicitud (JSON):

```json
{
  "number_of_images": 1,
  "image": "<base64_image>",
  "aspect_ratio": "1:1",
  "prompt": "example",
  "negative_prompt": "",
  "annotation_model": "grounded_sam",
  "strength": 1.0,
  "labels": [
    { "id": 1, "name": "object", "prompt": "object" }
  ]
}
```

#### 📤 Respuesta:

Devuelve la imagen original junto con las anotaciones generadas.

---

## 🧩 Integración con el ecosistema Datagen

| Proyecto                 | Relación con AutoDistill                                                                       |
| ------------------------ | ---------------------------------------------------------------------------------------------- |
| **Datagen Orchestrator** | Utiliza este servicio para aplicar anotaciones automáticas a imágenes generadas con Diffusers. |
| **Datagen Segmentators** | Alternativa o complemento para tareas de segmentación rápida.                                  |
| **Datagen Backend**      | Puede almacenar las anotaciones generadas por AutoDistill como parte del dataset.              |
| **Datagen Frontend**     | Posible integración futura para anotación asistida visualmente.                                |

---

## 🧪 Pruebas

Este servicio cuenta con pruebas automatizadas. Para ejecutarlas:

```bash
pytest -q
```

---

## 🤝 Contribuciones

¡Toda colaboración es bienvenida!
Puedes forquear el repositorio y enviar un Pull Request con nuevas funciones, modelos soportados o mejoras generales.

---

## 🔗 Recursos útiles

* [Autodistill (Roboflow)](https://github.com/roboflow/autodistill)
* [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
* [Segment Anything (SAM)](https://segment-anything.com/)
* [FastAPI](https://fastapi.tiangolo.com/)

