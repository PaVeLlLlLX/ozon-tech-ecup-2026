# Образ решения: базовый образ организаторов плюс библиотеки, которых в нём нет.
#
# Проверка базового образа (odsai/ecup26-quality-baseline:1.0, 6.16 ГБ в сжатом виде
# при лимите 15) показала, что в нём отсутствуют easyocr, peft, cv2 и все бустинги.
# Без них недоступны три ветки решения сразу: распознавание текста на фото,
# LoRA-адаптеры над VLM и градиентный бустинг.
#
# ⚠ easyocr тянет за собой torch и torchvision. Ставить его с зависимостями нельзя:
# pip заменит собранный под CUDA torch на сборку из PyPI и решение потеряет видеокарту.
# Поэтому сам easyocr ставится без зависимостей, а его настоящие зависимости —
# отдельным списком, уже без torch.
#
# Веса easyocr кладутся В ОБРАЗ, а не в архив: в контейнере нет интернета, скачать их
# на месте нельзя, а бюджет архива в 5 ГБ тратить незачем.
#
# Сборка и публикация:
#   docker build -t <логин>/ecup26-quality:1.0 -f docker/Dockerfile .
#   docker push <логин>/ecup26-quality:1.0
# После публикации имя образа прописывается в docker/metadata.json.

FROM odsai/ecup26-quality-baseline:1.0

# Питон в образе помечен как системный (PEP 668), и pip отказывается в него ставить
# без явного разрешения. Контейнер одноразовый и служит одному решению — ломать нечего.
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    EASYOCR_MODULE_PATH=/opt/easyocr

# Бустинги и LoRA: обычные зависимости, torch не трогают.
RUN pip install --no-cache-dir \
        peft==0.19.1 \
        catboost==1.2.10 \
        lightgbm==4.7.0 \
        xgboost==3.2.0

# Распознавание текста. Сначала зависимости easyocr без torch, затем он сам без них.
RUN pip install --no-cache-dir \
        opencv-python-headless \
        scikit-image \
        python-bidi \
        shapely \
        pyclipper \
        ninja \
 && pip install --no-cache-dir --no-deps easyocr==1.7.2

# Веса распознавателя: детектор текста и кириллическая модель.
COPY artifacts/easyocr/ /opt/easyocr/model/
# ⚠ easyocr безусловно делает mkdir для каталога пользовательских сетей, а файловая
# система у проверяющей системы только для чтения — отсутствующий каталог означает
# «Read-only file system» ещё до первой картинки. Создаём его заранее, пустым.
RUN mkdir -p /opt/easyocr/user_network

# Проверка сборки: если чего-то не хватает, узнать об этом надо здесь, а не на сдаче.
RUN python -c "\
import torch, transformers, sklearn, peft, easyocr, catboost, lightgbm, xgboost, cv2;\
print('torch', torch.__version__, '| cuda-сборка:', torch.version.cuda);\
print('transformers', transformers.__version__, '| sklearn', sklearn.__version__);\
print('peft', peft.__version__, '| easyocr ок | catboost', catboost.__version__);\
print('lightgbm', lightgbm.__version__, '| xgboost', xgboost.__version__)"

WORKDIR /app
