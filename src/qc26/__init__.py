"""Пакет решения. При импорте глушит два известных бесполезных предупреждения.

⚠ Почему глушение стоит здесь, а не в каждом скрипте. Мы дважды забывали позвать
глушилку в новом файле, и терминал заливало тысячами строк — по строке на каждый
вызов процессора. Проверка, перечислявшая файлы по именам, каждый раз отставала
на один. Правило «не забыть позвать» не работает; работает «позвать нельзя забыть».
"""
import logging as _logging

# ⚠ Глушим ТОЧЕЧНО, два сообщения, а не весь канал. Именно из предупреждений
# transformers мы узнали, что у Qwen3.5 недоступен быстрый путь линейного внимания —
# и это спасло от прогона на 2000 руб мимо бюджета. Затыкать всё значит однажды
# не увидеть такое же важное.
#
#   1. «Kwargs passed to processor.__call__ have to be in processor_kwargs dict» —
#      печатается на КАЖДЫЙ вызов процессора. Совет из него ПРОВЕРЕН и вреден:
#      с processor_kwargs процессор возвращает списки вместо выровненного тензора.
#      Мы намеренно зовём по-старому, защёлка стоит в check_batch_padded.
#   2. «torch_dtype is deprecated» — один раз на загрузку модели.
_DROP = ("processor_kwargs", "torch_dtype` is deprecated")


class _KnownNoise(_logging.Filter):
    def filter(self, record):
        return not any(d in str(record.getMessage()) for d in _DROP)


def quiet_known_noise() -> None:
    """Ставит фильтр. Идемпотентна: повторный вызов ничего не ломает."""
    f = _KnownNoise()
    for name in ("transformers", "transformers.processing_utils",
                 "transformers.modeling_utils"):
        lg = _logging.getLogger(name)
        if not any(isinstance(x, _KnownNoise) for x in lg.filters):
            lg.addFilter(f)
    try:
        from transformers.utils import logging as _hf

        for lg in (_hf.get_logger(), _hf._get_library_root_logger()):
            if not any(isinstance(x, _KnownNoise) for x in lg.filters):
                lg.addFilter(f)
    except Exception:
        pass


quiet_known_noise()
