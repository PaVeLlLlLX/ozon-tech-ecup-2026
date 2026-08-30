"""Отбор в редкой категории — правилом рекорда, а не долей.

    python patch_select.py

⚠⚠ Зачем. Рекорд vlm_blend_rare2_a_t24 (публичные 0.9168088205) отбирал в категории
«Легковоспламеняющиеся» НЕ по доле, а по порогу 0.9699, применённому к квантилю
относительно ЗАФИКСИРОВАННОГО эталонного распределения из 1101 значения. Это
абсолютный критерий: «оценка выше 96.99% эталона». Доля — критерий относительный,
«верхние 3% ЭТОЙ выборки». На публичной выборке оба дают 22 товара случайно, на
приватной с другим составом они разойдутся.

⭐ Перенос точен и бесплатен. Рекорд считал
        sigmoid(logsumexp(логиты «Да») - logsumexp(логиты «Нет»)),
новый код — ровно то же выражение БЕЗ последнего sigmoid. Значит оценка рекорда
получается из новой одним преобразованием, и эталон с порогом переносятся как есть.

⚠ Порог 0.9699 отсекает всё выше 33-го сверху значения эталона, то есть выше
вероятности 0.000804. Распределение сильно скошено: почти вся выборка у нуля.

Вторая правка — освобождение памяти между адаптерами. Смоук упал с 22.35 ГБ занятых:
в памяти висели три модели разом, потому что после score_frame ссылки не собирались.
На H100 в 80 ГБ это не упало бы, но держать три модели незачем нигде.
"""
from pathlib import Path

RUN = Path(__file__).resolve().parent / "build/run.py"

OLD_SELECT = '''            raw_rate = sub_cfg.get("vlm_rate", 0.034)
            if isinstance(raw_rate, dict):'''

NEW_SELECT = '''            # ⚠⚠ ОТБОР ПРАВИЛОМ РЕКОРДА. В «Легковоспламеняющихся» решение, давшее
            # 0.9168088205, брало не долю, а порог 0.9699 на КВАНТИЛЕ относительно
            # эталонного распределения из 1101 значения, сохранённого при обучении.
            # Критерий абсолютный: «оценка выше 96.99% эталона». Доля — относительный,
            # «верхние 3% этой выборки»; на публичной они совпали случайно, на приватной
            # разойдутся. Эталон и порог едут в архиве, шкала приводится к той же.
            sel_raw = sub_cfg.get("vlm_select", "rate")
            sel_c = str(sel_raw.get(c, "rate") if isinstance(sel_raw, dict) else sel_raw)
            if sel_c == "calibrated":
                import json as _json

                cal_raw = sub_cfg.get("vlm_calibration")
                cal_p = cal_raw.get(c) if isinstance(cal_raw, dict) else cal_raw
                if not cal_p:
                    raise SystemExit(f"для «{c}» задан отбор по эталону, но эталона нет")
                cal = _json.loads(resolve_path(cal_p).read_text(encoding="utf-8"))
                ref = np.asarray(cal["эталон"], dtype=np.float64)
                thr_c = float(cal["порог"])
                if not np.all(np.diff(ref) >= 0):
                    raise SystemExit("эталон не отсортирован — квантиль посчитается неверно")
                # Шкала рекорда: sigmoid от логит-разности. Новый код отдаёт саму
                # логит-разность, поэтому sigmoid возвращаем здесь, во float32 — ровно
                # той точности, в которой эталон и строился.
                z = np.asarray(scores, dtype=np.float32)[m]
                prob = (1.0 / (1.0 + np.exp(-z.astype(np.float32)))).astype(np.float64)
                left = np.searchsorted(ref, prob, side="left")
                right = np.searchsorted(ref, prob, side="right")
                q = (left + right) / (2.0 * len(ref))
                blended[pos[m]] = q
                loaded.thresholds[c] = thr_c
                n_sel = int((q >= thr_c).sum())
                print(f"отбор по эталону рекорда: порог {thr_c}, эталон {len(ref)} "
                      f"значений, отобрано {n_sel} из {int(m.sum())}", flush=True)
                print(f"   оценка: логит-разность от {float(z.min()):+.2f} до "
                      f"{float(z.max()):+.2f} → вероятность от {prob.min():.6f} до "
                      f"{prob.max():.6f}; эталон от {ref[0]:.6f} до {ref[-1]:.6f}",
                      flush=True)
                if n_sel == 0:
                    raise SystemExit(
                        f"по эталону в «{c}» не отобрано ни одного товара — "
                        f"шкала оценки не совпала с эталоном")
                continue

            raw_rate = sub_cfg.get("vlm_rate", 0.034)
            if isinstance(raw_rate, dict):'''

OLD_FREE = '''                scores[m] = s_c
                print(f"категория «{c}»: {int(m.sum())} товаров адаптером {ap_c.name}, "
                      f"оценки от {float(s_c.min()):+.2f} до {float(s_c.max()):+.2f}",
                      flush=True)'''

NEW_FREE = '''                scores[m] = s_c
                print(f"категория «{c}»: {int(m.sum())} товаров адаптером {ap_c.name}, "
                      f"оценки от {float(s_c.min()):+.2f} до {float(s_c.max()):+.2f}",
                      flush=True)
                # ⚠ Освобождаем карту перед следующим адаптером. Без этого смоук упал
                # с 22.35 ГБ занятых: score_frame держит модель до сборки мусора, и
                # три загрузки подряд жили одновременно.
                import gc

                gc.collect()
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass'''


def main() -> None:
    s = RUN.read_text(encoding="utf-8")
    for old, new, what in ((OLD_SELECT, NEW_SELECT, "отбор по эталону"),
                           (OLD_FREE, NEW_FREE, "освобождение памяти")):
        if old not in s:
            raise SystemExit(f"не нашёл место для правки «{what}»")
        s = s.replace(old, new, 1)
    RUN.write_text(s, encoding="utf-8")
    print("правки применены: отбор по эталону + освобождение памяти")


if __name__ == "__main__":
    main()
