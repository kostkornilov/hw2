RegBench: 200 эпох, train 1k, один seed; best checkpoint по validation loss.

| Method | Train s/epoch | Validation s/epoch | 200 epochs, h | Test accuracy | Test TVD | Last-token accuracy | Last-token TVD |
|---|---:|---:|---:|---:|---:|---:|---:|
| softmax | 9.00 | 3.73 | 0.89 | 93.39% | 34.31% | 98.00% | 29.63% |
| epanechnikov | 11.69 | 6.05 | 1.17 | 93.62% | 33.26% | 98.00% | 27.84% |
| relumax | 12.01 | 4.64 | 1.11 | 93.49% | 36.87% | 98.20% | 32.65% |

Прогноз включает validation на каждой эпохе, сохранение best/last и финальную оценку.
Первую эпоху исключаем из медианы; scheduler остаётся рассчитанным на 200 эпох.
Суммарный прогноз трёх методов: 3.17 ч вычислений.
Accuracy и TVD пересчитаны по всем позициям сохранённых тестовых предсказаний
по формулам авторского plot_ICL.py; это не отдельная last-token метрика.
