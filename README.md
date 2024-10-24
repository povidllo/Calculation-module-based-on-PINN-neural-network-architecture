# Calculation-module-based-on-PINN-neural-network-architecture

<b>Использовать:</b><br />
python 3.9<br />

<b>Зависимости:</b><br />
pip install -r requirements.txt<br />

<b>Как запускать:</b><br />
Запустить скрипт habr_test.py<br />
http://localhost:8010/train - url для обучения модели. Сервер ответит в таком формате {"ok":true,"training":false}, где "ok" значит, что модель обучилась, а "training" значит идет ли обучение в данный момент.<br />
Если параметр не передавать, то сеть будет обучена за 10 шагов. Если же сеть была обучена, то она не переобучится.<br />
Eсли передать параметр в ulr, например так http://localhost:8010/train?N=100, и не идет обучения (т.е. "training":false), сеть будет переобучена с заданным количеством шагов.<br />
http://localhost:8010/pinn - ulr для расчета модели, будет работать только после того как сеть обучилась(т.е. "ok":true).<br />