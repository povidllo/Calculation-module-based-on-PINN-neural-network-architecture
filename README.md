# Calculation-module-based-on-PINN-neural-network-architecture

<b>Использовать:</b><br />
python 3.9<br />

<b>Зависимости:</b><br />
fastapi-jsonrpc==3.1.1<br />
fastapi==0.109.2<br />
pybase64==1.3.2<br />
pydantic==2.5.0<br />
Jinja2==3.1.4<br />
torch==2.4.1<br />
numpy==1.21.0<br />
pandas==1.3.0<br />
matplotlib==3.4.2<br />
uvicorn==0.24.0.post1<br />

<b>Как запускать:</b><br />
Запустить скрипт habr_test.py<br />
http://localhost:8010/train - url для обучения модели. Сервер ответит в таком формате {"ok":true,"training":false}, где "ok" значит, что модель обучилась, а "training" значит идет ли обучение в данный момент.<br />
Если параметр не передавать, то сеть будет обучена за 10 шагов. Если же сеть была обучена, то она не переобучится.<br />
Eсли передать параметр в ulr, например так http://localhost:8010/train?N=100, и не идет обучения (т.е. "training":false), сеть будет переобучена с заданным количеством шагов.<br />
http://localhost:8010/pinn - ulr для расчета модели, будет работать только после того как сеть обучилась(т.е. "ok":true).<br />