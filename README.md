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
http://localhost:8010/train - url для обучения модели. Когда ответит {"ok":true,"training":true} значит модель обучилась.<br />
http://localhost:8010/pinn - ulr для расчета модели, будет работать только после того как сеть обучилась.<br />