# Calculation-module-based-on-PINN-neural-network-architecture

Использовать:
python 3.9

Зависимости:
fastapi-jsonrpc==3.1.1
fastapi==0.109.2
pybase64==1.3.2
pydantic==2.5.0
Jinja2==3.1.4
torch==2.4.1
numpy==1.21.0
pandas==1.3.0
matplotlib==3.4.2
uvicorn==0.24.0.post1

Как использовать:
http://localhost:8010/train - url для обучения модели. Когда ответит {"ok":true,"training":true} значит модель обучилась.
http://localhost:8010/pinn - ulr для расчета модели, будет работать только после того как сеть обучилась.