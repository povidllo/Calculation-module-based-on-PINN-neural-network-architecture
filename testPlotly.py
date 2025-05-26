from fastapi.templating import Jinja2Templates
import matplotlib.pyplot as plt
import numpy as np
import io
import base64


def generate_loss_plot_html():
    epochs = np.arange(1, 11)
    loss = np.random.uniform(0.2, 1.0, size=len(epochs))
    time = np.linspace(0, 1200, len(epochs))
    # Создаем фигуру
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Нижняя ось X — эпохи
    ax1.plot(epochs, loss, marker='o', label='Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_xticks(epochs)
    ax1.grid(True)

    # Верхняя ось X — время
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(epochs)
    ax2.set_xticklabels([f"{int(t)}s" for t in time])
    ax2.set_xlabel('Time')

    plt.title('Loss over Epochs and Time')
    plt.tight_layout()

    # Сохраняем график в буфер в формате JPG
    buffer = io.BytesIO()
    plt.savefig(buffer, format='jpg', dpi=300)
    plt.close(fig)
    buffer.seek(0)

    # Кодируем в base64
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    return img_base64


class InputData:
    def __init__(self, id, default_value):
        self.id = id
        self.default_value = default_value


class OutputData:
    def __init__(self, id, content):
        self.id = id
        self.content = content


class GraphData:
    def __init__(self, id, img_base64):
        self.id = id
        self.img_base64 = img_base64


def main():

    # Jinja-шаблоны
    templates = Jinja2Templates(directory="templates")

    # Элементы
    items = [
        ('tmp_input', InputData('tmp_input', None)),
        ('tmp_label', OutputData("new_label", "hello world!")),
        ('tmp_bokeh', GraphData('graph_1', generate_loss_plot_html()))
    ]

    # Рендерим
    final_html = templates.get_template("html/new test.html").render({
        "items": items
    })

    with open("output.html", "w", encoding="utf-8") as f:
        f.write(final_html)


if __name__ == "__main__":
    main()
