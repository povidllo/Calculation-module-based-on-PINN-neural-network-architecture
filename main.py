from bokeh.plotting import figure, curdoc
from bokeh.models import Button, ColumnDataSource, Div
from bokeh.layouts import column, row, Spacer
from bokeh.models.widgets import Select
import numpy as np
import time
import random

# Инициализация данных
button = Button(label="Обучить модель")  # Кнопка запуска обучения
delete_button = Button(label="Удалить", visible=False)  # Кнопка удаления графиков
select = Select(title="", value=None, options=[], visible=False)

# Хранение графиков и источников данных
graphs = {}  # Словарь для хранения графиков с названиями
sources = {}  # Словарь для хранения источников данных для каждого графика
current_plot = None  # Переменная для хранения текущего графика на экране


# Функция, которая выполняет расчеты и возвращает данные
def long_running_function():
    time.sleep(2)  # Имитируем долгую операцию
    x = np.linspace(0, 10, 100)
    y = np.sin(x) * np.random.rand()
    return x, y


# Функция, которая возвращает случайное дробное значение для отображения над графиком
def generate_fractional_value():
    return round(random.uniform(0, 1), 3)


# Обработчик нажатия на кнопку обучения модели
def add_graph():
    global current_plot

    # Получаем новые данные и случайное дробное значение
    x, y = long_running_function()
    fractional_value = generate_fractional_value()

    # Создаем новый источник данных
    source = ColumnDataSource(data=dict(x=x, y=y))

    # Создаем новый график
    plot_number = len(graphs) + 1
    plot = figure(title=f"Обучение {plot_number}", x_axis_label="x", y_axis_label="y")
    plot.line('x', 'y', source=source)

    # Сохраняем график и источник в словари
    graph_name = f"Обучение {plot_number}"
    graphs[graph_name] = (plot, fractional_value)  # Сохраняем график и его значение
    sources[graph_name] = source

    # Обновляем список графиков в виджете select
    select.options = list(graphs.keys())
    select.value = graph_name  # Автоматически выбираем последний созданный график

    # Показ элементов: select и кнопка удаления графиков
    select.visible = True
    delete_button.visible = True

    # Показываем последний добавленный график
    show_graph(graph_name)


# Функция для отображения выбранного графика
def show_graph(graph_name):
    global current_plot

    # Удаляем текущий график и дробное значение, если они уже отображаются
    if current_plot in right_layout.children:
        right_layout.children.remove(current_plot)
    if len(right_layout.children) > 0:
        right_layout.children.pop(0)

    # Получаем график и дробное значение по названию
    plot, fractional_value = graphs[graph_name]
    fractional_value_div = Div(text=f"Loss: {fractional_value}", width=200)

    # Добавляем дробное значение и график в макет
    current_plot = plot
    right_layout.children.extend([fractional_value_div, plot])


# Обработчик для удаления выбранного графика
def delete_graph():
    global current_plot
    selected_graph = select.value

    # Проверяем, что график выбран
    if selected_graph and selected_graph in graphs:
        # Удаляем график и данные из словарей
        del graphs[selected_graph]
        del sources[selected_graph]

        # Обновляем список графиков
        select.options = list(graphs.keys())
        select.value = None  # Сбрасываем выбор

        # Очищаем правую часть макета
        right_layout.children.clear()
        current_plot = None

        # Скрываем кнопки, если графиков больше нет
        if not graphs:
            select.visible = False
            delete_button.visible = False


# Привязываем обработчики
button.on_click(add_graph)
delete_button.on_click(delete_graph)
select.on_change('value', lambda attr, old, new: show_graph(new) if new else None)

# Макет левой и правой части
left_layout = column(button, select, delete_button)
right_layout = column()

# Полный макет с выравниванием и пустым пространством
layout = row(Spacer(width=200), left_layout, Spacer(width=50), right_layout, Spacer(width=200))
curdoc().add_root(layout)
