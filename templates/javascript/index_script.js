const BASE_URL = 'http://localhost:8010/';

// --- Утилиты для fetch-запросов ---
const fetchJSON = async (url, options) => {
    const response = await fetch(url, {
        mode: 'cors',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        ...options,
    });
    return response.json();
};

const callBffPost = (path, body) => fetchJSON(`${BASE_URL}${path}`, {
    method: 'POST',
    body: JSON.stringify(body),
});

const callBffGet = (path) => fetchJSON(`${BASE_URL}${path}`, {
    method: 'GET',
});

// --- Обработчики событий ---
const handleLoadButton = async () => {
    const selectedModelId = document.getElementById('model_select')?.value;
    if (selectedModelId) {
        await callBffPost('load_model', { stored_item_id: selectedModelId });
    }
};

const handleLoadModelList = async () => {
    const value = model_list.value;
    console.log(value);
    const response = await callBffPost('load_model', { value });
    console.log('response:', response);
};

const handleTrainButton = async () => {
    await callBffGet('train');
};

const handleRunButton = async () => {
    await callBffPost('run', {});
};

const handleCreateButton = async () => {
    const desc = document.getElementById('neural_desc')?.value;
    const model_type = document.getElementById('equation_type').value || 'oscil';

    const getValue = id => parseInt(document.getElementById(id)?.value) || 1;
    const getOptional = (id, parser) => {
        const val = document.getElementById(id)?.value;
        return val ? parser(val) : null;
    };

    const hiddenSizes = document.getElementById('hidden_sizes').value ?
        document.getElementById('hidden_sizes').value.split(',').map(x => parseInt(x.trim())) :
        [32, 32, 32];

    const params = {
        my_model_type: model_type,
        my_model_desc: desc,
        input_dim: getValue('input_dim'),
        output_dim: getValue('output_dim'),
        hidden_sizes: hiddenSizes,
        Fourier: document.getElementById('fourier')?.checked || false,
        FinputDim: getOptional('finput_dim', parseInt),
        FourierScale: getOptional('fourier_scale', parseFloat),
        path_true_data: '/data/OSC.npy',
        epochs: getValue('epochs'),
    };

    const response = await callBffPost('create_model', params);
    console.log('Model created:', response);
};

const handleUpdateParamsButton = async () => {
    const optimizer = document.getElementById('optimizer')?.value || 'Adam';
    const params = {};
    optimizer_params_list.forEach((item) => {
        const input = document.getElementById(item);
        if (input.type === 'checkbox') {
            params[item] = input.checked;
        } else if (item === 'betas') {
            params[item] = input.value.split(',').map(x => parseFloat(x.trim()));
        } else {
            params[item] = parseFloat(input.value);
        }
    })

    const updateData = {
        epochs: parseInt(document.getElementById('epochs')?.value) || 100,
        method: optimizer,
        params,
    };

    const response = await callBffPost('update_train_params', updateData);
    console.log('Update params response:', response);
};

const handleClearDatabase = async () => {
    if (confirm('Удалить все записи в базе данных?')) {
        await callBffGet('clear_database');
    }
};

const handleSendCommand = async () => {
    const command = document.getElementById('command_input')?.value;
    if (command === 't') {
        await callBffGet('train');
    } else if (command === 'r') {
        await callBffPost('run', {});
    } else if (command === 'g') {
        await callBffGet('show_graph')
    } else if (command === 'records') {
        await callBffGet("print_rec")
    } else if (command === 'chat') {
        await callBffGet("inc_chat")
    }
};

var optimizer_params_list;

// --- Обновление параметров формы ---
const updateOptimizerParams = async () => {
    const val = document.getElementById("optimizer").value;

    const response = await callBffPost("update_optimizer_params", {optimizer_name: val});
    optimizer_params_list = response.list_of_id;
    console.log("response:", response);
};

// --- Инициализация при загрузке страницы ---
window.addEventListener('DOMContentLoaded', async () => {
    const modelSelect = document.getElementById('model_select');
    if (modelSelect?.options.length) {
        await callBffPost('load_model', { stored_item_id: modelSelect.value });
    }

    if (document.getElementById('optimizer')?.options.length) {
        updateOptimizerParams();
    }

    console.log('Страница полностью загружена!');
});

// --- WebSocket ---
const wsPing = new WebSocket('ws://localhost:8010/ws_ping');

wsPing.onmessage = ({ data }) => {
    const res = JSON.parse(data);
    console.log('Пришло сообщение:', res);

    if (res.msg_type === 'jinja_tmpl') {
        const [id, html] = res.data;
        const element = document.getElementById(id);

        if (id === 'model_desc') {
            console.log('Обновляем модели:', html);
            element.innerHTML = html;

            const select = document.getElementById('model_select');
            select.selectedIndex = select.options.length - 1;

            handleLoadButton();
        }

        if (id === 'chat_container_id') {
            console.log('Выводим картинку:', html);
            element.innerHTML = html;
        }

        if (id === 'optimizer-params') {
            console.log('Change optimizer param in ui', html);
            element.innerHTML = html;
        }
    }
};
