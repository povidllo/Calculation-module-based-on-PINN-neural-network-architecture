async function call_bff_post(path, body_item){
    return await fetch(base_url+path, {
        method: 'POST',
        mode: 'cors',
        credentials: 'include',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(body_item)
    }).then((response) => response.json())
}

async function call_bff_get(path){
    return await fetch(base_url+path, {
        method: 'GET',
        mode: 'cors',
        credentials: 'include',
        headers: {
            'Content-Type': 'application/json',
        },
    }).then((response) => response.json())
}

const handle_load_button = async () => {
    const selectElement = document.getElementById("model_select");
    const selectedModelId = selectElement.value;

    const response = await call_bff_post("load_model", { "stored_item_id": selectedModelId });
}

const handle_load_model_list = async() => {
    value = model_list.value;
    console.log(value);

    const a = await call_bff_post('load_model', {"value": value});

    console.log('response: ', a);
}

const handle_train_button = async (nn_id) => {
    const a = await call_bff_get('train')
}

const handle_run_button = async (nn_id) => {
    const a = await call_bff_post('run', {})
}

const handle_create_button = async () => {
    const desc = document.getElementById('neural_desc').value;

    // Получаем значения гиперпараметров
    const params = {
        'my_model_type': "oscil",
        'my_model_desc': desc,

        // Базовые параметры сети
        'input_dim': parseInt(document.getElementById('input_dim').value) || 1,
        'output_dim': parseInt(document.getElementById('output_dim').value) || 1,
        'hidden_sizes': document.getElementById('hidden_sizes').value ?
            document.getElementById('hidden_sizes').value.split(',').map(x => parseInt(x.trim())) :
            [32, 32, 32],

        // Параметры Фурье
        'Fourier': document.getElementById('fourier').checked,
        'FinputDim': parseInt(document.getElementById('finput_dim').value) || null,
        'FourierScale': parseFloat(document.getElementById('fourier_scale').value) || null,

        // Путь к данным
        'path_true_data': "/data/OSC.npy",

        'epochs': parseInt(document.getElementById('epochs').value) || 100,
        'my_model_type': "oscil"
    };

    const a = await call_bff_post('create_model', params);

    console.log('pass: ', a);
};

const old_handle_update_params_button = async () => {
    const params = {
        'epochs': parseInt(document.getElementById('epochs').value) || 100,
        'optimizer': document.getElementById('optimizer').value || "Adam",
        'optimizer_lr': parseFloat(document.getElementById('optimizer_lr').value) || 0.001,
        // 'my_model_type': "oscil"
    };

    const response = await call_bff_post('update_train_params', params);
    console.log('update params response:', response);
};

const handle_update_params_button = async () => {
    const optimizer = document.getElementById('optimizer').value || "Adam";
    const params = {};
    
    for (const [key, _] of Object.entries(optimizerParams[optimizer])) {
        const input = document.getElementById(`optimizer-${key}`);
        if (input.type === 'checkbox') {
            params[key] = input.checked;
        } else if (key === 'betas') {
            params[key] = input.value.split(',').map(x => parseFloat(x.trim()));
        } else {
            params[key] = parseFloat(input.value);
        }
    }

    const updateData = {
        epochs: parseInt(document.getElementById('epochs').value) || 100,
        method: optimizer,
        params: params
    };
    
    const response = await call_bff_post('update_train_params', updateData);
    console.log('update params response:', response);
};

const handle_clear_database = async () => {
    if (confirm('Удалить все записи в базе данных?')) {
        const a = await call_bff_get('clear_database');
    }
};

const handle_send_command = async () => {
    const command = document.getElementById('command_input').value;

    if (command === "train")
        await call_bff_get('train')
    else if (command === "run")
        await call_bff_post('run', {})
}

const optimizerParams = {
    'Adam': {
        'lr': { default: 0.001, type: 'number', label: 'Learning Rate' },
        'betas': { default: '0.9,0.999', type: 'text', label: 'Betas (через запятую)' },
        'eps': { default: 1e-8, type: 'number', label: 'Epsilon' },
        'weight_decay': { default: 0, type: 'number', label: 'Weight Decay' }
    },
    'SGD': {
        'lr': { default: 0.01, type: 'number', label: 'Learning Rate' },
        'momentum': { default: 0.9, type: 'number', label: 'Momentum' },
        'weight_decay': { default: 0, type: 'number', label: 'Weight Decay' },
        'nesterov': { default: false, type: 'checkbox', label: 'Nesterov' }
    },
    'RMSprop': {
        'lr': { default: 0.01, type: 'number', label: 'Learning Rate' },
        'alpha': { default: 0.99, type: 'number', label: 'Alpha' },
        'eps': { default: 1e-8, type: 'number', label: 'Epsilon' },
        'momentum': { default: 0, type: 'number', label: 'Momentum' }
    },
    'Adagrad': {
        'lr': { default: 0.01, type: 'number', label: 'Learning Rate' },
        'lr_decay': { default: 0, type: 'number', label: 'Learning Rate Decay' },
        'weight_decay': { default: 0, type: 'number', label: 'Weight Decay' }
    }
};

function updateOptimizerParams() {
    const optimizer = document.getElementById('optimizer').value;
    const paramsContainer = document.getElementById('optimizer-params');
    paramsContainer.innerHTML = '';

    const params = optimizerParams[optimizer];
    for (const [key, value] of Object.entries(params)) {
        const paramDiv = document.createElement('div');
        paramDiv.className = 'param-group';

        const label = document.createElement('label');
        label.htmlFor = `optimizer-${key}`;
        label.textContent = value.label;

        const input = document.createElement('input');
        input.id = `optimizer-${key}`;
        input.type = value.type;
        input.value = value.default;
        input.placeholder = value.label;

        if (value.type === 'checkbox') {
            input.checked = value.default;
        }

        paramDiv.appendChild(label);
        paramDiv.appendChild(input);
        paramsContainer.appendChild(paramDiv);
    }
}

window.onload = async () =>{
    const selectElement = document.getElementById("model_select");

    if (selectElement && selectElement.options.length > 0) {

        const selectedModelId = selectElement.value;
        const response = await call_bff_post("load_model", { "stored_item_id": selectedModelId });
    }

    const secondSelectElement = document.getElementById("optimizer");

    if (secondSelectElement && secondSelectElement.options.length > 0) {
        updateOptimizerParams();
    }

    console.log("Страница полностью загружена!");
};


var base_url = "http://localhost:8010/";




//---------------------------------------------//




var ws_ping = new WebSocket(`ws://localhost:8010/ws_ping`);

ws_ping.onmessage = function(event)
{
    const res = JSON.parse(event.data);
    console.log("Пришло сообщение:", res); // Проверка

    if (res.msg_type == 'jinja_tmpl') {
        if (res.data[0] === 'model_desc') {
            console.log("Обновляем модели:", res.data[1]); // Проверка
            document.getElementById(res.data[0]).innerHTML = res.data[1];

            const select = document.getElementById("model_select"); // Замените на ваш ID
            select.selectedIndex = select.options.length - 1;

            handle_load_button();
        }

        if (res.data[0] === 'chat_container_id') {
            console.log("Выводим картинку:", res.data[1]); // Проверка
            document.getElementById(res.data[0]).innerHTML = res.data[1];
        }
    }

    // if (res.msg_type == 'hahaha') {
        
    //     const opt = document.createElement("option");
    //     opt.value = res.data[0];
    //     opt.text = res.data[0];

    //     model_list.add(opt, null);
    // }
}

// create_btn.addEventListener('click', handle_create_button);
// document.getElementById('update_params_btn').addEventListener('click', handle_update_params_button);
// clear_db_btn.addEventListener('click', handle_clear_database);