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
};

const handle_load_model_list = async() => {
    value = model_list.value;
    console.log(value);

    call_bff_post('load_model', {"value": value});
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
        'path_true_data': "/data/OSC.npy"
    };

    const a = await call_bff_post('create_model', params);
    console.log('pass: ', a);
};

const handle_update_params_button = async () => {
    const params = {
        'epochs': document.getElementById('epochs').value || 100,
        'optimizer': document.getElementById('optimizer').value || "Adam",
        'optimizer_lr': parseFloat(document.getElementById('optimizer_lr').value) || 0.001,
        'my_model_type': "oscil"
    };

    const response = await call_bff_post('update_train_params', params);
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

var base_url = "http://localhost:8010/";

var ws_ping = new WebSocket(`ws://localhost:8010/ws_ping`);

ws_ping.onmessage = function(event)
{
    const res = JSON.parse(event.data);
    console.log("Пришло сообщение:", res); // Проверка

    if (res.msg_type == 'jinja_tmpl') {
        if (res.data[0] === 'model_desc') {
            console.log("Обновляем модели:", res.data[1]); // Проверка
            document.getElementById(res.data[0]).innerHTML = res.data[1];
        }
    }

    if (res.msg_type == 'add_model_to_list') {
        
        const opt = document.createElement("option");
        opt.value = res.data[0];
        opt.text = res.data[0];

        model_list.add(opt, null);
    }

    if (res.msg_type == 'start_model_list_conf') {

        for (const item of res.data) {
            const opt = document.createElement("option");
            opt.value = item;
            opt.text = item;

            model_list.add(opt, null);
        }
    }
}

// create_btn.addEventListener('click', handle_create_button);
// document.getElementById('update_params_btn').addEventListener('click', handle_update_params_button);
// clear_db_btn.addEventListener('click', handle_clear_database);