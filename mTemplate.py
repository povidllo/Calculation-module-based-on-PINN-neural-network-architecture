from mongo_schemas import *

mtemplate = lambda gscripts, gdivs_left, gdivs_right: """
<!DOCTYPE html>
<html lang="en">
    <head>
        <style>
            .row {
              display: flex;
            }

            .infobox {
              position: fixed;
              position-anchor: --myAnchor;
              position-area: top;
            }
            .input_box {
              width: 400px;
              width:150px;
            }
            .left{
                width:70%;
                height: auto;
                float: right;
                overflow: scroll;
            }
            .right{
                width:50%;
                height: auto;
                overflow: scroll;
            }
            .train-parameters {
                border: 2px solid #000;
                padding: 15px;
                width: 250px;
                margin-bottom: 20px;
                border-radius: 10px;
                background-color: #f8f8f8;
                display: flex;
                flex-direction: column;
                gap: 10px;
            }
            
            .train-parameters h3 {
                margin-top: 0;
                font-size: 18px;
                text-align: center;
            }
            
            .train-parameters label {
                font-weight: bold;
            }
            
            .train-parameters input {
                width: 100%;
                height: 30px;
                font-size: 16px;
            }
            .danger-button {
                width: 150px;
                height: 40px;
                background-color: #ff4444;
                color: white;
                margin-top: 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }
            .danger-button:hover {
                background-color: #ff0000;
            }
            .hyperparams-container {
                border: 2px solid #000;
                padding: 15px;
                width: 250px;
                margin-bottom: 20px;
                border-radius: 10px;
                background-color: #f8f8f8;
            }
            
            .param-group {
                margin-bottom: 15px;
            }
            
            .param-group label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
            }
            
            .param-group input {
                width: 100%;
                height: 30px;
                font-size: 16px;
            }
        </style>

        <script>
            var base_url = "http://""" + str(cur_host) + """:8010/"

            async function call_bff(method, path, body_item){
                return await fetch(base_url+path, {
                    method: method,
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
            
                if (!selectedModelId) {
                    alert("Выберите модель!");
                    return;
                }
            
                const response = await call_bff("POST", "load_model", { "stored_item_id": selectedModelId });
            
                if (response.mymodel_desc) {
                    document.getElementById("selected_model_desc").innerText = response.mymodel_desc;
                    document.getElementById("model_controls").style.display = "block"; // Показываем кнопки
                }
            };
            
            const handle_train_button = async (nn_id) => {
                const a = await call_bff_get('train')
            }
            const handle_run_button = async (nn_id) => {
                const a = await call_bff('POST', 'run', {})
            }

            const handle_create_button = async () => {
                const desc = document.getElementById('neural_desc').value;
                
                // Получаем значения гиперпараметров
                const params = {
                    'mymodel_type': "oscil",
                    'mymodel_desc': desc,
                    
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

                const a = await call_bff('POST', 'create_model', params);
                console.log('pass: ', a);
            };

            const handle_update_params_button = async () => {
                const params = {
                    'epochs': document.getElementById('epochs').value || 100,
                    'optimizer': document.getElementById('optimizer').value || "Adam",
                    'optimizer_lr': parseFloat(document.getElementById('optimizer_lr').value) || 0.001,
                    'mymodel_type': "oscil"
                };
                
                const response = await call_bff('POST', 'update_train_params', params);
                console.log('update params response:', response);
            };

            const handle_clear_database = async () => {
                if (confirm('Удалить все записи в базе данных?')) {
                    window.location.href = base_url + 'clear_database';
                }
            };
        </script>    
        """ + str(gscripts) + """
    </head>
    <body>
        <div class="row">
            <div style="display: flex; flex-direction: column; gap: 10px; margin-bottom: 20px;">
                <div>
                    <label for="neural_desc">Описание модели:</label>
                    <input style="width: 200px; height: 30px; font-size: 16px;" 
                           type="text" id="neural_desc" placeholder="Описание модели"/>
                </div>
                
                <div class="hyperparams-container">
                    <h3>Гиперпараметры архитектуры</h3>
                    <div class="param-group">
                        <label for="input_dim">Input Dimension:</label>
                        <input type="number" id="input_dim" placeholder="1"/>
                        
                        <label for="output_dim">Output Dimension:</label>
                        <input type="number" id="output_dim" placeholder="1"/>
                        
                        <label for="hidden_sizes">Hidden Sizes (через запятую):</label>
                        <input type="text" id="hidden_sizes" placeholder="32,32,32"/>
                    </div>
                    
                    <h3>Параметры Фурье</h3>
                    <div class="param-group">
                        <label>
                            <input type="checkbox" id="fourier"/> Использовать Фурье
                        </label>
                        
                        <label for="finput_dim">Fourier Input Dimension:</label>
                        <input type="number" id="finput_dim" placeholder="Optional"/>
                        
                        <label for="fourier_scale">Fourier Scale:</label>
                        <input type="number" id="fourier_scale" placeholder="Optional"/>
                    </div>
                </div>
                
                <div class="train-parameters">
                    <h3>Train Parameters</h3>
                    <label for="epochs">Количество эпох:</label>
                    <input type="text" id="epochs" placeholder="Количество эпох"/>
                    
                    <label for="optimizer">Оптимизатор:</label>
                    <input type="text" id="optimizer" placeholder="Оптимизатор"/>
                    
                    <label for="optimizer_lr">Оптимизатор learning rate:</label>
                    <input type="text" id="optimizer_lr" placeholder="Оптимизатор learning rate"/>
                    
                    <input style="width: 150px; height: 30px; margin-top: 10px;" 
                           type="button" value="Update Parameters" 
                           onclick="handle_update_params_button()"/>
                </div>
                <input style="width: 80px; height: 40px" type="button" value="Create" id="create_btn"/>
                <input type="button" class="danger-button" 
                       value="Clear Database" 
                       onclick="handle_clear_database()"
                       id="clear_db_btn"/>
                <div id="model_desc">Выбранная модель: <span id="selected_model_desc">---</span></div>
            </div>
        </div>
        <div class="row">
            <div class="left">""" + str(gdivs_left) + """</div>
            <div class="right">""" + str(gdivs_right) + """</div>
        </div>
    </body>
    <script>

        var ws_ping = new WebSocket(`ws://""" + str(cur_host) + """:8010/ws_ping`);
        ws_ping.onmessage = function(event) 
        {
            const res = JSON.parse(event.data);
            console.log("Пришло сообщение:", res); // Проверка
        
            if (res.msg_type == 'jinja_tmpl') {
                if (res.data[0] === 'model_desc') {
                    console.log("Обновляем описание модели:", res.data[1]); // Проверка
                    document.getElementById('selected_model_desc').innerText = res.data[1];
                } else {
                    console.log("Обновляем модели:", res.data[1]); // Проверка
                    document.getElementById(res.data[0]).innerHTML = res.data[1];
                }
            }
        }


        create_btn.addEventListener('click', handle_create_button);
        document.getElementById('update_params_btn').addEventListener('click', handle_update_params_button);
        document.getElementById('clear_db_btn').addEventListener('click', handle_clear_database);
    </script>
</html>
"""