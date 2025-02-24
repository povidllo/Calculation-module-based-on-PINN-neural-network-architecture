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

            const handle_load_button = async (nn_id) => {
                const a = await call_bff('POST', 'load_model', {'stored_item_id' : nn_id})
            }
            const handle_train_button = async (nn_id) => {
                const epochs = document.getElementById('epochs').value || 100;
                const optimizer = document.getElementById('optimizer').value || "Adam";
                const optimizer_lr = document.getElementById('optimizer_lr').value || 0.001;
                
                const train_params = {
                    'epochs': epochs
                    'optimizer': optimizer,
                    'optimizer_lr': optimizer_lr
                };
                
                const a = await call_bff_get('train', train_params)
            }
            const handle_run_button = async (nn_id) => {
                const a = await call_bff('POST', 'run', {})
            }

            const handle_create_button = async () => {
                const desc = document.getElementById('neural_desc').value;
                const weights_path = document.getElementById('weights_path').value || '/osc_1d.pth';  // Значение по умолчанию

                const params = {
                    'mymodel_type': "oscil", 
                    'mymodel_desc': desc,
                    'save_weights_path': weights_path
                };

                const a = await call_bff('POST', 'create_model', params);
                console.log('pass: ' , a);
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
                <div class="train-parameters">
                    <h3>Train Parameters</h3>
                    <label for="epochs">Количество эпох:</label>
                    <input type="text" id="epochs" placeholder="Количество эпох"/>
                    
                    <label for="optimizer">Оптимизатор:</label>
                    <input type="text" id="optimizer" placeholder="Оптимизатор"/>
                    
                    <label for="optimizer_lr">Оптимизатор learning rate:</label>
                    <input type="text" id="optimizer_lr" placeholder="Оптимизатор learning rate"/>
                </div>
                <div>
                    <label for="weights_path">Путь к весам:</label>
                    <input style="width: 200px; height: 30px; font-size: 16px;" 
                           type="text" id="weights_path" placeholder="/osc_1d.pth"/>
                </div>
                <input style="width: 80px; height: 40px" type="button" value="Create" id="create_btn"/>
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
            const res = JSON.parse(event.data)
            if (res.msg_type == 'hello')
            {
                console.log('hello from ws', res.data)
            }else if (res.msg_type == 'jinja_tmpl')
            {
                        //console.log('div name: ', res.data[0])
                        //console.log('content: ', res.data[1])
                        document.getElementById(res.data[0]).innerHTML = res.data[1]
            }
        }


        create_btn.addEventListener('click', handle_create_button);   
    </script>
</html>
"""