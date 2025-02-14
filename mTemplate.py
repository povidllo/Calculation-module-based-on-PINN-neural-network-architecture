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
            }
            .right{
                width:50%;
                height: auto;
                overflow: scroll;
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
                
        </script>    
        """ + str(gscripts) + """
    </head>
    <body>
        <div class="row">
        <input style= "width: 200px; height: 60px; font-size: 24px;" type="text" id="neural_desc"  />
        <input style= "width: 80px; height: 60px" type=button value="Create"  id="create_btn") />
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
        
        
        const callback_ok_create_btn = async () => {
        
             const a = await call_bff('POST', 'create_model', {'mymodel_type': "oscil", 'mymodel_desc': neural_desc.value} );
             console.log('pass: ' , a );
             
       };
       create_btn.addEventListener('click', callback_ok_create_btn);   
    </script>
</html>
"""