const socket = new WebSocket('ws://localhost:8080');

        socket.onopen = function (event) {
            alert('You are Connected to WebSocket Server');
        };