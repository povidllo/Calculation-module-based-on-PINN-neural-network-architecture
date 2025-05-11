optimizer_dict = {
    'Adam': [
        {'id': 'lr', 'type': 'number', 'default_value': 0.001, 'label': 'Learning Rate'},
        {'id': 'betas', 'type': 'text', 'default_value': '0.9,0.999', 'label': 'Betas (через запятую)'},
        {'id': 'eps', 'type': 'number', 'default_value': 1e-8, 'label': 'Epsilon'},
        {'id': 'weight_decay', 'type': 'number', 'default_value': 0, 'label': 'Weight Decay'}
    ],
    'SGD': [
        {'id': 'lr', 'type': 'number', 'default_value': 0.01, 'label': 'Learning Rate'},
        {'id': 'momentum', 'type': 'number', 'default_value': 0.9, 'label': 'Momentum'},
        {'id': 'weight_decay', 'type': 'number', 'default_value': 0, 'label': 'Weight Decay'},
        {'id': 'nesterov', 'type': 'checkbox', 'default_value': False, 'label': 'Nesterov'}
    ],
    'RMSprop': [
        {'id': 'lr', 'type': 'number', 'default_value': 0.01, 'label': 'Learning Rate'},
        {'id': 'alpha', 'type': 'number', 'default_value': 0.99, 'label': 'Alpha'},
        {'id': 'eps', 'type': 'number', 'default_value': 1e-8, 'label': 'Epsilon'},
        {'id': 'momentum', 'type': 'number', 'default_value': 0, 'label': 'Momentum'}
    ],
    'Adagrad': [
        {'id': 'lr', 'type': 'number', 'default_value': 0.01, 'label': 'Learning Rate'},
        {'id': 'lr_decay', 'type': 'number', 'default_value': 0, 'label': 'Learning Rate Decay'},
        {'id': 'weight_decay', 'type': 'number', 'default_value': 0, 'label': 'Weight Decay'}
    ]
}
