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