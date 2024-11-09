function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}
function getCsrfToken() {
    console.log('ran')
    return document.querySelector('meta[name="csrf-token"]').getAttribute('content');
}


document.getElementById('summarize').addEventListener('click', function () {
    const article = document.getElementById('article').value;
    const resultDiv = document.getElementById('result');
    const loadingDiv = document.getElementById('loading');
    const overlayDiv = document.getElementById('overlay');

    loadingDiv.style.display = 'block';
    overlayDiv.style.display = 'block';
    resultDiv.innerText = '';  

    fetch('/summarize', {
        method: 'POST',
        headers: {
            'Content-type': 'text/plain',
            'X-CSRFToken': getCsrfToken(), 
        },
        body: article
    })
        .then(response => response.json())
        .then(data => {

            loadingDiv.style.display = 'none';
            overlayDiv.style.display = 'none';

            resultDiv.innerText = data.result;
        })
        .catch(error => {

            loadingDiv.style.display = 'none';
            overlayDiv.style.display = 'none';
            resultDiv.innerText = 'Error: ' + error;
            console.error('Error:', error);
        });
});
