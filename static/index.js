let mousePressed = false;
let lastX, lastY;
let ctx;
let canvas;

// Function called when the window finishes loading
window.onload = function() {
    
    canvas = document.getElementById('myCanvas');
    ctx = canvas.getContext("2d");
    ctx.fillRect(0, 0, canvas.width, canvas.height);


    canvas.addEventListener('mousedown', function (e) {
        mousePressed = true;
        Draw(e.pageX - canvas.offsetLeft, e.pageY - canvas.offsetTop, false);
    });

    canvas.addEventListener('mousemove', function (e) {
        if (mousePressed) {
            Draw(e.pageX - canvas.offsetLeft, e.pageY - canvas.offsetTop, true);
        }
    });

    canvas.addEventListener('mouseup', function (e) {
        mousePressed = false;
    });

    canvas.addEventListener('mouseleave', function (e) {
        mousePressed = false;
    });
}
// Function to draw on the canvas

function Draw(x, y, isDown) {
    if (isDown) {
        ctx.beginPath();
        // ctx.strokeStyle = document.getElementById('selColor').value;
        ctx.strokeStyle = "white";
        ctx.lineWidth = 7;
        ctx.lineJoin = "round";
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(x, y);
        ctx.closePath();
        ctx.stroke();
    }
    lastX = x; lastY = y;
}

// Function to clear the canvas area
function clearArea() {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

// Function to send the image to the server for prediction
function sendImage() {
    let canvas = document.getElementById('myCanvas');
    let image = canvas.toDataURL('http://127.0.0.1:400/image/png');
    let label = document.getElementById("label").value;


    let xhr = new XMLHttpRequest();
    xhr.open('POST', '/predict', true);
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.onload = function(){
        if (xhr.status === 200) {
            let jsonresponse = JSON.parse(xhr.responseText);
            let predicted_class = jsonresponse.predicted_class
            let confidence = jsonresponse.confidence
            document.getElementById("class").innerHTML = predicted_class
            document.getElementById("confidence").innerHTML = confidence

        }
    }
    xhr.send(JSON.stringify({ image: image, label: label}));

}


// Function to train the model with the drawn image
function trainImage() {
    // modif
    let canvas = document.getElementById('myCanvas');
    let image = canvas.toDataURL('image/png');
    
    let label = document.getElementById("label").value;


    let xhr = new XMLHttpRequest();

    xhr.open('POST', 'http://127.0.0.1:4000/train', true);
    xhr.setRequestHeader('Content-Type', 'application/json');

    xhr.onload = function(){
        if (xhr.status === 200) {
            let jsonresponse = JSON.parse(xhr.responseText);
            let train_complete = jsonresponse.train_complete;
            if (train_complete){

                let train_complete_div = document.getElementById("train_complete")
                train_complete_div.innerHTML = "model has been trained !";

                setTimeout(()=>{
                    train_complete_div.innerHTML = "";
                }, 5000)
            }
        }
    }
    xhr.send(JSON.stringify({ image: image, label: label}));


}