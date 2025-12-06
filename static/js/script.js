let predictionBox = document.getElementById("prediction");
let startBtn = document.getElementById("startBtn");
let videoDisplay = document.getElementById("videoDisplay");

let sentenceBox = document.getElementById("sentenceBox");
let clearBtn = document.getElementById("clearBtn");

let ws;
let videoElement;         // REAL camera video element
let fullSentence = "";
let lastAcceptedTime = 0;
let inputDelay = 3000;
let lastSentTime = 0;

// ---------------------------
// START CAMERA + WEBSOCKET
// ---------------------------

startBtn.onclick = () => {

    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {

            // Create hidden video element ONCE
            videoElement = document.createElement("video");
            videoElement.style.display = "none";
            document.body.appendChild(videoElement);

            videoElement.srcObject = stream;
            videoElement.play();

            // Start WebSocket
            ws = new WebSocket("ws://localhost:8000/ws");

            ws.onmessage = (event) => {
                let data = JSON.parse(event.data);

                let letter = data.letter;
                predictionBox.textContent = letter;

                // Show processed frame
                videoDisplay.src = "data:image/jpeg;base64," + data.frame;

                let now = Date.now();

                // Accept new input every 3 seconds
                if (letter !== "-" && now - lastAcceptedTime >= inputDelay) {
                    // Handle special gestures
                    if (letter === "del") {
                        // Delete last character
                        fullSentence = fullSentence.slice(0, -1);
                    } else if (letter === "space") {
                        // Add space
                        fullSentence += " ";
                    } else {
                        // Add regular letter
                        fullSentence += letter;
                    }
                    sentenceBox.value = fullSentence;
                    lastAcceptedTime = now;
                }
            };

            // Capture + attempt to send a frame regularly, but only actually
            // send over the websocket at most once every `inputDelay` ms.
            // This keeps the local video smooth while preventing the server
            // from being flooded and avoids UI lag.
            setInterval(captureFrame, 150);

        })
        .catch(err => {
            alert("Camera access failed: " + err);
        });
};


// ---------------------------
// CAPTURE FRAME PROPERLY
// ---------------------------

function captureFrame() {
    if (!videoElement) return;

    let canvas = document.createElement("canvas");
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;

    let ctx = canvas.getContext("2d");
    ctx.drawImage(videoElement, 0, 0);

    let dataURL = canvas.toDataURL("image/jpeg");

    // Throttle actual sends to once per `inputDelay` milliseconds.
    let now = Date.now();
    if (ws && ws.readyState === WebSocket.OPEN && now - lastSentTime >= inputDelay) {
        ws.send(dataURL);
        lastSentTime = now;
    }
}


// ---------------------------
// BUTTONS
// ---------------------------

clearBtn.onclick = () => {
    fullSentence = "";
    sentenceBox.value = "";
};


