// dom elements
const chatContainer = document.getElementById('chat-container');
const toggleButton = document.getElementById('toggle-listen');
const statusElement = document.getElementById('status');

// state variables
let isListening = false;
let webrtcConnection = null;
let websocket = null;

// add message to chat
function addMessage(text, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message');
    messageDiv.classList.add(isUser ? 'user-message' : 'agent-message');
    messageDiv.textContent = text;
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// update status
function updateStatus(text) {
    statusElement.textContent = text;
}

// toggle listening
toggleButton.addEventListener('click', () => {
    if (isListening) {
        stopListening();
    } else {
        startListening();
    }
});

// start listening
async function startListening() {
    try {
        updateStatus('connecting...');
        
        // initialize webrtc connection with stun servers
        webrtcConnection = new RTCPeerConnection({
            iceServers: [
                { urls: 'stun:stun.l.google.com:19302' },
                { urls: 'stun:stun1.l.google.com:19302' },
            ]
        });
        
        // set up audio stream
        const stream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                echoCancellation: false,
                noiseSuppression: true,
                autoGainControl: true,
                sampleRate: 24000,
                sampleSize: 16,
                channelCount: 1
            } 
        });
        
        console.log("Audio stream obtained:", stream);
        
        stream.getTracks().forEach(track => {
            console.log("Adding track to connection:", track);
            webrtcConnection.addTrack(track, stream);
        });
        
        // create data channel for debugging
        const dataChannel = webrtcConnection.createDataChannel("text");
        dataChannel.onopen = () => {
            console.log("Data channel opened");
            updateStatus("connected (data channel open)");
            dataChannel.send("Hello from client");
        };
        
        dataChannel.onmessage = (event) => {
            console.log("Received data channel message:", event.data);
            try {
                const data = JSON.parse(event.data);
                if (data.type === "debug") {
                    console.log("Debug info:", data);
                }
            } catch (e) {
                console.log("Raw data channel message:", event.data);
            }
        };
        
        // handle incoming audio
        webrtcConnection.ontrack = event => {
            console.log("received audio track", event);
            const audio = new Audio();
            audio.srcObject = event.streams[0];
            
            // debug audio stream
            const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            const source = audioCtx.createMediaStreamSource(event.streams[0]);
            const analyser = audioCtx.createAnalyser();
            source.connect(analyser);
            
            // log audio levels periodically
            const bufferLength = analyser.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);
            
            const checkAudioLevels = () => {
                if (isListening) {
                    analyser.getByteFrequencyData(dataArray);
                    let sum = 0;
                    for (let i = 0; i < bufferLength; i++) {
                        sum += dataArray[i];
                    }
                    const average = sum / bufferLength;
                    if (average > 0) {
                        console.log("Audio level:", average);
                    }
                    setTimeout(checkAudioLevels, 1000);
                }
            };
            checkAudioLevels();
            
            audio.play().catch(e => console.error("error playing audio:", e));
        };
        
        // handle ice candidates
        webrtcConnection.onicecandidate = event => {
            if (event.candidate) {
                // send ice candidates to server
                fetch('/webrtc/offer', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        candidate: event.candidate.toJSON(),
                        webrtc_id: Math.random().toString(36).substring(7),
                        type: "ice-candidate"
                    })
                }).catch(error => {
                    console.error('error sending ice candidate:', error);
                });
            }
        };
        
        // create offer
        const offer = await webrtcConnection.createOffer();
        await webrtcConnection.setLocalDescription(offer);
        
        console.log("created offer", offer);
        
        // wait for ice gathering to complete
        await new Promise(resolve => {
            if (webrtcConnection.iceGatheringState === 'complete') {
                resolve();
            } else {
                const checkState = () => {
                    if (webrtcConnection.iceGatheringState === 'complete') {
                        webrtcConnection.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                };
                webrtcConnection.addEventListener('icegatheringstatechange', checkState);
                
                // add a timeout just in case
                setTimeout(resolve, 1000);
            }
        });
        
        // send offer to server
        const response = await fetch('/webrtc/offer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sdp: webrtcConnection.localDescription.sdp,
                type: webrtcConnection.localDescription.type,
                webrtc_id: Math.random().toString(36).substring(7)
            })
        });
        
        if (!response.ok) {
            throw new Error(`server responded with ${response.status}: ${response.statusText}`);
        }
        
        const answerData = await response.json();
        console.log("received answer", answerData);
        
        if (!answerData.sdp || !answerData.type) {
            console.error('invalid answer data:', answerData);
            throw new Error('received invalid answer from server');
        }
        
        // set remote description
        await webrtcConnection.setRemoteDescription(new RTCSessionDescription({
            sdp: answerData.sdp,
            type: answerData.type
        }));
        
        // set up websocket for transcript and response messages
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        websocket = new WebSocket(`${wsProtocol}//${window.location.host}/ws`);
        
        websocket.onopen = () => {
            console.log('websocket connection established');
            updateStatus('listening...');
        };
        
        websocket.onmessage = event => {
            const message = JSON.parse(event.data);
            if (message.type === "transcript") {
                addMessage(message.text, true);
                updateStatus("processing response...");
            } else if (message.type === "response") {
                addMessage(message.text, false);
                updateStatus("listening...");
            } else if (message.type === "debug") {
                console.log("Debug info:", message);
                updateStatus(`debug: ${JSON.stringify(message.data).substring(0, 30)}...`);
            }
        };
        
        websocket.onclose = () => {
            console.log('websocket connection closed');
            if (isListening) {
                stopListening();
            }
        };
        
        // update ui
        isListening = true;
        toggleButton.textContent = 'stop listening';
        toggleButton.classList.add('listening');
        
    } catch (error) {
        console.error('error starting listening:', error);
        updateStatus(`error: ${error.message}`);
    }
}

// stop listening
function stopListening() {
    // close webrtc connection
    if (webrtcConnection) {
        webrtcConnection.close();
        webrtcConnection = null;
    }
    
    // close websocket connection
    if (websocket) {
        websocket.close();
        websocket = null;
    }
    
    // update ui
    isListening = false;
    toggleButton.textContent = 'start listening';
    toggleButton.classList.remove('listening');
    updateStatus('ready');
} 