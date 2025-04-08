let audioContext;
let isProcessing = false;
let pitchHistory = [];
const MAX_HISTORY = 100;
const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const startButton = document.getElementById('startButton');
const pitchDisplay = document.getElementById('pitchDisplay');
const noteName = document.getElementById('noteName');
const confidenceBar = document.getElementById('confidenceBar');
const pitchCanvas = document.getElementById('pitchCanvas');
const ctx = pitchCanvas.getContext('2d');
let lastAnalysisTime = 0;
const ANALYSIS_INTERVAL = 250; 
let currentTestFrequency = 440;


let crepeWorker;
let workerReady = false;


function setupCanvas() {
    pitchCanvas.width = pitchCanvas.clientWidth;
    pitchCanvas.height = pitchCanvas.clientHeight;
    ctx.clearRect(0, 0, pitchCanvas.width, pitchCanvas.height);
}


function setupTestControls() {
    const controlsContainer = document.createElement('div');
    controlsContainer.classList.add('controls-container');
    
    const freqSlider = document.createElement('input');
    freqSlider.type = 'range';
    freqSlider.min = '80';
    freqSlider.max = '1000';
    freqSlider.step = '1';
    freqSlider.value = '440';
    freqSlider.id = 'frequencySlider';
    freqSlider.classList.add('frequency-slider');
    
    const freqDisplay = document.createElement('div');
    freqDisplay.id = 'freqDisplay';
    freqDisplay.textContent = '440 Hz';
    freqDisplay.classList.add('frequency-display');
    
    freqSlider.addEventListener('input', () => {
        currentTestFrequency = parseInt(freqSlider.value, 10);
        freqDisplay.textContent = `${currentTestFrequency} Hz`;
        
        if (isProcessing) {
            lastAnalysisTime = 0;
            processAudio();
        }
    });
    
    const label = document.createElement('div');
    label.textContent = 'Test Tone Frequency:';
    label.style.fontWeight = 'bold';
    
    const perfIndicator = document.createElement('div');
    perfIndicator.id = 'perfIndicator';
    perfIndicator.style.fontSize = '12px';
    perfIndicator.style.marginTop = '10px';
    perfIndicator.style.color = '#666';
    
    controlsContainer.appendChild(label);
    controlsContainer.appendChild(freqDisplay);
    controlsContainer.appendChild(freqSlider);
    controlsContainer.appendChild(perfIndicator);
    
    pitchCanvas.parentNode.insertBefore(controlsContainer, pitchCanvas);
}

function frequencyToNoteName(frequency) {
    if (frequency < 20) return "--";
    const noteNum = Math.round(12 * Math.log2(frequency / 440)) + 69;
    return NOTE_NAMES[noteNum % 12] + (Math.floor(noteNum / 12) - 1);
}

function initWorker() {
    if (window.Worker) {
        crepeWorker = new Worker('thread.js');
        
        crepeWorker.onmessage = function(e) {
            const data = e.data;
            
            switch (data.type) {
                case 'INIT_COMPLETE':
                    console.log("Worker initialized successfully");
                    workerReady = true;
                    startButton.disabled = false;
                    startButton.textContent = "Start Test Tone";
                    break;
                    
                case 'ANALYSIS_COMPLETE':
                    handleAnalysisResults(data);
                    break;
                    
                case 'ERROR':
                    console.error("Worker error:", data.message);
                    break;
            }
        };
        
        crepeWorker.onerror = function(error) {
            console.error("Worker error:", error);
        };
        
        crepeWorker.postMessage({
            command: 'INIT',
            wasmUrl: 'crepe_wasm.js'
        });
    } else {
        console.error("Web Workers not supported in this browser");
        alert("Your browser doesn't support Web Workers. Please use a modern browser.");
    }
}

function handleAnalysisResults(data) {
    const { results, testFrequency, processingTime } = data;
    
    if (results && results.length > 0) {
        const { pitch, confidence } = results[results.length - 1];
        
        console.log(`Detected pitch: ${pitch.toFixed(1)} Hz (confidence: ${confidence.toFixed(2)}, in ${processingTime.toFixed(0)}ms)`);
        
        const perfIndicator = document.getElementById('perfIndicator');
        if (perfIndicator) {
            perfIndicator.textContent = `Processing time: ${processingTime.toFixed(0)}ms`;
        }
        
        confidenceBar.style.width = `${confidence * 100}%`;
        
        if (confidence > 0.5) {
            confidenceBar.style.backgroundColor = '#4CAF50'; // Green
        } else if (confidence > 0.2) {
            confidenceBar.style.backgroundColor = '#FFC107'; // Yellow
        } else {
            confidenceBar.style.backgroundColor = '#F44336'; // Red
        }
        
        pitchDisplay.style.opacity = Math.max(0.3, confidence);
        pitchDisplay.textContent = `${pitch.toFixed(1)} Hz`;
        
        noteName.textContent = frequencyToNoteName(pitch);
        
        pitchHistory.push({ pitch, confidence });
        if (pitchHistory.length > MAX_HISTORY) {
            pitchHistory.shift();
        }
        
        drawPitchHistory();
    }
    
    if (isProcessing) {
        lastAnalysisTime = Date.now();
        setTimeout(processAudio, ANALYSIS_INTERVAL);
    }
}

startButton.addEventListener('click', async () => {
    if (isProcessing) {
        isProcessing = false;
        startButton.textContent = "Start Test Tone";
        return;
    }

    try {
        if (!audioContext) {
            audioContext = new AudioContext();
        }
        
        isProcessing = true;
        startButton.textContent = "Stop Test Tone";
        processAudio();
    } catch (error) {
        console.error("Error:", error);
        alert("Error: " + error.message);
    }
});

function generateTestTone(frequency, duration, sampleRate) {
    const numSamples = Math.floor(duration * sampleRate);
    const buffer = new Float32Array(numSamples);
    for (let i = 0; i < numSamples; i++) {
        buffer[i] = Math.sin(2 * Math.PI * frequency * i / sampleRate) * 0.5;
    }
    return buffer;
}

function processAudio() {
    if (!isProcessing || !workerReady) {
        return;
    }

    const now = Date.now();
    const shouldAnalyze = now - lastAnalysisTime >= ANALYSIS_INTERVAL;
    
    if (shouldAnalyze) {
        const bufferLength = 16000; 
        const audioData = generateTestTone(currentTestFrequency, 1, 16000);
        
        console.log(`Sending test tone at ${currentTestFrequency} Hz to worker`);
        
        crepeWorker.postMessage({
            command: 'ANALYZE',
            audioData: audioData.buffer,
            testFrequency: currentTestFrequency
        }, [audioData.buffer]);
        
    } else {
        requestAnimationFrame(processAudio);
    }
}

function drawPitchHistory() {
    if (window.isDrawing) return;
    
    window.isDrawing = true;
    requestAnimationFrame(() => {
        ctx.clearRect(0, 0, pitchCanvas.width, pitchCanvas.height);
        
        const maxPitch = 800, minPitch = 50;
        const scaleY = pitch => pitchCanvas.height - ((pitch - minPitch) / (maxPitch - minPitch)) * pitchCanvas.height;

        ctx.strokeStyle = '#eee';
        ctx.lineWidth = 1;
        ctx.font = '10px sans-serif';
        ctx.fillStyle = '#999';
        
        for (let p = 100; p < maxPitch; p += 100) {
            const y = scaleY(p);
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(pitchCanvas.width, y);
            ctx.stroke();
            ctx.fillText(`${p} Hz`, 5, y - 2);
        }
        
        const testY = scaleY(currentTestFrequency);
        ctx.strokeStyle = '#999';
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(0, testY);
        ctx.lineTo(pitchCanvas.width, testY);
        ctx.stroke();
        ctx.setLineDash([]);
        
        if (pitchHistory.length > 1) {
            ctx.strokeStyle = '#1a73e8';
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            let firstPoint = true;
            
            pitchHistory.forEach((point, i) => {
                const x = (i / MAX_HISTORY) * pitchCanvas.width;
                const y = scaleY(point.pitch);
                
                if (firstPoint || i === 0) {
                    ctx.moveTo(x, y);
                    firstPoint = false;
                } else {
                    ctx.lineTo(x, y);
                }
            });
            
            ctx.stroke();
            
            pitchHistory.forEach((point, i) => {
                const x = (i / MAX_HISTORY) * pitchCanvas.width;
                const y = scaleY(point.pitch);
                
                ctx.fillStyle = `rgba(26, 115, 232, ${point.confidence})`;
                ctx.beginPath();
                ctx.arc(x, y, 2, 0, Math.PI * 2);
                ctx.fill();
            });
        }
        
        window.isDrawing = false;
    });
}

window.addEventListener('load', () => {
    setupCanvas();
    setupTestControls();
    
    startButton.disabled = true;
    startButton.textContent = "Loading CREPE...";
    initWorker();
});

window.addEventListener('resize', setupCanvas);

window.addEventListener('beforeunload', () => {
    if (crepeWorker) {
        crepeWorker.postMessage({ command: 'CLEANUP' });
        crepeWorker.terminate();
    }
    
    if (audioContext && audioContext.state !== 'closed') {
        audioContext.close();
    }
});