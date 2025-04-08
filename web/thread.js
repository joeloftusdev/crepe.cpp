let crepeModule;
let isInitialized = false;
const ANALYSIS_INTERVAL = 250; 

onmessage = function(e) {
    const data = e.data;
    
    switch(data.command) {
        case 'INIT':
            initializeWasm(data.wasmUrl);
            break;
            
        case 'ANALYZE':
            if (!isInitialized) {
                postMessage({ 
                    type: 'ERROR', 
                    message: 'WASM module not initialized'
                });
                return;
            }
            analyzePitch(data.audioData, data.testFrequency);
            break;
            
        case 'CLEANUP':
            if (isInitialized && crepeModule._cleanup) {
                crepeModule._cleanup();
            }
            break;
    }
};

function initializeWasm(wasmUrl) {
    importScripts(wasmUrl);
    
    CrepeModule({
        onRuntimeInitialized: function() {
            console.log("[Worker] WASM runtime initialized");
        }
    }).then(module => {
        crepeModule = module;
        isInitialized = true;
        console.log("[Worker] CREPE module loaded successfully");
        
        postMessage({ 
            type: 'INIT_COMPLETE'
        });
    }).catch(err => {
        console.error("[Worker] Failed to load CREPE module:", err);
        postMessage({ 
            type: 'ERROR', 
            message: 'Failed to load CREPE module: ' + err.message 
        });
    });
}

function analyzePitch(audioData, testFrequency) {
    try {
        const startTime = performance.now();
        
        audioData = new Float32Array(audioData);
        
        const bufferBytes = audioData.length * 4;
        const inputPtr = crepeModule._malloc(bufferBytes);
        
        if (!inputPtr) {
            postMessage({ 
                type: 'ERROR', 
                message: 'Failed to allocate memory in WASM'
            });
            return;
        }
        
        crepeModule.HEAPF32.set(audioData, inputPtr / 4);
        
        const resultPtr = crepeModule._analyse_audio(inputPtr, audioData.length);
        
        crepeModule._free(inputPtr);
        
        if (!resultPtr) {
            postMessage({ 
                type: 'ERROR', 
                message: 'Null result from WASM module'
            });
            return;
        }
        
        const numFrames = crepeModule.HEAPF32[resultPtr/4];
        
        if (numFrames <= 0) {
            crepeModule._free(resultPtr);
            postMessage({ 
                type: 'ERROR', 
                message: 'No frames returned from analysis'
            });
            return;
        }
        
        const results = [];
        for (let i = 0; i < numFrames; i++) {
            const baseIndex = resultPtr/4 + 1 + i*3;
            results.push({
                pitch: crepeModule.HEAPF32[baseIndex],
                confidence: crepeModule.HEAPF32[baseIndex + 1],
                time: crepeModule.HEAPF32[baseIndex + 2]
            });
        }
        
        crepeModule._free(resultPtr);
        
        const processingTime = performance.now() - startTime;
        
        postMessage({ 
            type: 'ANALYSIS_COMPLETE',
            results: results,
            testFrequency: testFrequency,
            processingTime: processingTime
        });
        
    } catch (error) {
        postMessage({ 
            type: 'ERROR', 
            message: 'Error in pitch analysis: ' + error.message
        });
    }
}