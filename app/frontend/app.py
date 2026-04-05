from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import httpx
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

app = FastAPI(title="Vegetable Classifier Frontend")

API_URL = os.getenv("API_URL", "http://localhost:8000")

# HTML страница как строка
HTML_PAGE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vegetable Classifier</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }

        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 10px;
        }

        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }

        .task-selector {
            display: flex;
            gap: 15px;
            margin-bottom: 30px;
            justify-content: center;
        }

        .task-btn {
            padding: 12px 30px;
            background: #2c3e50;
            border: 2px solid #ddd;
            border-radius: 10px;
            cursor: pointer;
            font-size: 16px;
            transition: all 0.3s ease;
        }

        .task-btn.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-color: #667eea;
        }

        .task-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }

        .task-panel {
            display: none;
        }

        .task-panel.active {
            display: block;
        }

        .upload-area {
            border: 2px dashed #667eea;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 20px;
        }

        .upload-area:hover {
            background: #f7f7ff;
            border-color: #764ba2;
        }

        .upload-area.dragover {
            background: #f0f0ff;
            border-color: #764ba2;
        }

        .file-input {
            display: none;
        }

        .preview-container {
            text-align: center;
            margin: 20px 0;
        }

        .preview-container-dual {
            display: flex;
            gap: 20px;
            margin: 20px 0;
            justify-content: center;
            flex-wrap: wrap;
        }

        .preview-item {
            flex: 1;
            min-width: 200px;
            text-align: center;
        }

        .preview-item img {
            max-width: 100%;
            max-height: 250px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }

        .preview-label {
            margin-top: 10px;
            font-weight: bold;
            color: #333;
        }

        button {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            cursor: pointer;
            transition: transform 0.2s;
        }

        button:hover {
            transform: translateY(-2px);
        }

        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .result {
            margin-top: 30px;
            padding: 20px;
            background: #f7f7ff;
            border-radius: 10px;
            text-align: center;
        }

        .result h2 {
            color: #333;
            margin-bottom: 10px;
        }

        .confidence-bar {
            width: 100%;
            height: 30px;
            background: #e0e0e0;
            border-radius: 15px;
            overflow: hidden;
            margin-top: 10px;
        }

        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            color: white;
            line-height: 30px;
            padding-right: 10px;
            text-align: right;
            transition: width 0.5s ease;
        }

        .similarity-score {
            font-size: 48px;
            font-weight: bold;
            color: #667eea;
            margin: 20px 0;
        }

        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .status {
            text-align: center;
            margin-bottom: 20px;
            padding: 10px;
            border-radius: 10px;
        }

        .status.online {
            background: #d4edda;
            color: #155724;
        }

        .status.offline {
            background: #f8d7da;
            color: #721c24;
        }

        .clear-btn {
            margin-top: 10px;
            background: #6c757d;
            width: auto;
            padding: 8px 20px;
            font-size: 14px;
        }

        .clear-btn:hover {
            background: #5a6268;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Vegetable Classifier</h1>
        <div class="subtitle">Upload vegetable photos for recognition or comparison</div>

        <div id="status" class="status">Checking connection...</div>

        <div class="task-selector">
            <button class="task-btn active" data-task="classify">Classification</button>
            <button class="task-btn" data-task="similarity">Similarity</button>
        </div>

        <!-- Classification Panel -->
        <div id="classifyPanel" class="task-panel active">
            <div class="upload-area" id="uploadAreaClassify">
                <p>Click or drag and drop an image here</p>
                <input type="file" id="fileInputClassify" class="file-input" accept="image/*">
            </div>

            <div class="preview-container" id="previewContainerClassify" style="display: none;">
                <img id="previewClassify" alt="Preview">
            </div>

            <button id="predictBtn" disabled>Recognize Vegetable</button>
            <button class="clear-btn" id="clearClassifyBtn" style="display: none;">Clear</button>

            <div id="resultClassify" class="result" style="display: none;"></div>
        </div>

        <!-- Similarity Panel -->
        <div id="similarityPanel" class="task-panel">
            <div class="upload-area" id="uploadAreaSimilarity1">
                <p>First image - click or drag and drop</p>
                <input type="file" id="fileInputSimilarity1" class="file-input" accept="image/*">
            </div>

            <div class="preview-container-dual" id="previewContainerSimilarity" style="display: none;">
                <div class="preview-item">
                    <img id="previewSimilarity1" alt="Preview 1">
                    <div class="preview-label">Image 1</div>
                </div>
                <div class="preview-item">
                    <img id="previewSimilarity2" alt="Preview 2">
                    <div class="preview-label">Image 2</div>
                </div>
            </div>

            <div class="upload-area" id="uploadAreaSimilarity2">
                <p>Second image - click or drag and drop</p>
                <input type="file" id="fileInputSimilarity2" class="file-input" accept="image/*">
            </div>

            <button id="similarityBtn" disabled>Compare Images</button>
            <button class="clear-btn" id="clearSimilarityBtn" style="display: none;">Clear All</button>

            <div id="resultSimilarity" class="result" style="display: none;"></div>
        </div>
    </div>

    <script>
        // Task switching
        let currentTask = 'classify';

        const taskBtns = document.querySelectorAll('.task-btn');
        const classifyPanel = document.getElementById('classifyPanel');
        const similarityPanel = document.getElementById('similarityPanel');

        taskBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                taskBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');

                currentTask = btn.dataset.task;

                if (currentTask === 'classify') {
                    classifyPanel.classList.add('active');
                    similarityPanel.classList.remove('active');
                } else {
                    classifyPanel.classList.remove('active');
                    similarityPanel.classList.add('active');
                }
            });
        });

        // Classification logic
        let classifyFile = null;
        const uploadAreaClassify = document.getElementById('uploadAreaClassify');
        const fileInputClassify = document.getElementById('fileInputClassify');
        const previewClassify = document.getElementById('previewClassify');
        const previewContainerClassify = document.getElementById('previewContainerClassify');
        const predictBtn = document.getElementById('predictBtn');
        const resultClassify = document.getElementById('resultClassify');
        const clearClassifyBtn = document.getElementById('clearClassifyBtn');

        uploadAreaClassify.addEventListener('click', () => fileInputClassify.click());

        uploadAreaClassify.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadAreaClassify.classList.add('dragover');
        });

        uploadAreaClassify.addEventListener('dragleave', () => {
            uploadAreaClassify.classList.remove('dragover');
        });

        uploadAreaClassify.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadAreaClassify.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                handleClassifyFile(file);
            }
        });

        fileInputClassify.addEventListener('change', (e) => {
            if (e.target.files[0]) {
                handleClassifyFile(e.target.files[0]);
            }
        });

        function handleClassifyFile(file) {
            classifyFile = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                previewClassify.src = e.target.result;
                previewContainerClassify.style.display = 'block';
                predictBtn.disabled = false;
                resultClassify.style.display = 'none';
                clearClassifyBtn.style.display = 'inline-block';
            };
            reader.readAsDataURL(file);
        }

        clearClassifyBtn.addEventListener('click', () => {
            classifyFile = null;
            previewContainerClassify.style.display = 'none';
            previewClassify.src = '';
            predictBtn.disabled = true;
            resultClassify.style.display = 'none';
            clearClassifyBtn.style.display = 'none';
            fileInputClassify.value = '';
        });

        predictBtn.addEventListener('click', async () => {
            if (!classifyFile) return;

            predictBtn.disabled = true;
            predictBtn.textContent = 'Recognizing...';
            resultClassify.style.display = 'block';
            resultClassify.innerHTML = '<div class="spinner"></div><p>Analyzing image...</p>';

            const formData = new FormData();
            formData.append('file', classifyFile);

            try {
                const response = await fetch('/classify', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (response.ok) {
                    const confidencePercent = (data.confidence * 100).toFixed(2);
                    resultClassify.innerHTML = `
                        <h2>Result: ${data.class_name}</h2>
                        <p>Model confidence:</p>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${confidencePercent}%">
                                ${confidencePercent}%
                            </div>
                        </div>
                        <p style="margin-top: 10px; font-size: 12px; color: #666;">Model: ${data.model_used || 'ONNX'}</p>
                    `;
                } else {
                    resultClassify.innerHTML = `<p style="color: red;">Error: ${data.detail || 'Unknown error'}</p>`;
                }
            } catch (error) {
                resultClassify.innerHTML = `<p style="color: red;">Connection error to server</p>`;
            } finally {
                predictBtn.disabled = false;
                predictBtn.textContent = 'Recognize Vegetable';
            }
        });

        // Similarity logic
        let similarityFile1 = null;
        let similarityFile2 = null;
        const uploadAreaSimilarity1 = document.getElementById('uploadAreaSimilarity1');
        const uploadAreaSimilarity2 = document.getElementById('uploadAreaSimilarity2');
        const fileInputSimilarity1 = document.getElementById('fileInputSimilarity1');
        const fileInputSimilarity2 = document.getElementById('fileInputSimilarity2');
        const previewSimilarity1 = document.getElementById('previewSimilarity1');
        const previewSimilarity2 = document.getElementById('previewSimilarity2');
        const previewContainerSimilarity = document.getElementById('previewContainerSimilarity');
        const similarityBtn = document.getElementById('similarityBtn');
        const resultSimilarity = document.getElementById('resultSimilarity');
        const clearSimilarityBtn = document.getElementById('clearSimilarityBtn');

        function setupSimilarityUpload(uploadArea, fileInput, preview, fileVar, fileNumber) {
            uploadArea.addEventListener('click', () => fileInput.click());

            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });

            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });

            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                const file = e.dataTransfer.files[0];
                if (file && file.type.startsWith('image/')) {
                    handleSimilarityFile(file, preview, fileVar, fileNumber);
                }
            });

            fileInput.addEventListener('change', (e) => {
                if (e.target.files[0]) {
                    handleSimilarityFile(e.target.files[0], preview, fileVar, fileNumber);
                }
            });
        }

        function handleSimilarityFile(file, preview, fileVar, fileNumber) {
            if (fileNumber === 1) {
                similarityFile1 = file;
            } else {
                similarityFile2 = file;
            }

            const reader = new FileReader();
            reader.onload = (e) => {
                preview.src = e.target.result;
                previewContainerSimilarity.style.display = 'flex';

                if (similarityFile1 && similarityFile2) {
                    similarityBtn.disabled = false;
                }

                clearSimilarityBtn.style.display = 'inline-block';
                resultSimilarity.style.display = 'none';
            };
            reader.readAsDataURL(file);
        }

        setupSimilarityUpload(uploadAreaSimilarity1, fileInputSimilarity1, previewSimilarity1, 'similarityFile1', 1);
        setupSimilarityUpload(uploadAreaSimilarity2, fileInputSimilarity2, previewSimilarity2, 'similarityFile2', 2);

        clearSimilarityBtn.addEventListener('click', () => {
            similarityFile1 = null;
            similarityFile2 = null;
            previewContainerSimilarity.style.display = 'none';
            previewSimilarity1.src = '';
            previewSimilarity2.src = '';
            similarityBtn.disabled = true;
            resultSimilarity.style.display = 'none';
            clearSimilarityBtn.style.display = 'none';
            fileInputSimilarity1.value = '';
            fileInputSimilarity2.value = '';
        });

        similarityBtn.addEventListener('click', async () => {
            if (!similarityFile1 || !similarityFile2) return;

            similarityBtn.disabled = true;
            similarityBtn.textContent = 'Comparing...';
            resultSimilarity.style.display = 'block';
            resultSimilarity.innerHTML = '<div class="spinner"></div><p>Comparing images...</p>';

            const formData = new FormData();
            formData.append('file1', similarityFile1);
            formData.append('file2', similarityFile2);

            try {
                const response = await fetch('/similarity', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (response.ok) {
                    const similarityPercent = data.similarity_percentage.toFixed(2);
                    let similarityClass = '';
                    let message = '';

                    if (data.similarity_percentage > 80) {
                        similarityClass = 'high';
                        message = 'Very similar images';
                    } else if (data.similarity_percentage > 50) {
                        similarityClass = 'medium';
                        message = 'Moderately similar';
                    } else {
                        similarityClass = 'low';
                        message = 'Different images';
                    }

                    resultSimilarity.innerHTML = `
                        <h2>Similarity Result</h2>
                        <div class="similarity-score">${similarityPercent}%</div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${similarityPercent}%; text-align: center;">
                                ${similarityPercent}%
                            </div>
                        </div>
                        <p style="margin-top: 15px;">${message}</p>
                        <p style="margin-top: 10px; font-size: 12px; color: #666;">
                            Cosine similarity: ${data.similarity_score.toFixed(4)}
                        </p>
                    `;
                } else {
                    resultSimilarity.innerHTML = `<p style="color: red;">Error: ${data.detail || 'Unknown error'}</p>`;
                }
            } catch (error) {
                resultSimilarity.innerHTML = `<p style="color: red;">Connection error to server</p>`;
            } finally {
                similarityBtn.disabled = false;
                similarityBtn.textContent = 'Compare Images';
            }
        });

        // Health check
        async function checkStatus() {
            try {
                const response = await fetch('/health');
                const data = await response.json();
                if (response.ok && data.api_status !== 'unreachable') {
                    statusDiv.textContent = 'Server ready';
                    statusDiv.className = 'status online';
                } else {
                    statusDiv.textContent = 'Server not responding';
                    statusDiv.className = 'status offline';
                }
            } catch (error) {
                statusDiv.textContent = 'Connection error';
                statusDiv.className = 'status offline';
            }
        }

        const statusDiv = document.getElementById('status');
        checkStatus();
        setInterval(checkStatus, 30000);
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    """Return HTML page"""
    return HTMLResponse(content=HTML_PAGE)


@app.post("/classify")
async def classify_proxy(file: UploadFile = File(...)):
    """Proxy for classification API"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        content = await file.read()
        files = {"file": (file.filename, content, file.content_type)}
        response = await client.post(f"{API_URL}/classify", files=files)

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        return JSONResponse(content=response.json())


@app.post("/similarity")
async def similarity_proxy(
        file1: UploadFile = File(...),
        file2: UploadFile = File(...)
):
    """Proxy for similarity API"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        content1 = await file1.read()
        content2 = await file2.read()

        files = {
            "file1": (file1.filename, content1, file1.content_type),
            "file2": (file2.filename, content2, file2.content_type)
        }
        response = await client.post(f"{API_URL}/similarity", files=files)

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        return JSONResponse(content=response.json())


@app.get("/health")
async def health():
    """Check API health"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{API_URL}/health")
            return JSONResponse(content=response.json())
        except Exception as e:
            return JSONResponse(content={"api_status": "unreachable", "error": str(e)}, status_code=503)