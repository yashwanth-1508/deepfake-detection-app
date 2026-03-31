document.addEventListener('DOMContentLoaded', () => {
    const sections = ['home', 'login', 'detection'];
    const navLinks = document.querySelectorAll('nav a, button[onclick^="showSection"]');
    
    // Global function to show sections
    window.showSection = (sectionId) => {
        // Hide all sections
        document.querySelectorAll('.section').forEach(s => s.style.display = 'none');
        // Show target section
        const target = document.getElementById(`${sectionId}-section`);
        if (target) {
            target.style.display = sectionId === 'home' ? 'flex' : 'flex'; // Both use flex center
            if (sectionId === 'home') {
                document.querySelector('.site-overlay').style.background = 'linear-gradient(90deg, #05060f 0%, rgba(5, 6, 15, 0.8) 50%, rgba(5, 6, 15, 0.2) 100%)';
            } else {
                document.querySelector('.site-overlay').style.background = 'rgba(5, 6, 15, 0.85)';
            }
        }
        
        // Update nav active state
        document.querySelectorAll('.nav-links a').forEach(a => {
            a.classList.remove('active');
            if (a.getAttribute('data-section') === sectionId) a.classList.add('active');
        });

        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    };

    // Handle Nav Clicks
    document.querySelectorAll('nav a').forEach(link => {
        link.addEventListener('click', (e) => {
            const section = link.getAttribute('data-section');
            if (section) {
                e.preventDefault();
                showSection(section);
            }
        });
    });

    // Login Form Mock
    const loginForm = document.getElementById('login-form');
    if (loginForm) {
        loginForm.addEventListener('submit', (e) => {
            e.preventDefault();
            // Mock Login Success
            const btn = loginForm.querySelector('button');
            const originalText = btn.textContent;
            btn.textContent = "VERIFYING...";
            btn.disabled = true;
            
            setTimeout(() => {
                alert("Identity Verified. Access Granted.");
                showSection('detection');
                btn.textContent = originalText;
                btn.disabled = false;
            }, 1000);
        });
    }

    // --- Scanner Logic ---
    const fileInput = document.getElementById('file-input');
    const detectBtn = document.getElementById('detect-btn');
    const fileNameDisplay = document.getElementById('file-name');
    const resultDiv = document.getElementById('result');
    const loadingDiv = document.getElementById('loading');
    const predictionText = document.getElementById('prediction-text');
    const confidenceBar = document.getElementById('confidence-bar');
    const confidenceText = document.getElementById('confidence-text');
    
    const API_URL = (window.location.hostname === 'localhost' || window.location.hostname.startsWith('127.') || window.location.hostname === '[::]' || window.location.hostname === '::1')
        ? 'http://localhost:8000/detect' 
        : 'https://deepfake-detection-app-production.up.railway.app/detect'; 


    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            fileNameDisplay.textContent = `Target: ${e.target.files[0].name}`;
            detectBtn.disabled = false;
            resultDiv.style.display = 'none';
        } else {
            fileNameDisplay.textContent = '';
            detectBtn.disabled = true;
        }
    });
    
    detectBtn.addEventListener('click', async () => {
        if (fileInput.files.length === 0) return;
        
        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append('file', file);
        
        // UI updates
        detectBtn.disabled = true;
        loadingDiv.style.display = 'block';
        resultDiv.style.display = 'none';
        
        // Clear old heatmap and list
        document.getElementById('heatmap-container').style.display = 'none';
        document.getElementById('heatmap-img').src = '';
        document.getElementById('faces-list').style.display = 'none';
        
        try {
            const startTime = Date.now();
            const response = await fetch(API_URL, {
                method: 'POST',
                body: formData,
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Neural analysis failed');
            }
            
            const data = await response.json();
            
            // Scanner feel delay
            const elapsed = Date.now() - startTime;
            if (elapsed < 1200) await new Promise(r => setTimeout(r, 1200 - elapsed));

            loadingDiv.style.display = 'none';
            resultDiv.style.display = 'block';
            
            const confidencePercent = (data.confidence * 100).toFixed(1);
            predictionText.textContent = data.prediction;
            confidenceText.textContent = `Confidence: ${confidencePercent}%`;
            
            // Apply Dynamic Styling based on the prediction tier
            let statusClass = 'undetermined';
            let statusColor = 'var(--neutral)';
            
            if (data.prediction === 'Real') {
                statusClass = 'real';
                statusColor = 'var(--real)';
            } else if (data.prediction === 'Likely Real') {
                statusClass = 'likely-real';
                statusColor = 'var(--warn)';
            } else if (data.prediction === 'Deepfake') {
                statusClass = 'deepfake';
                statusColor = 'var(--fake)';
            } else if (data.prediction.includes('Undetermined')) {
                statusClass = 'undetermined';
                statusColor = 'var(--neutral)';
            }
            
            resultDiv.className = `result-card ${statusClass}`;
            predictionText.style.color = statusColor;
            confidenceBar.style.backgroundColor = statusColor;
            
            setTimeout(() => {
                confidenceBar.style.width = `${confidencePercent}%`;
                
                // Animate Neon Pie Chart
                const pieChart = document.getElementById('neural-pie');
                const pieLabel = document.querySelector('.pie-label');
                if (pieChart) {
                    const deg = (data.confidence * 360).toFixed(0);
                    pieChart.style.background = `conic-gradient(${statusColor} ${deg}deg, rgba(255, 255, 255, 0.05) 0%)`;
                    if(pieLabel) pieLabel.textContent = data.prediction === 'Real' ? 'REAL' : 'FAKE';
                }
            }, 100);

            // Handle Multi-Face Details (Frames Grid)
            const facesList = document.getElementById('faces-list');
            const facesContainer = document.getElementById('faces-container');
            const heatmapContainer = document.getElementById('heatmap-container');
            const heatmapImg = document.getElementById('heatmap-img');
            
            facesContainer.innerHTML = ''; // Clear old

            if (data.faces && data.faces.length > 0) {
                facesList.style.display = 'block';
                // Take up to 6 faces to fit the grid nicely
                const displayFaces = data.faces.slice(0, 8);
                displayFaces.forEach((face) => {
                    const faceItem = document.createElement('div');
                    faceItem.className = 'frame-item';
                    
                    const fColor = face.prediction === 'Real' ? 'var(--real)' : 'var(--fake)';
                    
                    // We use the heatmap thumbnail as the visual
                    faceItem.innerHTML = `
                        <img src="data:image/jpeg;base64,${face.heatmap}" class="frame-thumbnail" onclick="showHeatmap('${face.heatmap}')" style="cursor: pointer;" alt="Analyzed Target">
                        <div class="frame-score" style="color: ${fColor}">
                            ${(face.confidence * 100).toFixed(0)}% ${face.prediction.substring(0, 1)}
                        </div>
                    `;
                    facesContainer.appendChild(faceItem);
                });
            } else {
                facesList.style.display = 'none';
            }

            // Global function for buttons
            window.showHeatmap = (b64) => {
                if (b64 && b64 !== 'null') {
                    heatmapImg.src = `data:image/jpeg;base64,${b64}`;
                    heatmapContainer.style.display = 'block';
                    heatmapContainer.scrollIntoView({ behavior: 'smooth' });
                }
            };
            
            // Legacy support for single heatmap if provided at top level
            if (data.heatmap) {
                heatmapImg.src = `data:image/jpeg;base64,${data.heatmap}`;
                heatmapContainer.style.display = 'block';
            }
            
        } catch (error) {
            console.error('Error:', error);
            alert("Analysis Error: " + error.message);
        } finally {
            detectBtn.disabled = false;
            loadingDiv.style.display = 'none';
        }
    });
});
