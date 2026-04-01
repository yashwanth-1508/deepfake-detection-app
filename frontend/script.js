// TruthLense AI v2.0 — Frontend Script

document.addEventListener('DOMContentLoaded', () => {

    const isLocal = ['localhost', '127.0.0.1', '::1', ''].some(h =>
        window.location.hostname === h || window.location.hostname.startsWith('127.')
    );
    const BASE_URL      = isLocal ? 'http://localhost:8000' : 'https://deepfake-detection-app-production.up.railway.app';
    const API_DETECT    = `${BASE_URL}/detect`;
    const API_LIVE      = `${BASE_URL}/live-detect`;
    const API_WATERMARK = `${BASE_URL}/watermark`;

    const delay = ms => new Promise(r => setTimeout(r, ms));
    let lastDetectedFaces = []; // Store latest analysis for switching

    // ─── Section Router ────────────────────────────────────────────────────
    window.showSection = (id) => {
        document.querySelectorAll('.section').forEach(s => s.style.display = 'none');
        const target = document.getElementById(`${id}-section`);
        if (target) {
            target.style.display = 'flex';
            document.querySelector('.site-overlay').style.background =
                id === 'home'
                    ? 'linear-gradient(90deg,#05060f 0%,rgba(5,6,15,.8) 50%,rgba(5,6,15,.2) 100%)'
                    : 'rgba(5,6,15,0.88)';
        }
        document.querySelectorAll('.nav-links a').forEach(a => {
            a.classList.remove('active');
            if (a.getAttribute('data-section') === id) a.classList.add('active');
        });
        window.scrollTo({ top: 0, behavior: 'smooth' });
        if (id !== 'live' && liveIntervalId) stopLiveScan();
    };

    document.querySelectorAll('nav a[data-section]').forEach(link => {
        link.addEventListener('click', e => {
            e.preventDefault();
            showSection(link.getAttribute('data-section'));
        });
    });

    const hamburger = document.getElementById('nav-hamburger');
    const navLinks  = document.querySelector('.nav-links');
    if (hamburger) hamburger.addEventListener('click', () => navLinks.classList.toggle('open'));

    // ═══════════════════════════════════════════════════════════════════════
    // DETECTION — Image / Video Upload
    // ═══════════════════════════════════════════════════════════════════════
    const fileInput      = document.getElementById('file-input');
    const detectBtn      = document.getElementById('detect-btn');
    const fileNameDisplay= document.getElementById('file-name');
    const resultDiv      = document.getElementById('result');
    const loadingDiv     = document.getElementById('loading');
    const predictionText = document.getElementById('prediction-text');
    const confidenceBar  = document.getElementById('confidence-bar');
    const confidenceText = document.getElementById('confidence-text');
    const pieLabel       = document.getElementById('pie-label');

    if (fileInput) {
        fileInput.addEventListener('change', e => {
            if (e.target.files.length > 0) {
                fileNameDisplay.textContent = `Target: ${e.target.files[0].name}`;
                detectBtn.disabled = false;
                resultDiv.style.display = 'none';
            } else {
                fileNameDisplay.textContent = '';
                detectBtn.disabled = true;
            }
        });
    }

    const dropArea = document.getElementById('drop-area');
    if (dropArea) {
        dropArea.addEventListener('dragover', e => { e.preventDefault(); dropArea.classList.add('drag-over'); });
        dropArea.addEventListener('dragleave', () => dropArea.classList.remove('drag-over'));
        dropArea.addEventListener('drop', e => {
            e.preventDefault();
            dropArea.classList.remove('drag-over');
            if (e.dataTransfer.files[0]) {
                fileInput.files = e.dataTransfer.files;
                fileNameDisplay.textContent = `Target: ${e.dataTransfer.files[0].name}`;
                detectBtn.disabled = false;
            }
        });
    }

    if (detectBtn) {
        detectBtn.addEventListener('click', async () => {
            if (!fileInput.files.length) return;
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            detectBtn.disabled = true;
            loadingDiv.style.display = 'block';
            resultDiv.style.display  = 'none';
            document.getElementById('heatmap-container').style.display   = 'none';
            document.getElementById('heatmap-img').src = '';
            document.getElementById('faces-list').style.display          = 'none';
            document.getElementById('xai-panel').style.display           = 'none';
            document.getElementById('consistency-panel').style.display   = 'none';

            try {
                const t0  = Date.now();
                const res = await fetch(API_DETECT, { method: 'POST', body: formData });
                if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Analysis failed'); }
                const data = await res.json();
                const elapsed = Date.now() - t0;
                if (elapsed < 1200) await delay(1200 - elapsed);

                loadingDiv.style.display = 'none';
                resultDiv.style.display  = 'block';
                renderDetectionResult(data);
            } catch (err) {
                console.error(err);
                alert('Analysis Error: ' + err.message);
            } finally {
                detectBtn.disabled = false;
                loadingDiv.style.display = 'none';
            }
        });
    }

    function renderDetectionResult(data) {
        const pct = (data.confidence * 100).toFixed(1);
        predictionText.textContent = data.prediction;
        confidenceText.textContent = `Confidence: ${pct}%`;

        let color = 'var(--neutral)', cls = 'undetermined';
        if (data.prediction === 'Real')                       { color = 'var(--real)';    cls = 'real'; }
        else if (data.prediction === 'Likely Real')           { color = 'var(--warn)';    cls = 'likely-real'; }
        else if (data.prediction === 'Deepfake')              { color = 'var(--fake)';    cls = 'deepfake'; }
        else if (data.prediction.includes('Undetermined'))    { color = 'var(--warn)';    cls = 'undetermined'; }

        resultDiv.className = `result-card ${cls}`;
        predictionText.style.color = color;
        confidenceBar.style.backgroundColor = color;

        setTimeout(() => {
            confidenceBar.style.width = `${pct}%`;
            const pie = document.getElementById('neural-pie');
            if (pie) {
                pie.style.background = `conic-gradient(${color} ${(data.confidence*360).toFixed(0)}deg, rgba(255,255,255,0.05) 0%)`;
                if (pieLabel) pieLabel.textContent = data.prediction === 'Real' ? 'REAL' : 'FAKE';
            }
        }, 100);

        // Face grid
        const facesList  = document.getElementById('faces-list');
        const facesBox   = document.getElementById('faces-container');
        facesBox.innerHTML = '';
        lastDetectedFaces = data.faces || [];

        if (lastDetectedFaces.length > 0) {
            facesList.style.display = 'block';
            lastDetectedFaces.slice(0, 8).forEach((face, i) => {
                const fc = face.prediction === 'Real' ? 'var(--real)' : 'var(--fake)';
                const item = document.createElement('div');
                item.className = `frame-item ${i === 0 ? 'active' : ''}`;
                item.id = `face-item-${i}`;
                item.innerHTML = `
                    <img src="data:image/jpeg;base64,${face.heatmap}" class="frame-thumbnail"
                         onclick="selectFace(${i})" alt="Target">
                    <div class="frame-score" style="color:${fc}">${(face.confidence*100).toFixed(0)}% ${face.prediction.charAt(0)}</div>`;
                facesBox.appendChild(item);
            });
            selectFace(0);
        } else {
            facesList.style.display = 'none';
        }

        if (data.consistency) renderConsistencyPanel(data.consistency);
    }

    // ─── XAI Panel ────────────────────────────────────────────────────────
    function renderXaiPanel(xai, prediction) {
        const panel = document.getElementById('xai-panel');
        panel.style.display = 'block';
        document.getElementById('xai-summary').textContent = xai.xai_summary || '';

        const reasonsEl = document.getElementById('xai-reasons');
        reasonsEl.innerHTML = '';
        (xai.reasons || []).forEach(r => {
            const chip = document.createElement('div');
            chip.className = `xai-reason-chip ${prediction === 'Real' ? 'chip-real' : 'chip-fake'}`;
            chip.textContent = r;
            reasonsEl.appendChild(chip);
        });

        const zonesEl = document.getElementById('xai-zones');
        zonesEl.innerHTML = '';
        if (xai.zone_activations) {
            const hdr = document.createElement('div');
            hdr.className = 'xai-zones-header';
            hdr.textContent = 'TOP ACTIVATION ZONES';
            zonesEl.appendChild(hdr);
            Object.entries(xai.zone_activations).forEach(([zone, val]) => {
                const pct = Math.round(val * 100);
                const c   = pct > 55 ? 'var(--fake)' : pct > 35 ? 'var(--warn)' : 'var(--real)';
                const row = document.createElement('div');
                row.className = 'zone-row';
                row.innerHTML = `
                    <span class="zone-name">${zone.replace('_',' ')}</span>
                    <div class="zone-bar-wrap"><div class="zone-bar" style="width:${pct}%;background:${c};"></div></div>
                    <span class="zone-pct" style="color:${c}">${pct}%</span>`;
                zonesEl.appendChild(row);
            });
        }
    }

    window.toggleXai = () => {
        const body = document.getElementById('xai-body');
        const icon = document.getElementById('xai-toggle-icon');
        const open = body.style.maxHeight && body.style.maxHeight !== '0px';
        body.style.maxHeight = open ? '0px' : '600px';
        icon.textContent = open ? '▼' : '▲';
    };

    window.selectFace = (index) => {
        const face = lastDetectedFaces[index];
        if (!face) return;

        // Update active thumbnail
        document.querySelectorAll('.frame-item').forEach((el, i) => {
            el.classList.toggle('active', i === index);
        });

        // Swap heatmap image
        if (face.heatmap) {
            const block = document.getElementById('heatmap-container');
            document.getElementById('heatmap-img').src = `data:image/jpeg;base64,${face.heatmap}`;
            block.style.display = 'block';
        }

        // Swap XAI panel
        if (face.xai) {
            renderXaiPanel(face.xai, face.prediction);
        }
    };

    window.showHeatmap = (b64) => {
        if (!b64 || b64 === 'null') return;
        const block = document.getElementById('heatmap-container');
        document.getElementById('heatmap-img').src = `data:image/jpeg;base64,${b64}`;
        block.style.display = 'block';
        block.scrollIntoView({ behavior: 'smooth' });
    };

    // ─── Consistency Panel ─────────────────────────────────────────────────
    function renderConsistencyPanel(c) {
        const panel = document.getElementById('consistency-panel');
        panel.style.display = 'block';
        setGauge('gauge-lip-fill',    'gauge-lip-val',    c.lip_sync?.lip_motion_score, 172);
        setGauge('gauge-light-fill',  'gauge-light-val',  c.lighting?.lighting_score,   172);
        setGauge('gauge-overall-fill','gauge-overall-val',c.overall_score,              172);

        const details = document.getElementById('consistency-details');
        const parts = [];
        if (c.lip_sync?.details)  parts.push(`Lip: ${c.lip_sync.details}`);
        if (c.lighting?.details)  parts.push(`Lighting: ${c.lighting.details}`);
        details.innerHTML = parts.map(p => `<div class="cdet-row">▶ ${p}</div>`).join('');

        const audioEl = document.getElementById('audio-status');
        if (c.audio?.audio_available) {
            const icon = c.audio.has_speech ? '🔊' : '🔇';
            audioEl.innerHTML = `${icon} Audio: ${c.audio.details}`;
        } else {
            audioEl.textContent = c.audio?.details || '';
        }
    }

    function setGauge(fillId, valId, score, arcLen) {
        const fillEl = document.getElementById(fillId);
        const valEl  = document.getElementById(valId);
        if (!fillEl || !valEl) return;
        if (score == null) {
            fillEl.setAttribute('stroke-dasharray', `0 ${arcLen}`);
            fillEl.style.stroke = 'var(--neutral)';
            valEl.textContent = 'N/A';
            return;
        }
        const dash  = (score * arcLen).toFixed(1);
        const color = score >= 0.65 ? 'var(--real)' : score >= 0.35 ? 'var(--warn)' : 'var(--fake)';
        fillEl.setAttribute('stroke-dasharray', `${dash} ${arcLen}`);
        fillEl.style.stroke  = color;
        valEl.textContent    = `${(score * 100).toFixed(0)}%`;
        valEl.style.color    = color;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // LIVE WEBCAM SCAN
    // ═══════════════════════════════════════════════════════════════════════
    let liveIntervalId = null, liveScanActive = false, liveStream = null, facingMode = 'user';

    const startBtn  = document.getElementById('start-live-btn');
    const stopBtn   = document.getElementById('stop-live-btn');
    const flipBtn   = document.getElementById('flip-cam-btn');
    const liveVideo = document.getElementById('webcam-video');
    const liveCanvas= document.getElementById('webcam-canvas');

    if (startBtn) { startBtn.addEventListener('click', startLiveScan); }
    if (stopBtn)  { stopBtn.addEventListener('click', stopLiveScan); }
    if (flipBtn)  { flipBtn.addEventListener('click', flipCamera); }

    async function startLiveScan() {
        if (window.location.protocol !== 'https:' && !isLocal) {
            alert('Live scan requires an HTTPS connection. Works on localhost.');
            return;
        }
        try {
            if (liveStream) liveStream.getTracks().forEach(t => t.stop());
            liveStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode, width: { ideal: 640 }, height: { ideal: 480 } }, audio: false
            });
            liveVideo.srcObject = liveStream;
            await liveVideo.play();
            liveScanActive = true;
            startBtn.disabled = true; stopBtn.disabled = false;
            document.getElementById('live-badge').style.display = 'flex';
            const st = document.getElementById('live-status');
            st.textContent = 'Scanning live feed...'; st.style.color = 'var(--real)';
            document.getElementById('live-result-card').style.display = 'block';
            liveIntervalId = setInterval(captureAndSend, 1000); // Calibrated to 1s
            captureAndSend();
        } catch (err) {
            const st = document.getElementById('live-status');
            st.textContent = `Camera error: ${err.message}`; st.style.color = 'var(--fake)';
        }
    }

    function stopLiveScan() {
        if (liveIntervalId) clearInterval(liveIntervalId); liveIntervalId = null;
        liveScanActive = false;
        if (liveStream) liveStream.getTracks().forEach(t => t.stop()); liveStream = null;
        liveVideo.srcObject = null;
        startBtn.disabled = false; stopBtn.disabled = true;
        document.getElementById('live-badge').style.display = 'none';
        const st = document.getElementById('live-status');
        st.textContent = 'Camera inactive'; st.style.color = 'var(--text-dim)';
        document.getElementById('live-verdict-overlay').style.display = 'none';
        document.getElementById('webcam-frame').style.borderColor = '';
    }

    async function flipCamera() {
        facingMode = facingMode === 'user' ? 'environment' : 'user';
        if (liveScanActive) { stopLiveScan(); await delay(300); startLiveScan(); }
    }

    async function captureAndSend() {
        if (!liveScanActive || !liveStream || liveVideo.videoWidth === 0) return;
        liveCanvas.width  = liveVideo.videoWidth;
        liveCanvas.height = liveVideo.videoHeight;
        liveCanvas.getContext('2d').drawImage(liveVideo, 0, 0);
        liveCanvas.toBlob(async (blob) => {
            if (!blob) return;
            const fd = new FormData(); fd.append('file', blob, 'frame.jpg');
            try {
                const res  = await fetch(API_LIVE, { method: 'POST', body: fd });
                const data = await res.json();
                renderLiveResult(data);
            } catch (e) { console.warn('Live scan frame error:', e); }
        }, 'image/jpeg', 0.82);
    }

    function renderLiveResult(data) {
        const verdict = data.prediction;
        const conf    = (data.confidence * 100).toFixed(1);
        const faces   = (data.faces || []).length;
        const color   = verdict === 'Real' ? 'var(--real)' : verdict === 'Deepfake' ? 'var(--fake)' : 'var(--warn)';

        const lrc = document.getElementById('live-result-card');
        if (lrc) lrc.style.display = 'block';
        const lrv = document.getElementById('lr-verdict');
        if (lrv) { lrv.textContent = verdict; lrv.style.color = color; }
        const lrf = document.getElementById('lr-conf');   if (lrf) lrf.textContent = `${conf}%`;
        const lrfc= document.getElementById('lr-faces');  if (lrfc) lrfc.textContent = faces;

        const overlay = document.getElementById('live-verdict-overlay');
        const predEl  = document.getElementById('live-pred');
        const confEl  = document.getElementById('live-conf');
        if (overlay) overlay.style.display = 'flex';
        if (predEl)  { predEl.textContent = verdict; predEl.style.color = color; }
        if (confEl)    confEl.textContent = `${conf}% confidence`;

        const frame = document.getElementById('webcam-frame');
        if (frame) frame.style.borderColor = color;

        // History
        const history = document.getElementById('live-history');
        if (history) {
            const now  = new Date().toLocaleTimeString([], { hour:'2-digit', minute:'2-digit', second:'2-digit' });
            const item = document.createElement('div');
            item.className = 'history-item';
            item.innerHTML = `<span class="history-time">${now}</span>
                              <span class="history-verdict" style="color:${color}">${verdict}</span>
                              <span class="history-conf">${conf}%</span>`;
            history.prepend(item);
            while (history.children.length > 20) history.removeChild(history.lastChild);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // WATERMARK DETECTION
    // ═══════════════════════════════════════════════════════════════════════
    const wmFileInput = document.getElementById('wm-file-input');
    const wmDetectBtn = document.getElementById('wm-detect-btn');
    const wmFileName  = document.getElementById('wm-file-name');
    const wmLoading   = document.getElementById('wm-loading');
    const wmResult    = document.getElementById('wm-result');
    const wmDropArea  = document.getElementById('wm-drop-area');

    if (wmFileInput) {
        wmFileInput.addEventListener('change', e => {
            if (e.target.files.length > 0) {
                wmFileName.textContent = `File: ${e.target.files[0].name}`;
                wmDetectBtn.disabled   = false;
                wmResult.style.display = 'none';
            }
        });
    }
    if (wmDropArea) {
        wmDropArea.addEventListener('dragover', e => { e.preventDefault(); wmDropArea.classList.add('drag-over'); });
        wmDropArea.addEventListener('dragleave', () => wmDropArea.classList.remove('drag-over'));
        wmDropArea.addEventListener('drop', e => {
            e.preventDefault(); wmDropArea.classList.remove('drag-over');
            if (e.dataTransfer.files[0]) {
                wmFileInput.files = e.dataTransfer.files;
                wmFileName.textContent   = `File: ${e.dataTransfer.files[0].name}`;
                wmDetectBtn.disabled     = false;
                wmResult.style.display   = 'none';
            }
        });
    }

    if (wmDetectBtn) {
        wmDetectBtn.addEventListener('click', async () => {
            if (!wmFileInput.files.length) return;
            const fd = new FormData(); fd.append('file', wmFileInput.files[0]);
            wmDetectBtn.disabled   = true;
            wmLoading.style.display= 'block';
            wmResult.style.display = 'none';
            try {
                const t0  = Date.now();
                const res = await fetch(API_WATERMARK, { method: 'POST', body: fd });
                if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Watermark analysis failed'); }
                const data = await res.json();
                if (Date.now() - t0 < 800) await delay(800 - (Date.now() - t0));
                wmLoading.style.display= 'none';
                wmResult.style.display = 'block';
                renderWatermarkResult(data);
            } catch (err) {
                console.error(err); alert('Watermark Error: ' + err.message);
            } finally {
                wmDetectBtn.disabled   = false;
                wmLoading.style.display= 'none';
            }
        });
    }

    function renderWatermarkResult(data) {
        const score = data.watermark_score;
        const pct   = Math.round(score * 100);
        const circ  = 2 * Math.PI * 50;
        const dash  = (score * circ).toFixed(1);
        const color = score >= 0.65 ? 'var(--fake)' : score >= 0.35 ? 'var(--warn)' : 'var(--real)';

        const ringFill = document.getElementById('wm-ring-fill');
        const ringVal  = document.getElementById('wm-ring-value');
        setTimeout(() => { ringFill.setAttribute('stroke-dasharray', `${dash} ${circ}`); }, 100);
        ringFill.style.stroke = color;
        ringVal.textContent   = `${pct}%`;
        ringVal.style.color   = color;

        const vt = document.getElementById('wm-verdict-text');
        vt.textContent = score >= 0.65 ? '⚠ AI-GENERATED' : score >= 0.35 ? '? POSSIBLY AI' : '✓ LIKELY AUTHENTIC';
        vt.style.color = color;

        const list = document.getElementById('wm-artifacts-list');
        list.innerHTML = '';
        (data.artifacts_found || []).forEach(a => {
            const chip = document.createElement('div');
            chip.className = `artifact-chip ${a === 'None detected' ? 'chip-real' : 'chip-fake'}`;
            chip.textContent = a; list.appendChild(chip);
        });

        document.getElementById('wm-details').textContent = data.details || '';
        const checker = document.getElementById('wm-checker');
        if (checker) checker.textContent = data.checkerboard_ratio != null ? data.checkerboard_ratio.toFixed(4) : '—';
    }

    // Boot
    showSection('home');
});
