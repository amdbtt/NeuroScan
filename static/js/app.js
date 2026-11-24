document.addEventListener('DOMContentLoaded', () => {
    // Server-verified class order for BrainTumor_VGG16.h5
    const DEFAULT_LABELS = [
        "Other / Non-MRI",
        "Meningioma",
        "No Tumor",
        "Glioma",
        "Pituitary Tumor"
    ];

    // Bump key so stale wrong mappings in localStorage are discarded
    const LABEL_STORAGE_KEY = 'neuroscan_labels_v2';

    let classLabels = [...DEFAULT_LABELS];
    let currentPredictions = null;
    let currentInferenceTime = 0;

    const uploadPlaceholder = document.getElementById('upload-placeholder');
    const imageInput = document.getElementById('image-input');
    const imagePreview = document.getElementById('image-preview');
    const mriViewer = document.getElementById('mri-viewer');
    const selectImageBtn = document.getElementById('select-image-btn');
    const resetImageBtn = document.getElementById('reset-image-btn');
    const scanningOverlay = document.getElementById('scanning-overlay');

    const diagnosticsIdle = document.getElementById('diagnostics-idle');
    const diagnosticsResults = document.getElementById('diagnostics-results');

    const diagnosisCard = document.getElementById('diagnosis-card');
    const primaryDiagnosis = document.getElementById('primary-diagnosis');
    const primaryConfidence = document.getElementById('primary-confidence');
    const riskBadge = document.getElementById('risk-badge');
    const probabilitiesList = document.getElementById('probabilities-list');
    const metaTime = document.getElementById('meta-time');

    const settingsModal = document.getElementById('settings-modal');
    const openSettingsBtn = document.getElementById('open-settings-btn');
    const closeSettingsBtn = document.getElementById('close-settings-btn');
    const modalBackdrop = document.getElementById('modal-backdrop');
    const labelConfigForm = document.getElementById('label-config-form');
    const resetLabelsBtn = document.getElementById('reset-labels-btn');

    function loadLabelsFromStorage() {
        // Drop obsolete v1 mappings that had the wrong class order
        localStorage.removeItem('neuroscan_labels');

        const stored = localStorage.getItem(LABEL_STORAGE_KEY);
        if (stored) {
            try {
                const parsed = JSON.parse(stored);
                if (Array.isArray(parsed) && parsed.length === 5) {
                    classLabels = parsed;
                }
            } catch (e) {
                console.error("Error parsing labels from localStorage", e);
            }
        }
    }

    function saveLabelsToStorage() {
        localStorage.setItem(LABEL_STORAGE_KEY, JSON.stringify(classLabels));
    }

    function populateSettingsInputs() {
        classLabels.forEach((label, idx) => {
            const input = document.getElementById(`label-${idx}`);
            if (input) input.value = label;
        });
    }

    loadLabelsFromStorage();

    // Prefer authoritative labels from the server
    fetch('/labels')
        .then(r => r.json())
        .then(data => {
            if (data.success && Array.isArray(data.labels) && data.labels.length === 5) {
                // Only override if user has not customized v2 labels
                if (!localStorage.getItem(LABEL_STORAGE_KEY)) {
                    classLabels = data.labels;
                }
            }
        })
        .catch(() => {});

    ['dragenter', 'dragover'].forEach(eventName => {
        mriViewer.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            mriViewer.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        mriViewer.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            mriViewer.classList.remove('dragover');
        }, false);
    });

    mriViewer.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) handleImageFile(files[0]);
    });

    uploadPlaceholder.addEventListener('click', () => imageInput.click());
    selectImageBtn.addEventListener('click', () => imageInput.click());

    imageInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleImageFile(e.target.files[0]);
    });

    resetImageBtn.addEventListener('click', () => {
        imageInput.value = '';
        imagePreview.src = '';
        imagePreview.classList.add('hidden');
        uploadPlaceholder.classList.remove('hidden');
        resetImageBtn.classList.add('hidden');
        mriViewer.classList.remove('scanning');
        scanningOverlay.classList.add('hidden');

        diagnosticsResults.classList.add('hidden');
        diagnosticsIdle.classList.remove('hidden');

        currentPredictions = null;
        currentInferenceTime = 0;
    });

    openSettingsBtn.addEventListener('click', () => {
        populateSettingsInputs();
        settingsModal.classList.remove('hidden');
    });

    const hideSettings = () => settingsModal.classList.add('hidden');
    closeSettingsBtn.addEventListener('click', hideSettings);
    modalBackdrop.addEventListener('click', hideSettings);

    labelConfigForm.addEventListener('submit', (e) => {
        e.preventDefault();
        classLabels = [0, 1, 2, 3, 4].map(idx =>
            document.getElementById(`label-${idx}`).value.trim()
        );
        saveLabelsToStorage();
        hideSettings();
        if (currentPredictions) {
            renderPredictions(currentPredictions, currentInferenceTime);
        }
    });

    resetLabelsBtn.addEventListener('click', () => {
        DEFAULT_LABELS.forEach((label, idx) => {
            document.getElementById(`label-${idx}`).value = label;
        });
    });

    function stopScanningUI() {
        scanningOverlay.classList.add('hidden');
        mriViewer.classList.remove('scanning');
    }

    function handleImageFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Invalid file format. Please upload an image file (MRI scan).');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.classList.remove('hidden');
            uploadPlaceholder.classList.add('hidden');
            resetImageBtn.classList.remove('hidden');
        };
        reader.readAsDataURL(file);

        mriViewer.classList.add('scanning');
        scanningOverlay.classList.remove('hidden');
        diagnosticsIdle.classList.add('hidden');
        diagnosticsResults.classList.add('hidden');

        const formData = new FormData();
        formData.append('file', file);

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(async response => {
            const data = await response.json().catch(() => ({}));
            if (!response.ok || !data.success) {
                throw new Error(data.error || 'Prediction API responded with an error');
            }
            return data;
        })
        .then(data => {
            // Prefer server labels when present (authoritative for this model)
            if (Array.isArray(data.labels) && data.labels.length === 5) {
                classLabels = data.labels;
            }
            currentPredictions = data.predictions;
            currentInferenceTime = data.inference_time_ms;
            renderPredictions(data.predictions, data.inference_time_ms, data);
        })
        .catch(error => {
            console.error('Error during image processing:', error);
            alert(error.message || 'Failed to connect to the diagnostic server.');
            resetImageBtn.click();
        })
        .finally(() => {
            stopScanningUI();
        });
    }

    function classifyRisk(label) {
        const lower = (label || '').toLowerCase();
        if (lower.includes('no tumor') || lower.includes('healthy') || lower.includes('normal')) {
            return 'normal';
        }
        if (lower.includes('other') || lower.includes('non-mri') || lower.includes('nonmri')) {
            return 'warning';
        }
        return 'tumor';
    }

    function renderPredictions(predictions, inferenceTimeMs, serverData = null) {
        if (!Array.isArray(predictions) || predictions.length === 0) {
            alert('Server returned empty predictions.');
            return;
        }

        let highestIndex = 0;
        let highestProb = -1;
        predictions.forEach((prob, idx) => {
            const value = Number(prob);
            if (value > highestProb) {
                highestProb = value;
                highestIndex = idx;
            }
        });

        if (serverData && typeof serverData.predicted_index === 'number') {
            highestIndex = serverData.predicted_index;
            highestProb = Number(serverData.confidence ?? predictions[highestIndex]);
        }

        const predictedLabel = (serverData && serverData.predicted_label)
            || classLabels[highestIndex]
            || `Class ${highestIndex}`;
        const percentString = (highestProb * 100).toFixed(1) + '%';
        const risk = classifyRisk(predictedLabel);

        primaryDiagnosis.textContent = predictedLabel.toUpperCase();
        primaryConfidence.textContent = percentString;
        metaTime.textContent = Number(inferenceTimeMs).toFixed(1) + ' ms';

        diagnosisCard.className = 'diagnosis-card ' + risk;
        if (risk === 'normal') {
            riskBadge.textContent = 'LOW RISK';
        } else if (risk === 'warning') {
            riskBadge.textContent = 'UNCERTAIN';
        } else {
            riskBadge.textContent = 'HIGH RISK';
        }
        riskBadge.style.display = 'inline-block';

        probabilitiesList.innerHTML = '';
        predictions.forEach((prob, idx) => {
            const label = classLabels[idx] || `Class ${idx}`;
            const isHighest = idx === highestIndex;
            const value = Number(prob);
            const probPct = (value * 100).toFixed(1) + '%';

            const item = document.createElement('div');
            item.className = 'prob-item' + (isHighest ? ' highest ' + risk : '');

            item.innerHTML = `
                <div class="prob-header">
                    <span class="prob-label">${label}</span>
                    <span class="prob-value">${probPct}</span>
                </div>
                <div class="prob-bar-container">
                    <div class="prob-bar" style="width: 0%"></div>
                </div>
            `;

            probabilitiesList.appendChild(item);

            requestAnimationFrame(() => {
                const bar = item.querySelector('.prob-bar');
                if (bar) bar.style.width = Math.max(0, Math.min(100, value * 100)) + '%';
            });
        });

        diagnosticsIdle.classList.add('hidden');
        diagnosticsResults.classList.remove('hidden');
    }
});
