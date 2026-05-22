const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadSection = document.querySelector('.upload-section');
const loadingState = document.getElementById('loading');
const resultsPanel = document.getElementById('results-panel');

let probChartInstance = null;

// Drag and Drop Events
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
});

dropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
});

fileInput.addEventListener('change', function() {
    handleFiles(this.files);
});

function handleFiles(files) {
    if (files.length > 0) {
        uploadFile(files[0]);
    }
}

function uploadFile(file) {
    // Show loading UI
    dropZone.classList.add('hidden');
    loadingState.classList.remove('hidden');
    resultsPanel.classList.add('hidden');

    const formData = new FormData();
    formData.append('file', file);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert("Error: " + data.error);
            resetApp();
            return;
        }
        displayResults(data);
    })
    .catch(error => {
        console.error('Error:', error);
        alert("An error occurred during prediction.");
        resetApp();
    });
}

function displayResults(data) {
    // Hide loading, show results
    loadingState.classList.add('hidden');
    uploadSection.classList.add('hidden');
    resultsPanel.classList.remove('hidden');

    // Update Badge
    const badge = document.getElementById('diagnosis-badge');
    const diagText = document.getElementById('diagnosis-text');
    const confText = document.getElementById('confidence-text');

    diagText.textContent = data.prediction;
    confText.textContent = data.confidence.toFixed(1) + '%';

    if (data.prediction === 'NORMAL') {
        badge.className = 'diagnosis-badge diagnosis-safe';
    } else {
        badge.className = 'diagnosis-badge diagnosis-danger';
    }

    // Update Images
    document.getElementById('orig-img').src = data.original_image;
    document.getElementById('heatmap-img').src = data.heatmap_image;

    // Render Chart
    renderChart(data.probabilities);
}

function renderChart(probs) {
    const ctx = document.getElementById('probChart').getContext('2d');
    
    if (probChartInstance) {
        probChartInstance.destroy();
    }

    const labels = Object.keys(probs);
    const data = Object.values(probs);

    Chart.defaults.color = '#94a3b8';
    Chart.defaults.font.family = "'Inter', sans-serif";

    probChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Confidence (%)',
                data: data,
                backgroundColor: [
                    labels[0] === 'NORMAL' ? 'rgba(16, 185, 129, 0.6)' : 'rgba(239, 68, 68, 0.6)',
                    labels[1] === 'NORMAL' ? 'rgba(16, 185, 129, 0.6)' : 'rgba(239, 68, 68, 0.6)',
                    labels[2] === 'NORMAL' ? 'rgba(16, 185, 129, 0.6)' : 'rgba(239, 68, 68, 0.6)'
                ],
                borderColor: [
                    labels[0] === 'NORMAL' ? 'rgb(16, 185, 129)' : 'rgb(239, 68, 68)',
                    labels[1] === 'NORMAL' ? 'rgb(16, 185, 129)' : 'rgb(239, 68, 68)',
                    labels[2] === 'NORMAL' ? 'rgb(16, 185, 129)' : 'rgb(239, 68, 68)'
                ],
                borderWidth: 1,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }
                },
                x: {
                    grid: { display: false }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

function resetApp() {
    resultsPanel.classList.add('hidden');
    uploadSection.classList.remove('hidden');
    loadingState.classList.add('hidden');
    dropZone.classList.remove('hidden');
    fileInput.value = '';
}
