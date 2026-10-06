import re

with open('templates/index.html', 'r') as f:
    content = f.read()

# 1. Insert progress bar HTML before resultsSection
progress_html = """
        <!-- Progress Bar (Hidden by default) -->
        <div id="progressContainer" class="hidden bg-bgSurface p-6 rounded-xl border border-gray-800 mb-6">
            <h2 class="text-sm font-semibold text-textMuted mb-2">Processing Pipeline Status</h2>
            <div class="w-full bg-gray-900 rounded-full h-2.5 mb-2">
                <div id="progressBar" class="bg-accentPrimary h-2.5 rounded-full transition-all duration-300" style="width: 0%"></div>
            </div>
            <p id="progressMessage" class="text-xs text-textPrimary animate-pulse">Initializing...</p>
        </div>

        <!-- Results Section (Hidden by default) -->
"""
content = content.replace('        <!-- Results Section (Hidden by default) -->', progress_html)

# 2. Update JS Variables
js_vars_old = """        const resultsSection = document.getElementById('resultsSection');
        const ecgChartCanvas = document.getElementById('ecgChart');"""
js_vars_new = """        const resultsSection = document.getElementById('resultsSection');
        const progressContainer = document.getElementById('progressContainer');
        const progressBar = document.getElementById('progressBar');
        const progressMessage = document.getElementById('progressMessage');
        const ecgChartCanvas = document.getElementById('ecgChart');"""
content = content.replace(js_vars_old, js_vars_new)

# 3. Update Clear Button
clear_old = """        clearBtn.addEventListener('click', () => {
            ecgFile.value = '';
            resultsSection.classList.add('hidden');
            if(chartInstance) {"""
clear_new = """        clearBtn.addEventListener('click', () => {
            ecgFile.value = '';
            resultsSection.classList.add('hidden');
            progressContainer.classList.add('hidden');
            if(chartInstance) {"""
content = content.replace(clear_old, clear_new)

# 4. Update form submission fetch logic
fetch_old = """            // Reset UI
            resultsSection.classList.add('hidden');
            // Start cute graph animation
            startECGAnimation(fileInput.files[0]);

            const formData = new FormData();
            for (let i = 0; i < fileInput.files.length; i++) {
                formData.append('files', fileInput.files[i]);
            }

            try {
                const response = await fetch(`/api/analyze?sampling_rate=${samplingRate}`, {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const error = await response.json();
                    alert(`Error: ${error.detail}`);
                    return;
                }

                const data = await response.json();
                renderResults(data);
                
            } catch (err) {
                console.error(err);
                alert("An error occurred during analysis.");
            }"""
            
fetch_new = """            // Reset UI
            resultsSection.classList.add('hidden');
            progressContainer.classList.remove('hidden');
            progressBar.style.width = '0%';
            progressBar.className = 'bg-accentPrimary h-2.5 rounded-full transition-all duration-300';
            progressMessage.innerText = 'Initializing...';
            progressMessage.className = 'text-xs text-textPrimary animate-pulse';

            // Start cute graph animation
            startECGAnimation(fileInput.files[0]);

            const formData = new FormData();
            for (let i = 0; i < fileInput.files.length; i++) {
                formData.append('files', fileInput.files[i]);
            }

            try {
                const response = await fetch(`/api/analyze?sampling_rate=${samplingRate}`, {
                    method: 'POST',
                    body: formData
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\\n');
                    buffer = lines.pop(); // Keep incomplete line
                    
                    for (const line of lines) {
                        if (!line.trim()) continue;
                        const msg = JSON.parse(line);
                        
                        if (msg.status === 'progress') {
                            progressBar.style.width = msg.progress + '%';
                            progressMessage.innerText = msg.message;
                        } else if (msg.status === 'complete') {
                            progressBar.style.width = '100%';
                            progressMessage.innerText = 'Complete!';
                            progressMessage.classList.remove('animate-pulse');
                            progressMessage.classList.add('text-green-400');
                            
                            setTimeout(() => {
                                progressContainer.classList.add('hidden');
                                renderResults(msg.data);
                            }, 600);
                        } else if (msg.status === 'error') {
                            throw new Error(msg.message);
                        }
                    }
                }
            } catch (err) {
                console.error(err);
                progressMessage.innerText = `Error: ${err.message}`;
                progressMessage.className = 'text-xs text-red-500 font-semibold';
                progressBar.className = 'bg-red-500 h-2.5 rounded-full transition-all duration-300';
            }"""
content = content.replace(fetch_old, fetch_new)

with open('templates/index.html', 'w') as f:
    f.write(content)
print("Updated index.html")
