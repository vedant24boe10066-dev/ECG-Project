with open("templates/index.html", "r") as f:
    html = f.read()

# 1. Add progress bar before resultsPanel
html = html.replace('        <!-- Results Section (Hidden initially) -->', '''        <!-- Progress Bar (Hidden by default) -->
        <div id="progressContainer" class="hidden bg-bgSurface p-6 rounded-xl border border-gray-800 mb-6">
            <h2 class="text-sm font-semibold text-textMuted mb-2">Processing Pipeline Status</h2>
            <div class="w-full bg-gray-900 rounded-full h-2.5 mb-2">
                <div id="progressBar" class="bg-accentPrimary h-2.5 rounded-full transition-all duration-300" style="width: 0%"></div>
            </div>
            <p id="progressMessage" class="text-xs text-textPrimary animate-pulse">Initializing...</p>
        </div>

        <!-- Results Section (Hidden initially) -->''')

# 2. Update clear logic
html = html.replace('''        document.getElementById('clearBtn').addEventListener('click', () => {
            // Reset the file input and stop animation
            document.getElementById('uploadForm').reset();
            if(animationInterval) clearInterval(animationInterval);
            if(ecgChart) {
                ecgChart.destroy();
                ecgChart = null;
            }
            
            // Hide panels
            document.getElementById('ecgMonitorPanel').classList.add('hidden');
            document.getElementById('resultsPanel').classList.add('hidden');
            document.getElementById('error').classList.add('hidden');
            document.getElementById('loading').classList.add('hidden');
        });''', '''        document.getElementById('clearBtn').addEventListener('click', () => {
            // Reset the file input and stop animation
            document.getElementById('uploadForm').reset();
            if(animationInterval) clearInterval(animationInterval);
            if(ecgChart) {
                ecgChart.destroy();
                ecgChart = null;
            }
            
            // Hide panels
            document.getElementById('ecgMonitorPanel').classList.add('hidden');
            document.getElementById('resultsPanel').classList.add('hidden');
            document.getElementById('error').classList.add('hidden');
            document.getElementById('loading').classList.add('hidden');
            document.getElementById('progressContainer').classList.add('hidden');
        });''')

# 3. Update fetch logic
fetch_old = '''            try {
                const response = await fetch(`/api/analyze?sampling_rate=${samplingRate}`, {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.detail || 'Analysis failed');
                }

                renderResults(data);
                
            } catch (err) {
                error.textContent = err.message;
                error.classList.remove('hidden');
                clearInterval(animationInterval);
            } finally {
                loading.classList.add('hidden');
            }'''

fetch_new = '''            document.getElementById('progressContainer').classList.remove('hidden');
            document.getElementById('progressBar').style.width = '0%';
            document.getElementById('progressBar').className = 'bg-accentPrimary h-2.5 rounded-full transition-all duration-300';
            document.getElementById('progressMessage').innerText = 'Initializing...';
            document.getElementById('progressMessage').className = 'text-xs text-textPrimary animate-pulse';
            
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
                        try {
                            const msg = JSON.parse(line);
                            if (msg.status === 'progress') {
                                document.getElementById('progressBar').style.width = msg.progress + '%';
                                document.getElementById('progressMessage').innerText = msg.message;
                            } else if (msg.status === 'complete') {
                                document.getElementById('progressBar').style.width = '100%';
                                document.getElementById('progressMessage').innerText = 'Complete!';
                                document.getElementById('progressMessage').classList.remove('animate-pulse');
                                document.getElementById('progressMessage').classList.add('text-green-400');
                                
                                setTimeout(() => {
                                    document.getElementById('progressContainer').classList.add('hidden');
                                    renderResults(msg.data);
                                }, 600);
                            } else if (msg.status === 'error') {
                                throw new Error(msg.message);
                            }
                        } catch (parseErr) {
                            console.error("Failed to parse JSON line:", line, parseErr);
                        }
                    }
                }
            } catch (err) {
                console.error(err);
                document.getElementById('progressMessage').innerText = `Error: ${err.message}`;
                document.getElementById('progressMessage').className = 'text-xs text-red-500 font-semibold';
                document.getElementById('progressBar').className = 'bg-red-500 h-2.5 rounded-full transition-all duration-300';
                clearInterval(animationInterval);
            } finally {
                loading.classList.add('hidden');
            }'''

html = html.replace(fetch_old, fetch_new)

with open("templates/index.html", "w") as f:
    f.write(html)
print("Updated index.html correctly")
