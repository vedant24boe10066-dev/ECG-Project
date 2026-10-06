with open("templates/index.html", "r") as f:
    content = f.read()

bad_js = """                        try {
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
                        }"""

good_js = """                        let msg;
                        try {
                            msg = JSON.parse(line);
                        } catch (parseErr) {
                            console.error("Failed to parse JSON line:", line, parseErr);
                            continue;
                        }
                        
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
                        }"""

content = content.replace(bad_js, good_js)
with open("templates/index.html", "w") as f:
    f.write(content)
