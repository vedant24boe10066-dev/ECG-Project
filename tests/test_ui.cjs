// Exercise browser logic with a small DOM/canvas stub; no external browser libraries.
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');
const html = fs.readFileSync('templates/index.html', 'utf8');
const script = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].at(-1)[1];
function element() {
    const classes = new Set();
    return {children:[], style:{}, listeners:{}, value:'250', files:[], disabled:false,
        classList:{add(value){classes.add(value);},remove(value){classes.delete(value);},contains(value){return classes.has(value);},toggle(value,force){if(force)classes.add(value);else classes.delete(value);}},
        append(...values){this.children.push(...values);},appendChild(value){this.children.push(value);},
        replaceChildren(){this.children=[];},reset(){this.resetCalled=true;},
        addEventListener(name,handler){this.listeners[name]=handler;},
        click(){this.clicked=true;},remove(){this.removed=true;},getBoundingClientRect(){return {width:600,height:195};},
        getContext(){return {scale(){},clearRect(){},beginPath(){},moveTo(){},lineTo(){},stroke(){},fillText(){},setLineDash(){},arc(){},fill(){}}}};
}
const nodes=new Map();
const document={body:element(),getElementById(id){if(!nodes.has(id)) nodes.set(id,element());return nodes.get(id);},createElement:element};
const ctx={document,window:{devicePixelRatio:1,addEventListener(){}},console,AbortController,TextDecoder,Blob,URL:{createObjectURL(){return 'blob:test';},revokeObjectURL(){}},setTimeout(){},
    setInterval(){return 1;},clearInterval(){},
    FileReader:function(){this.readAsText=file=>this.onload({target:{result:file.text}});},
    FormData:function(){this.append=()=>{};}};
vm.createContext(ctx);vm.runInContext(script,ctx);
ctx.file={name:'wave.csv',text:'time,ECG\r\n'+Array.from({length:1200},(_,i)=>`${i},${Math.sin(i)}`).join('\r\n')};
vm.runInContext('startECGAnimation(file,250)',ctx);
assert.equal(vm.runInContext('previewPoints.length',ctx),1200);
assert.equal(vm.runInContext('previewPoints[1]',ctx),Math.sin(1));
assert(!nodes.get('ecgMonitorPanel').classList.contains('hidden'));
ctx.file={name:'record.dat',text:'binary'};vm.runInContext('startECGAnimation(file,250)',ctx);
assert.equal(vm.runInContext('previewPoints.length',ctx),1200);
const report={report_id:'TEST-REPORT',timestamp:'2026-10-06T12:00:00Z',source_file:'wave.csv',prediction:{is_demo:true,prediction_label:'Stress',confidence_score:.84,top_shap_features:[{feature_name:'SDNN',feature_value:12,shap_value:0}]},key_hrv_summary:{SDNN:12,LF_HF_Ratio:null},pipeline_metadata:{duration_s:60,r_peaks_detected:89,sampling_rate_hz:250,warnings:['LF unavailable']}};
report.signal_analytics={time_seconds:[1,2,3],rr_intervals_ms:[1000,1100,900],heart_rate_bpm:[60,54.5,66.7],consecutive_rr_ms:[[1000,1100],[1100,900]],total_intervals:3,mean_rr_ms:1000,mean_heart_rate_bpm:60,min_heart_rate_bpm:54.5,max_heart_rate_bpm:66.7};
ctx.report=report;vm.runInContext('renderResults(report)',ctx);
assert(nodes.get('modelNotice').textContent.includes('Demo model'));
assert.equal(nodes.get('predictionLabel').textContent,'Stress');
assert.equal(nodes.get('hrvList').children[1].children[1].textContent,'Unavailable');
assert.equal(nodes.get('shapList').children[0].children[1].children[0].style.width,'0%');
assert.equal(nodes.get('rateValue').textContent,'250 Hz');
assert(!nodes.get('analyticsPanel').classList.contains('hidden'));
assert.equal(nodes.get('meanHeartRate').textContent,'60.0 bpm');
assert.equal(vm.runInContext('resultAccent',ctx),'#F87171');
assert(nodes.get('poincareChart').width>0);
async function submitMessages(text, compare=false) {
    document.getElementById('modelSelect').value='svm';
    document.getElementById('ecgFile').files=[{name:'wave.csv',text:ctx.file.text}];
    const encoded=new TextEncoder().encode(text);
    ctx.fetch=async(url)=>{assert(url.includes('&model_name=svm'));assert(url.includes('&compare_models='+compare));return {ok:true,body:{getReader(){let index=0;const chunks=[encoded.slice(0,20),encoded.slice(20)];return {async read(){return index<chunks.length ? {done:false,value:chunks[index++]} : {done:true};}};}}};};
    await nodes.get('uploadForm').listeners.submit({preventDefault(){},submitter:compare ? {id:'compareBtn'} : undefined});
}
(async()=>{
    await submitMessages(JSON.stringify({status:'complete',data:report})); // Final line without a newline.
    assert.equal(nodes.get('confidenceScore').textContent,'84.0%');
    assert.equal(nodes.get('analyzeBtn').disabled,false);
    assert.equal(nodes.get('modelSelect').disabled,false);
    await submitMessages(JSON.stringify({status:'error',message:'Invalid ECG'})+'\n');
    assert.equal(nodes.get('error').textContent,'Invalid ECG');
    await submitMessages(JSON.stringify({status:'progress',progress:30,message:'Filtering'})+'\n');
    assert(nodes.get('error').textContent.includes('before a result'));
    const comparisonReport={...report,pipeline_metadata:{...report.pipeline_metadata,model_id:'random_forest'},model_comparison:[
        {model_id:'random_forest',model_name:'Random Forest',prediction:report.prediction,prediction_time_ms:10,explanation_time_ms:5,included_training:true,explanation_method:'TreeSHAP'},
        {model_id:'svm',model_name:'SVM (RBF)',prediction:{...report.prediction,prediction_label:'No-Stress',stress_probability:.2,confidence_score:.8},prediction_time_ms:3,explanation_time_ms:100,explanation_method:'KernelSHAP'},
        {model_id:'xgboost',model_name:'XGBoost',prediction:null,error:'Runtime unavailable'},
    ]};
    await submitMessages(JSON.stringify({status:'complete',data:comparisonReport})+'\n',true);
    assert.equal(nodes.get('comparisonRows').children.length,3);
    nodes.get('comparisonRows').children[1].children[6].children[0].listeners.click();
    assert.equal(nodes.get('predictionLabel').textContent,'No-Stress');
    assert.equal(vm.runInContext('resultAccent',ctx),'#4ADE80');
    assert(!nodes.get('analyticsPanel').classList.contains('hidden'));
    assert.equal(nodes.get('resultModelValue').textContent,'SVM (RBF)');
    nodes.get('downloadJsonBtn').listeners.click();
    assert.equal(document.body.children.at(-1).download,'TEST-REPORT.json');
    ctx.fetch=async(url,options)=>{assert.equal(url,'/api/report.pdf');assert.equal(JSON.parse(options.body).model_comparison.length,3);return {ok:true,blob:async()=>new Blob(['%PDF-test'],{type:'application/pdf'})};};
    await nodes.get('downloadPdfBtn').listeners.click();
    assert.equal(document.body.children.at(-1).download,'TEST-REPORT.pdf');
    assert.equal(nodes.get('downloadPdfBtn').disabled,false);
    nodes.get('clearBtn').listeners.click();
    assert(nodes.get('resultsPanel').classList.contains('hidden'));
    assert(nodes.get('uploadForm').resetCalled);
    assert(nodes.get('analyticsPanel').classList.contains('hidden'));
    assert.equal(nodes.get('analyzeBtn').disabled,false);
    assert(html.includes('color-scheme:dark'));
    console.log('PASS: comparison requests/table/model explanations, PDF/JSON downloads, dark theme, model-selection request, preview parsing, waveform rendering, demo notice, null metrics, SHAP, split streaming, errors, truncated responses, and Clear.');
})().catch(error=>{console.error(error);process.exitCode=1;});
