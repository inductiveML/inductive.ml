import {useEffect,useRef,useState} from 'react';
import MonarchMarkdown from './MonarchMarkdown';
import {monarchPrompts} from '../monarch/prompts';
type Mode='generate'|'benchmark';
type Bench={depth:number;tps:number;ms:number;runs:number;tokens:number;runTps:number[];spreadPct:number;device:string;measuredAt:string;protocol:string;exactRepeats:boolean};
export default function MonarchDemo(){
 const worker=useRef<Worker|null>(null),timeout=useRef<ReturnType<typeof setTimeout>|null>(null);
 const [mode,setMode]=useState<Mode>('generate'),[loaded,setLoaded]=useState(false),[busy,setBusy]=useState(false),[loading,setLoading]=useState(false);
 const [status,setStatus]=useState('The model loads only when you ask.'),[error,setError]=useState(''),[device,setDevice]=useState('Your browser · your GPU');
 const [progress,setProgress]=useState(0),[prompt,setPrompt]=useState<string>(monarchPrompts[0].text),[limit,setLimit]=useState(256),[depth,setDepth]=useState(192);
 const [output,setOutput]=useState(''),[tokens,setTokens]=useState(0),[tps,setTps]=useState<number|null>(null),[ttft,setTtft]=useState<number|null>(null);
 const [bench,setBench]=useState<Bench|null>(null),[runCount,setRunCount]=useState(0);
 const clearTimer=()=>{if(timeout.current)clearTimeout(timeout.current);timeout.current=null;};
 const release=()=>{worker.current?.terminate();worker.current=null;clearTimer();setLoaded(false);setBusy(false);setLoading(false);};
 useEffect(()=>{const hide=()=>{if(document.hidden)worker.current?.postMessage({type:'cancel'});};document.addEventListener('visibilitychange',hide);return()=>{worker.current?.terminate();if(timeout.current)clearTimeout(timeout.current);document.removeEventListener('visibilitychange',hide);};},[]);
 function createWorker(){
  const w=new Worker('/monarch/worker.js',{type:'module'});worker.current=w;
  w.onmessage=({data})=>{
   if(data.type==='support'){
    if(!data.supported){release();setError(data.reason);setStatus('This device cannot run the current kernel.');}
    else setDevice(data.device);
   }else if(data.type==='progress'){setStatus(data.stage);setProgress(data.total?data.loaded/data.total:0);}
   else if(data.type==='loaded'){clearTimer();setLoaded(true);setBusy(false);setLoading(false);setStatus('Ready · 32-token correctness check passed');setDevice(data.device);}
   else if(data.type==='generating'){setStatus(`Reading your ${data.promptTokens}-token prompt…`);}
   else if(data.type==='text'||data.type==='generated'){
    setOutput(data.text);setTokens(data.tokens);setTps(data.tps);setTtft(data.ttft);
    if(data.type==='generated'){clearTimer();setBusy(false);setStatus(data.stopped?'Stopped. Ready for another prompt.':'Finished · generated locally on your GPU');}else setStatus('Generating on your GPU…');
   }else if(data.type==='benchmarkProgress'){setRunCount(data.completed);setStatus(data.stage);}
   else if(data.type==='benchmarkResult'){clearTimer();setBench(data);setBusy(false);setStatus('Benchmark complete · all four continuations matched');}
   else if(data.type==='cancelled'){clearTimer();setBusy(false);setStatus('Benchmark stopped. No result recorded.');}
   else if(data.type==='error'){clearTimer();setError(data.message);setBusy(false);setLoading(false);setStatus('The run did not complete.');if(data.fatal)release();}
  };
  w.onerror=()=>{release();setError('The GPU worker could not start. Reload the page or try a recent Chrome browser.');};return w;
 }
 function guard(){clearTimer();timeout.current=setTimeout(()=>{release();setError('The run timed out. The GPU worker has been stopped; you can load the model again.');},180000);}
 function load(){setError('');setBusy(true);setLoading(true);setProgress(0);setStatus('Checking GPU compatibility…');const w=createWorker();guard();w.postMessage({type:'load'});}
 function run(){if(!loaded||busy)return;setError('');setBusy(true);guard();
  if(mode==='generate'){setOutput('');setTps(null);setTtft(null);setTokens(0);setStatus('Preparing your prompt…');worker.current?.postMessage({type:'generate',prompt,limit});}
  else {setBench(null);setRunCount(0);setStatus('Preparing the benchmark…');worker.current?.postMessage({type:'benchmark',depth});}
 }
 function stop(){if(loading){release();setStatus('Loading stopped. Verified chunks stay cached.');}else{worker.current?.postMessage({type:'cancel'});setStatus('Stopping after the current token group…');}}
 function download(){if(!bench)return;const url=URL.createObjectURL(new Blob([JSON.stringify({...bench,model:'LFM2.5-230M',kernel:'MIX_M40',browser:navigator.userAgent,scope:'LOCAL_QUICK_BENCHMARK'},null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='monarch-local-benchmark.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
 const value=mode==='benchmark'?bench?.tps:tps;
 return <div className="monarch-demo" aria-busy={busy}>
  <div className="demo-top"><div><span className={`status-dot ${loaded?'is-ready':''}`} aria-hidden="true"/><span className="mono">LIVE / LFM2.5-230M</span></div><span className="demo-local">On-device inference</span></div>
  <div className="demo-layout">
   <div className="demo-workspace">
    <div className="demo-mode" aria-label="Demo mode"><button aria-pressed={mode==='generate'} disabled={busy} onClick={()=>setMode('generate')}>Generate text</button><button aria-pressed={mode==='benchmark'} disabled={busy} onClick={()=>setMode('benchmark')}>Benchmark</button></div>
    {mode==='generate'?<>
     <label className="control-label" htmlFor="monarch-prompt">Give this small model a task</label>
     <textarea id="monarch-prompt" value={prompt} onChange={e=>setPrompt(e.target.value)} maxLength={6000} disabled={busy} rows={3} spellCheck={false}/>
     <div className="prompt-suggestions" role="group" aria-label="Prompt presets">{monarchPrompts.map(p=><button key={p.label} disabled={busy} aria-pressed={prompt===p.text} onClick={()=>setPrompt(p.text)}>{p.label} <span aria-hidden="true">↗</span></button>)}</div>
     <div className="demo-output" aria-label="Generated response"><span className="output-label">{output?'LFM2.5 / RESPONSE':'OUTPUT'}</span>{output?<MonarchMarkdown text={output}/>:<p className="output-placeholder">Your GPU will write the answer here.<br/><span>No canned responses. No inference server.</span></p>}{busy&&!loading&&output&&<span className="generation-cursor" aria-hidden="true">▍</span>}</div>
    </>:<>
     <label className="control-label" htmlFor="monarch-depth">Context window to measure</label>
     <div className="benchmark-controls"><select id="monarch-depth" value={depth} onChange={e=>setDepth(Number(e.target.value))} disabled={busy}><option value={192}>192 tokens</option><option value={1024}>1,024 tokens</option></select><span>4 runs × 256 generated tokens</span></div>
     <p className="benchmark-explainer">A fixed prompt, a warm GPU, and four free-running continuations. Model loading and prompt prefill stay outside the decode timer. Keep this tab visible and pause other GPU workloads.</p>
     <div className="run-chart" aria-label="Benchmark runs">{[0,1,2,3].map(i=><div className="run-column" key={i}><div className="run-track"><div className="run-bar" style={{height:bench?`${Math.max(5,bench.runTps[i]/Math.max(...bench.runTps)*100)}%`:0}}/></div><strong>{bench?Math.round(bench.runTps[i]).toLocaleString():busy&&i<runCount?'✓':'—'}</strong><span>Run 0{i+1}</span></div>)}</div>
     {bench&&<div className="benchmark-summary"><span>Spread {bench.spreadPct.toFixed(1)}% · exact repeats 4/4</span><button onClick={download}>Download results ↓</button></div>}
     {bench&&bench.spreadPct>10&&<p className="measurement-note">Timing spread exceeded 10%. Treat this as a noisy local observation; pause other apps and repeat for a steadier comparison.</p>}
    </>}
    {error&&<div role="alert" className="demo-error">{error}</div>}
    {loading&&<div className="load-progress" role="progressbar" aria-label="Model download" aria-valuenow={Math.round(progress*100)} aria-valuemin={0} aria-valuemax={100}><span style={{width:`${progress*100}%`}}/></div>}
    <div className="demo-actions">{!loaded&&!busy?<button className="solid-button" onClick={load}>Load model <span>≈174 MB ↓</span></button>:busy?<button className="solid-button stop-button" onClick={stop}>Stop {loading?'loading':'run'} <span>■</span></button>:<button className="solid-button" disabled={mode==='generate'&&!prompt.trim()} onClick={run}>{mode==='generate'?'Generate response':'Run benchmark'} <span>↗</span></button>}
     {mode==='generate'&&<label className="token-limit">Max tokens <select aria-label="Maximum generated tokens" value={limit} disabled={busy} onChange={e=>setLimit(Number(e.target.value))}><option value={128}>128</option><option value={256}>256</option><option value={512}>512</option></select></label>}
     {loaded&&!busy&&<button className="unload-button" onClick={()=>{release();setStatus('Model unloaded. GPU memory released.');}}>Unload</button>}
    </div>
    <p className="demo-status" role="status">{status}</p>
   </div>
   <aside className="demo-meter"><span className="meter-label">YOUR DEVICE / {mode==='benchmark'?'LAST BENCHMARK':'LIVE DECODE'}</span><div className="meter-number" aria-label={value?`${Math.round(value)} tokens per second`:'No speed measured yet'}>{value?Math.round(value).toLocaleString():'—'}</div><span className="meter-unit">tokens / second</span><p className="meter-device">{device}</p>
    <dl className="meter-details"><div><dt>{mode==='benchmark'?'ms per token':'Time to first token'}</dt><dd>{mode==='benchmark'?(bench?`${bench.ms.toFixed(3)} ms`:'—'):(ttft!==null?`${Math.round(ttft)} ms`:'—')}</dd></div><div><dt>{mode==='benchmark'?'Tokens per run':'Generated tokens'}</dt><dd>{mode==='benchmark'?(bench?'256':'—'):(tokens||'—')}</dd></div><div><dt>Kernel</dt><dd>MIX_M40</dd></div><div><dt>Dispatches / token</dt><dd>72</dd></div></dl>
    <div className="meter-footnote"><span>What this number means</span><p>{mode==='generate'?'Visible decode throughput after the first token, including streaming overhead. Short answers may finish before speed settles.':'The reciprocal of median wall time per token over four runs. A quick local measurement, not a reproduction of the full lab protocol.'}</p></div>
   </aside>
  </div>
  <div className="demo-bottom"><span>Prompts and generated text stay in this browser. Downloads come from this site.</span><a href="#compatibility">Device requirements ↗</a></div>
 </div>;
}
