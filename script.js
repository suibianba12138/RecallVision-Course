/**
 * RecallVision course project site.
 * Every C1 answer here is a transcription of the frozen public C1 query fixture.
 * This is a client-side REPLAY EXPLORER, NOT a live query or inference engine.
 * Source: https://github.com/suibianba12138/RecallVision-Course/tree/main/course_demo/fixture
 */
const C1 = Object.freeze({
  videoId: 'p1b_pixabay_bottle',
  duration: 7.9079,
  fps: 29.97002997002997,
  entityId: 'obj:bottle:0001',
  before: { start: 2.3023, end: 3.069733333333333, frameStart:69, frameEnd:92, zone:'left_region', observations:23 },
  event: { start: 3.069733333333333, end:3.3033, frameStart:92, frameEnd:99, type:'MOVE_OBJECT', personsAtP2D:[] },
  after: { start:3.3033, end:5.071733333333333, frameStart:99, frameEnd:152, zone:'center_region', observations:25 },
  evidence: {
    92: { time:3.069733333333333, zone:'left_region' },
    99: { time:3.3033, zone:'center_region' },
    152:{ time:5.071733333333333, zone:'center_region' }
  }
});

const queries = {
  where: {
    headline:'Center region',
    description:'The bottle was last observed in the center region at 5.07 s. This is its last supported observation, not a guarantee of its position at video end.',
    support:'Frame 152 · 5.0717s',
    entity:'obj:bottle:0001',
    frame:152,
    zone:'center_region'
  },
  when: {
    headline:'3.07 – 3.30 seconds',
    description:'A supported MOVE_OBJECT event places the movement between frame 92 and frame 99. This interval bounds observed evidence, not the exact physical movement time.',
    support:'Frames 92–99 · event interval',
    entity:'move-object:…:1→2',
    frame:92,
    zone:'left_region'
  },
  before: {
    headline:'Left region',
    description:'Before the supported movement event, the bottle was last observed in the left region, at frame 92.',
    support:'Frame 92 · 3.0697s',
    entity:'obj:bottle:0001',
    frame:92,
    zone:'left_region'
  },
  who: {
    headline:'obj:person:0001',
    description:'Person attribution was read from a pre-authorized F3 attributed event. It was not inferred from proximity, and no VLM is run when answering.',
    support:'Frames 92–99 · F3 authorized',
    entity:'obj:person:0001',
    frame:92,
    zone:'left_region'
  }
};

const $ = (id) => document.getElementById(id);
const qs = document.querySelectorAll('.query-option');
const frameBtns = document.querySelectorAll('.frame-card');
const video = $('sourceVideo');
const viewport = $('mediaViewport');
const sceneBottle = $('sceneBottle');
const scrubber = $('scrubber');
let currentTime = 5.071733333333333;
let currentQuery = 'where';
let hasVideo = false;
let playing = false;
let ticker = null;
let videoObjectUrl = null;
const pct = (n) => Math.max(0,Math.min(100,(n/C1.duration)*100));
const secs = (t) => Number(t).toFixed(3);

function renderTime(t, userScrub=false){
  currentTime = Math.max(0, Math.min(C1.duration,Number(t)||0));
  scrubber.value = String(currentTime);
  $('playerTime').textContent = secs(currentTime)+'s';
  $('frameTime').innerHTML = `${secs(currentTime)} <small>/ 07.908s</small>`;
  $('hudFrame').textContent = `FRAME ${String(Math.round(currentTime*C1.fps)).padStart(4,'0')}`;
  $('timelineCursor').style.left = `${pct(currentTime)}%`;
  // The stylized scene is schematic only; it is not a reconstructed observation.
  const zone = currentTime < C1.event.start ? 'left' : currentTime >= C1.event.end ? 'center' : 'transition';
  sceneBottle.style.left = zone === 'left' ? '24%' : zone === 'center' ? '50%' : `${24+26*(currentTime-C1.event.start)/(C1.event.end-C1.event.start)}%`;
  sceneBottle.style.opacity = (currentTime < C1.before.start || currentTime > C1.after.end) ? '.38' : '1';
  let closest=null,dist=Infinity;
  for(const [frame,item] of Object.entries(C1.evidence)){
    const d=Math.abs(item.time-currentTime); if(d<dist){closest=frame;dist=d;}
  }
  for (const el of frameBtns) el.classList.toggle('active',el.dataset.frame===closest && dist<.12);
  if(hasVideo && userScrub) { try{ video.currentTime = currentTime; }catch{} }
}
function pause(){playing=false; $('playToggle').textContent='▶';$('playToggle').setAttribute('aria-label','Play'); clearInterval(ticker);ticker=null;if(hasVideo)video.pause();}
function start(){
  if(playing) return;
  if(currentTime>=C1.duration-.05){seek(0);}
  playing=true;$('playToggle').textContent='Ⅱ';$('playToggle').setAttribute('aria-label','Pause');
  if(hasVideo){video.play().catch(()=>pause());return;}
  // Time scrubbing of a visibly labeled schematic—not video playback.
  let last=performance.now();ticker=setInterval(()=>{const now=performance.now();let dt=(now-last)/1000;last=now;let next=currentTime+dt;if(next>=C1.duration){pause();renderTime(C1.duration);return;}renderTime(next);},40);
}
function seek(time){if(playing)pause();renderTime(time,true);}
function selectQuery(key){
  const item=queries[key]; if(!item)return;
  currentQuery=key;
  qs.forEach(el=>{let active=el.dataset.query===key;el.classList.toggle('active',active);el.setAttribute('aria-pressed',String(active));});
  $('answerHeadline').textContent=item.headline;
  $('answerText').textContent=item.description;
  $('answerSupport').textContent=item.support;
  $('answerEntity').textContent=item.entity;
  $('leftBlock').style.filter=(key==='before'||key==='when'||key==='who')?'brightness(1.24)':'none';
  $('centerBlock').style.filter=(key==='where')?'brightness(1.25)':'none';
  $('eventBlock').style.boxShadow=(key==='when'||key==='who')?'0 0 0 2px rgba(255,204,133,.55)':'none';
  seek(C1.evidence[item.frame].time);
}
qs.forEach(btn=>btn.addEventListener('click',()=>selectQuery(btn.dataset.query)));
frameBtns.forEach(btn=>btn.addEventListener('click',()=>seek(C1.evidence[btn.dataset.frame].time)));
$('jumpEvidence').addEventListener('click',()=>seek(C1.evidence[queries[currentQuery].frame].time));
$('playToggle').addEventListener('click',()=>playing?pause():start());
scrubber.addEventListener('input',e=>seek(e.target.value));
video.addEventListener('timeupdate',()=>{if(hasVideo && !video.seeking)renderTime(video.currentTime)});
video.addEventListener('ended',()=>pause());
video.addEventListener('pause',()=>{if(hasVideo&&playing)pause()});

function enableVideo(src, label){
  pause();
  if(videoObjectUrl && videoObjectUrl!==src){URL.revokeObjectURL(videoObjectUrl);videoObjectUrl=null;}
  video.src=src;
  video.load();
  video.onloadedmetadata=()=>{
    const length=Number(video.duration);
    // The 7.9079s/237frame C1 fixture is bound to THIS original source.
    if(!Number.isFinite(length)||Math.abs(length-C1.duration)>0.7){
      $('clipNotice').textContent='Loaded a clip with a different duration. C1 answers may not match this video.';
    } else {$('clipNotice').textContent=`${label} · ${length.toFixed(2)}s · C1 source-aligned only if the original hash matches.`;}
    hasVideo=true;viewport.classList.add('has-video');$('mediaMode').textContent='LOCAL VIDEO LOADED';
    renderTime(currentTime,true);
  };
  video.onerror=()=>{
    // Gracefully restore the unmistakably schematic fallback.
    hasVideo=false;viewport.classList.remove('has-video');$('mediaMode').textContent='SCHEMATIC VIEW';
  };
}
$('videoFile').addEventListener('change',async (e)=>{
  const file=e.target.files&&e.target.files[0];if(!file)return;
  if(!file.type.startsWith('video/')&&!/\.(mp4|webm|mov)$/i.test(file.name)){
    $('clipNotice').textContent='Please choose a video file.';return;
  }
  // Cryptographically bind a chosen local MP4 to the recorded C1 fixture.
  // Without this check, a different clip could falsely appear to validate the C1 answers.
  const expected='24067f8ebc34d482728aa53df541afef3c28287a3841f7d5db255da202256730';
  if(!globalThis.crypto?.subtle){
    $('clipNotice').textContent='Hash verification unavailable in this browser context. Use a local HTTP server (localhost), then retry.';
    return;
  }
  const bytes=await file.arrayBuffer();
  const digest=await crypto.subtle.digest('SHA-256',bytes);
  const actual=Array.from(new Uint8Array(digest),x=>x.toString(16).padStart(2,'0')).join('');
  if(actual!==expected){
    $('clipNotice').textContent='Selected video does NOT match the C1 source SHA-256; showing schematic instead.';
    return;
  }
  const prev=videoObjectUrl;
  videoObjectUrl=URL.createObjectURL(file);
  if(prev) URL.revokeObjectURL(prev);
  enableVideo(videoObjectUrl,file.name);
});

// Automatically use the clip if the project owner placed it here.
// Never fetch or package third-party source media by default.
(function optionalBundledMedia(){
  const probe = document.createElement('video');
  probe.preload='metadata';
  probe.onloadedmetadata=async()=>{
    // The bundled-asset route is for local development. Verify original bytes before identifying it as source footage.
    try{
      const res=await fetch('assets/p1b_pixabay_bottle.mp4');
      if(!res.ok || !globalThis.crypto?.subtle) return;
      const buffer=await res.arrayBuffer();
      const digest=await crypto.subtle.digest('SHA-256',buffer);
      const hash=Array.from(new Uint8Array(digest),x=>x.toString(16).padStart(2,'0')).join('');
      if(hash==='24067f8ebc34d482728aa53df541afef3c28287a3841f7d5db255da202256730')
        enableVideo('assets/p1b_pixabay_bottle.mp4','Verified original C1 clip');
      else $('clipNotice').textContent='Bundled clip has a different SHA-256; showing schematic instead.';
    }catch{}
  };
  probe.onerror=()=>{};
  probe.src='assets/p1b_pixabay_bottle.mp4';
})();

// Small-screen navigation
$('menuToggle').addEventListener('click',()=>{
 const nav=document.querySelector('.primary-nav');const isOpen=nav.classList.toggle('open');$('menuToggle').setAttribute('aria-expanded',String(isOpen));
});
document.querySelectorAll('.primary-nav a').forEach(a=>a.addEventListener('click',()=>{document.querySelector('.primary-nav').classList.remove('open');$('menuToggle').setAttribute('aria-expanded','false')}));
selectQuery('where');