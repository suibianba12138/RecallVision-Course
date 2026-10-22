/**
 * RecallVision 3-case demo viewer.
 * Facts: published C1 frozen memory/query artifacts and recorded C2/C2-R results.
 * This is NOT live YOLO inference or a Python query service.
 */
(() => {
'use strict';
const $ = id => document.getElementById(id);
const CASES = {
 c1: {
  name:'C1 · development bottle', target:'obj:bottle:0001',
  source:'https://pixabay.com/videos/bottle-hands-water-liquid-table-82261/',
  hash:'24067f8ebc34d482728aa53df541afef3c28287a3841f7d5db255da202256730',
  duration:7.9079, fps:29.97002997002997, metadata:'7.9079 s • 29.97 fps • Pixabay 82261',
  badge:'FOUR QUERY ROUTES / RECORDED PASS', flag:'SUPPORTED', flagClass:'good-label',
  desc:'P2A-R1 grouped seven observed bottle track IDs into one persistent identity. Two state segments and one evidence-bounded movement were stored.',
  records:[['Source track IDs','7'],['Canonical observations','48'],['Target hypotheses','1 persistent bottle']],
  note:'P2D emitted a MOVE_OBJECT between observed states at 3.0697–3.3033 s. Neither this window nor the event denotes exact physical movement onset.',
  videoText:'The recorded C1 source is a 7.91 s real bottle clip. Load the exact original MP4 or use an annotated analysis video if published.',
  summary:'C1 development video · 7.9079 seconds · published frozen memory fixture',
  videos:'c1-analysis.mp4'
 },
 c2: {
  name:'C2 · independent cake', target:'cake / unresolved persistent identity',
  source:'https://mixkit.co/free-stock-video/placing-a-cake-in-the-center-of-a-party-table-5117/',
  hash:'c9524bc8dd28b775572b74e9f6d44865c9158d0ce8cdee315462b9b2e2397255',
  duration:10.041667, metadata:'10.0417 s • Mixkit 5117 • independent',
  badge:'P2B / AMBIGUOUS_TARGET', flag:'BLOCKED', flagClass:'warn-label',
  desc:'The new cake video produced 579 cake observations but 12 persistent cake candidates. P2B could not select an unambiguous target entity.',
  records:[['Cake observations','579'],['Persistent candidates','12'],['Target resolution','AMBIGUOUS_TARGET']],
  note:'No unique cake identity was established; do not interpret candidate numbers as 12 separate physical cakes, and do not claim successful downstream QA.',
  videoText:'Independent Mixkit cake clip · recorded as C2. The diagnostic result is ambiguous identity resolution, not successful QA.',
  summary:'C2 independent cake video · 10.041667 seconds · candidate ambiguity',
  videos:'c2-analysis.mp4'
 },
 c2r: {
  name:'C2-R · independent bottle', target:'bottle / 2 unresolved candidates',
  source:'https://www.pexels.com/video/person-placing-a-water-bottle-on-a-table-with-gemstones-6768198/',
  hash:'1b3a3efd32bc2c57bc775f8b9d68b66c25d3c8de264650601aeec22a5cbbf2c8',
  duration:29.268333, metadata:'29.2683 s • Pexels 6768198 • independent',
  badge:'P2B / AMBIGUOUS_TARGET', flag:'BLOCKED', flagClass:'warn-label',
  desc:'The independent Pexels bottle case has 625 bottle observations assigned to two competing persistent candidates (621 and 4). The unresolved second identity blocks unique target selection.',
  records:[['Bottle observations','625'],['Candidate 1','621 observations'],['Candidate 2','4 observations']],
  note:'P2C/P2D and downstream positive QA were not completed for this case. A 621/4 split demonstrates fragmented identity hypotheses, not a validated movement event.',
  videoText:'Independent 29.27 s Pexels bottle clip · recorded as C2-R. This case demonstrates an identity fragmentation failure.',
  summary:'C2-R independent bottle video · 29.268333 seconds · 621 + 4 split',
  videos:'c2r-analysis.mp4'
 }
};
const QUERY = {
 where:{title:'Center region',text:'The bottle was last observed in the center region at 5.0717 seconds (frame 152). This is its last supported observation, not a known location at the end of the video.',time:5.071733333333333,ref:'Frame 152 · 5.0717 s'},
 when:{title:'3.0697 – 3.3033 seconds',text:'A supported movement event links the left and center state segments. Frames 92–99 bound the evidence interval; the exact physical onset is not established.',time:3.069733333333333,ref:'Frames 92–99 · MOVE_OBJECT'},
 before:{title:'Left region',text:'The recorded state before the supported movement ends in the left region at frame 92, 3.0697 seconds.',time:3.069733333333333,ref:'Frame 92 · 3.0697 s'},
 who:{title:'obj:person:0001',text:'The identity is returned from an existing pre-authorized F3 attributed event, not inferred by a VLM running on this webpage or at query time.',time:3.069733333333333,ref:'Frames 92–99 · authorized F3 artifact'}
};
const FROZEN_QUERIES={
 where:'where_object.json',when:'when_object_moved.json',
 before:'where_object_before.json',who:'who_moved_object.json'
};
let caseId='c1',question='where',localBlobURL=null,uploadedCase=null,annotated=false;
const video=$('clip');
const fmt=t=>Number(t).toFixed(3)+' s';
const htmlEsc=s=>String(s).replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function setMessage(txt){$('videoNote').textContent=txt}
function clearVideo(){
 video.pause();video.removeAttribute('src');video.load();
 $('stage').classList.remove('has-video');
 if(localBlobURL){URL.revokeObjectURL(localBlobURL);localBlobURL=null}
 uploadedCase=null;annotated=false; $('timeLabel').textContent='00.000 s';
}
function bindVideo(src,kind){
 video.pause();video.src=src;video.load();
 video.onloadedmetadata=()=>{
  const duration=video.duration, expected=CASES[caseId].duration;
  if(!Number.isFinite(duration)||Math.abs(duration-expected)>.8){
    setMessage('Loaded media duration differs from frozen input. Treat this as explanatory video only; not frame-synchronized evidence.');
  }else{
    setMessage(kind==='ANALYTICAL EXPORT'
      ?'Published analysis export. Annotations are illustrative of saved detector/memory data; the query values are from frozen records.'
      :'Exact original MP4 SHA-256 verified locally. It is not uploaded to the website.');
  }
  $('stage').classList.add('has-video');
  $('videoLabel').textContent=kind+' · '+caseId.toUpperCase();
  video.currentTime=caseId==='c1'?Math.min(3.0697333,duration):0;
 };
 video.onerror=()=>{$('stage').classList.remove('has-video');setMessage('Video playback was unavailable. Original provider link and the recorded data remain accessible.')};
}
async function tryAnalyticalVideo(id){
 const path='./media/'+CASES[id].videos;
 try{
  const response=await fetch(path,{method:'HEAD',cache:'no-store'});
  if(response.ok && caseId===id) {annotated=true;bindVideo(path,'ANALYTICAL EXPORT');}
 }catch{} // Video is optional; never invent unavailable analysis media.
}
function timelineC1(){
 const dur=CASES.c1.duration;
 const segments=[
  ['Observed LEFT',2.3023,3.069733333333333,'left'],
  ['Evidence MOVE',3.069733333333333,3.3033,'event'],
  ['Observed CENTER',3.3033,5.071733333333333,'center']
 ];
 const bars=segments.map(([label,a,b,cls])=>{
  const left=(a/dur*100).toFixed(2), width=((b-a)/dur*100).toFixed(2);
  return '<div class="timeline-row"><span class="t-label">'+htmlEsc(label)+'</span><div class="timeline-track"><span class="segment '+cls+'" style="left:'+left+'%;width:'+width+'%" title="'+fmt(a)+' – '+fmt(b)+'"></span></div></div>';
 }).join('');
 return bars+'<div class="timeline-caption"><span>0.00 s</span><span>7.91 s</span></div><div class="timeline-legend"><span><i></i>Before</span><span><i class="event"></i>Evidence interval</span><span><i class="center"></i>After</span></div>';
}
function renderChart(id){
 if(id==='c1'){
  $('timeline').innerHTML=timelineC1();
  $('identityChart').innerHTML='<div class="identity-title">7 observed track IDs → 1 persistent bottle ID</div><div class="id-badges">'+
  ['16','17','23','1','58','71','76'].map(id=>'<span>Track '+id+'</span>').join('')+'</div>';
 }else if(id==='c2'){
  $('timeline').innerHTML='<p style="font-size:12px;color:#abc4d5;margin:0">No unique target timeline established at P2B. Unsupported timeline segments are intentionally not drawn.</p>';
  $('identityChart').innerHTML='<div class="identity-title">12 recorded cake candidate identities</div><div class="id-badges">'+
  Array.from({length:12},(_,i)=>'<span>Candidate '+String(i+1).padStart(2,'0')+'</span>').join('')+'</div>';
 }else{
  $('timeline').innerHTML='<p style="font-size:12px;color:#abc4d5;margin:0">Unique target unresolved: movement-event timeline not established.</p>';
  $('identityChart').innerHTML='<div class="identity-title">Recorded candidate observation split · 625 total</div>'+
  [['Candidate A',621],['Candidate B',4]].map(([label,n])=>
  '<div class="candidate-row"><span>'+label+'</span><div class="bar"><i style="width:'+100*n/625+'%"></i></div><span>'+n+'</span></div>').join('');
 }
}
function renderQuery(q){
 if(caseId!=='c1')return;
 question=q;
 $('qaButtons').querySelectorAll('button').forEach(b=>b.classList.toggle('active',b.dataset.q===q));
 const a=QUERY[q];
 $('answerLabel').textContent='SUPPORTED ANSWER / RECORDED C1 FIXTURE';
 $('answerTitle').textContent=a.title;
 $('answerText').textContent=a.text;
 $('evidence').innerHTML=htmlEsc(a.ref)+' <button type="button" id="jumpBtn">Jump to evidence ↗</button>';
 $('jumpBtn').addEventListener('click',jumpToAnswer);
 $('interpretNote').textContent='This selection reads the recorded C1 query result, not Python or model inference. WHO uses a pre-authorized person-attribution artifact.';
 jumpToAnswer();
}
function jumpToAnswer(){
 if(caseId!=='c1')return;
 const t=QUERY[question].time;
 $('timeLabel').textContent=fmt(t)+' · frame '+Math.round(t*CASES.c1.fps);
 if($('stage').classList.contains('has-video')&&video.readyState>=1){
  try{video.currentTime=Math.min(t,Math.max(0,video.duration-.01))}catch{}
 }
}
function renderCase(id){
 caseId=id;question='where';clearVideo();
 const c=CASES[id];
 $('caseTag').textContent=c.badge;
 $('caseMeta').textContent=c.metadata;
 $('stageText').textContent=c.videoText;
 $('targetName').textContent=c.target;
 $('memoryDescription').textContent=c.desc;
 $('memoryFlag').textContent=c.flag;
 $('memoryFlag').className=c.flagClass;
 $('records').innerHTML=c.records.map(([a,b])=>
  '<div class="record"><span>'+htmlEsc(a)+'</span><b>'+htmlEsc(b)+'</b></div>').join('');
 $('memoryNote').textContent=c.note;
 $('videoSource').href=c.source;
 $('dataSourceText').textContent=c.summary;
 $('videoLabel').textContent=id.toUpperCase()+' / NO VIDEO ATTACHED';
 $('timeLabel').textContent='00.000 s';
 $('stageEmpty').style.display='';
 $('localVideo').value='';
 setMessage('No media is bundled yet; choose the exact original MP4 locally, or use the provider link. Analysis exports will appear automatically if published.');
 renderChart(id);
 const isC1=id==='c1';
 $('qaTitle').textContent=isC1?'Ask the recorded memory':'Why can’t this case answer?';
 $('qaState').textContent=isC1?'C1 ARTIFACT REPLAY':'P2B BLOCKED';
 $('qaButtons').querySelectorAll('button').forEach(b=>{b.disabled=!isC1;b.classList.remove('active')});
 $('interpretNote').textContent=isC1?
 'Query results are replayed from frozen C1 memory, not a fresh-video run.': 'These independent-video diagnostics are real blocked outcomes. No supported WHO/WHEN/BEFORE/WHERE response is claimed.';
 if(isC1)renderQuery('where');
 else {
  $('answerLabel').textContent='STATUS / NO SUPPORTED ANSWER';
  $('answerTitle').textContent='AMBIGUOUS_TARGET';
  $('answerText').textContent=id==='c2'
   ?'Twelve competing cake entity candidates were produced. The target was not uniquely resolved, so downstream QA success cannot be claimed.'
   :'Two bottle entities (621 and 4 observations) remain. The system fails closed rather than selecting an unsupported target or inventing movement evidence.';
  $('evidence').textContent='Evidence: recorded P2B diagnosis · no positive downstream query';
 }
 document.querySelectorAll('[data-case]').forEach(tab=>{
  tab.classList.toggle('active',tab.dataset.case===id);
  tab.setAttribute('aria-selected',String(tab.dataset.case===id));
 });
 tryAnalyticalVideo(id);
}
document.querySelectorAll('[data-case]').forEach(tab=>tab.addEventListener('click',()=>renderCase(tab.dataset.case)));
$('qaButtons').querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>renderQuery(b.dataset.q)));
video.addEventListener('timeupdate',()=>{$('timeLabel').textContent=fmt(video.currentTime)});
$('localVideo').addEventListener('change',async ev=>{
 const file=ev.target.files?.[0];if(!file)return;
 const id=caseId, expected=CASES[id].hash;
 if(!globalThis.crypto?.subtle){setMessage('This browser cannot verify SHA-256. Open the page via HTTPS or localhost.');return}
 try{
  setMessage('Checking exact experimental source SHA-256…');
  const digest=await crypto.subtle.digest('SHA-256',await file.arrayBuffer());
  if(caseId!==id)return;
  const sha=[...new Uint8Array(digest)].map(b=>b.toString(16).padStart(2,'0')).join('');
  if(sha!==expected){setMessage('SHA-256 does not match frozen '+id.toUpperCase()+' source. Refusing to align this video to experimental records.');return}
  if(localBlobURL)URL.revokeObjectURL(localBlobURL);
  localBlobURL=URL.createObjectURL(file);uploadedCase=id;
  bindVideo(localBlobURL,'HASH-VERIFIED ORIGINAL');
 }catch(err){setMessage('Could not load local MP4: '+String(err))}
});
renderCase('c1');
})();
