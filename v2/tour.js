"use strict";
(()=>{
const $=id=>document.getElementById(id);
const video=$("v2C1Video");if(!video)return;
const panels={analyze:$("rv2AnalyzePanel"),explore:$("rv2ExplorePanel"),ask:$("rv2AskPanel")};
const screens={analyze:{heading:"Video analysis",description:"The archived V2 pipeline processed the C1 source video. These values are preserved from the D2 application report; no inference is performed in the browser."},explore:{heading:"Observation and identity review",description:"V2 provides explicit review states for object identity, frame observations, and movement candidates. The labels below illustrate the supported review categories without modifying saved records."},ask:{heading:"Evidence-gated historical retrieval",description:"The archived D2 query outcomes are shown below. Without explicitly reviewed frame or event evidence, V2 returns an insufficient-evidence status."}};
let currentStep="analyze",currentQuery="where";
const outputs={
where:{status:"INSUFFICIENT_TRUSTED_EVIDENCE",title:"No reviewed location evidence",detail:"The C1 automatic detections were not individually approved in this V2 run. WHERE requires an approved exact-frame observation."},
when:{status:"EVENT_REQUIRES_REVIEW",title:"Event requires evidence review",detail:"A movement candidate exists, but it was not approved. V2 does not treat this automatic candidate as a verified movement."},
before:{status:"EVENT_REQUIRES_REVIEW",title:"Insufficient prior-state evidence",detail:"WHERE-BEFORE needs a reviewed event and trusted before-state evidence. Neither was approved in the recorded D2 C1 run."}
};
function render(){
for(const [name,panel] of Object.entries(panels))panel.hidden=name!==currentStep;
for(const b of document.querySelectorAll("[data-rv2-step]")){const on=b.dataset.rv2Step===currentStep;b.classList.toggle("active",on);b.setAttribute("aria-pressed",String(on))}
$("rv2StepTitle").textContent=screens[currentStep].heading;
$("rv2StepDescription").textContent=screens[currentStep].description;
const ans=outputs[currentQuery];
$("rv2AskStatus").textContent=ans.status;$("rv2AskTitle").textContent=ans.title;$("rv2AskDetail").textContent=ans.detail;
for(const b of document.querySelectorAll("[data-rv2-query]")){const on=b.dataset.rv2Query===currentQuery;b.classList.toggle("selected",on);b.setAttribute("aria-pressed",String(on))}
}
for(const b of document.querySelectorAll("[data-rv2-step]"))b.addEventListener("click",()=>{currentStep=b.dataset.rv2Step;render()});
for(const b of document.querySelectorAll("[data-rv2-query]"))b.addEventListener("click",()=>{currentQuery=b.dataset.rv2Query;render()});
const reviewText={
"UNCERTAIN":["UNCERTAIN · Archived observation state","V2 D1 initializes unreviewed automatic observations as uncertain. The archived D2 C1 run did not approve these detections."],
"SUPPORTED":["SUPPORTED · Review-state definition","The selected status illustrates the review contract; this interaction does not change recorded results. The local application requires explicit confirmation of each corresponding source frame."],
"MISSING":["MISSING · Review-state definition","The selected status illustrates the review contract; this interaction does not change recorded results. Marking an observation missing does not interpolate adjacent frames or delete original detections."]
};
for(const b of document.querySelectorAll("[data-rv2-review]"))b.addEventListener("click",()=>{
 const label=b.dataset.rv2Review;for(const p of document.querySelectorAll("[data-rv2-review]"))p.classList.toggle("selected",p===b);
 $("rv2ReviewHeading").textContent=reviewText[label][0];$("rv2ReviewDetail").textContent=reviewText[label][1];
});
for(const b of document.querySelectorAll("[data-rv2-time]"))b.addEventListener("click",()=>{
 const secs=Number(b.dataset.rv2Time);if(Number.isFinite(secs)){if(video.readyState>=1){video.currentTime=Math.min(secs,Math.max(0,video.duration-0.1));}else video.addEventListener("loadedmetadata",()=>{video.currentTime=Math.min(secs,Math.max(0,video.duration-0.1));},{once:true});video.scrollIntoView({behavior:"smooth",block:"center"});}
});
render();
})();