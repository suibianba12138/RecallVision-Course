"use strict";
(()=>{
const $=id=>document.getElementById(id);
const video=$("v2C1Video");if(!video)return;
const panels={analyze:$("rv2AnalyzePanel"),explore:$("rv2ExplorePanel"),ask:$("rv2AskPanel")};
const screens={analyze:{heading:"Analyze the source",description:"A real C1 video was processed in the V2 local application. Inspect the recorded output; this website does not run YOLO or Streamlit."},explore:{heading:"Explore uncertain evidence",description:"V2 exposes identity, individual observations, and automatic movement candidates for review. Try switching the illustrative review labels below."},ask:{heading:"Ask the reviewed memory",description:"Replay the actual D2 query states. Without a human-approved frame or movement event, V2 correctly refuses to fabricate a trusted answer."}};
let currentStep="analyze",currentQuery="where";
const outputs={
where:{status:"INSUFFICIENT_TRUSTED_EVIDENCE",title:"No trusted location yet",detail:"The C1 automatic detections were not individually approved in this V2 run. WHERE requires an approved exact-frame observation."},
when:{status:"EVENT_REQUIRES_REVIEW",title:"Movement candidate awaits review",detail:"A movement candidate exists, but it was not approved. V2 does not treat this automatic candidate as a verified movement."},
before:{status:"EVENT_REQUIRES_REVIEW",title:"Earlier position not certified",detail:"WHERE-BEFORE needs a reviewed event and trusted before-state evidence. Neither was approved in the recorded D2 C1 run."}
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
"UNCERTAIN":["UNCERTAIN · Actual starting state","V2 D1 initializes unreviewed automatic observations as uncertain. The archived D2 C1 run did not approve these detections."],
"SUPPORTED":["SUPPORTED · What a manual review would do","This is a demonstration of the review meaning, not a saved operation. In the local V2 app a reviewer must inspect the exact source frame and explicitly approve it."],
"MISSING":["MISSING · What a manual review would do","This is a demonstration of the review meaning, not a saved operation. Marking one source observation missing does not delete raw detections or fill neighboring frames."]
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