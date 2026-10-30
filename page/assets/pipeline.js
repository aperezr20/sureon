// Pipeline controls run independently of gallery data and module imports.
(()=>{
'use strict';
const $=s=>document.querySelector(s);
const esc=value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const stageImages={all:'page/assets/pipeline.png',0:'page/assets/synthesize.png',1:'page/assets/refine.png',2:'page/assets/blindcuration.png',3:'page/assets/verification.png'};
const stageNotes={
 all:{label:'Full pipeline',title:'From narration to visual reasoning',paragraphs:[
  'Four stages convert narrated surgical videos into **visually grounded question–answer pairs**: synthesis, distractor refinement, blind-test curation, and expert verification.',
  'The curated training split forms SUREON-DB (**76,061 automatically pseudo-labeled QA pairs**). Surgeon review of the validation split produces the SUREON benchmark (**340 expert-verified questions**).'
 ]},
 0:{label:'Stage 1 · Synthesize',title:'Ground questions in narrated moments',paragraphs:[
  '**Ten category-specialized generator–validator agent pairs** identify **Semantic Grounding Moments**: intervals where the narration explicitly connects a surgical concept to the visible scene. Each generator produces questions, answers, and clip boundaries for its assigned category.',
  'Paired validators check **visual grounding, temporal alignment, hallucinations, and duplicate meaning**, alongside category-specific requirements.'
 ]},
 1:{label:'Stage 2 · Refine',title:'Reduce linguistic biases',paragraphs:[
  'Correct answers are grounded in transcripts—they tend to be **longer and more specific** than typical distractors, creating a detectable distributional signal a model can exploit without watching the video. We address this with **Retrieval-Augmented Distractor Generation (RADG)**: for each question, we retrieve **correct-answer exemplars** from the training set, matched by question category and procedure.',
  '**Three LLMs** then rewrite the distractors to **match the style and specificity of correct answers**, while remaining incorrect given the transcript context. **Mixing candidates across generators** eliminates cues tied to any single model.'
 ]},
 2:{label:'Stage 3 · Curate',title:'Remove what language alone can solve',paragraphs:[
  'We train LLMs on **questions and options alone (without video)** so they learn to answer blindly, explicitly specializing in the language shortcuts we want to eliminate. These **fine-tuned guessers**, combined with **off-the-shelf models**, form a blind ensemble that attempts to answer every question without seeing the surgical video. Questions that any model **consistently answers correctly on unseen samples are discarded**: if the correct answer is reachable without visual evidence, the question cannot test visual reasoning.',
  'For the benchmark, guessers are fine-tuned on the full training set before blind testing. For the training data, we use **four-fold cross-validation to prevent leakage**: each guesser is trained on three folds and evaluated on the held-out fold.'
 ]},
 3:{label:'Stage 4 · Verify',title:'Check the benchmark with surgeons',paragraphs:[
  'A panel of **six expert surgeons** reviews candidates from the validation split. Reviewers inspect the video, aligned transcript, question, reference answer, and distractors for **temporal alignment, correctness, visual grounding, and valid alternatives**.',
  '**Rejected or flagged samples are discarded**, yielding the **340-question SUREON benchmark**. Expert verification applies to this evaluation split; SUREON-DB training data remain **automatically pseudo-labeled**.'
 ]}
};
let transitionId=0;
let animations=[];
async function setStage(value){
 const stage=stageNotes[value];
 if(!stage)return;
 const id=++transitionId;
 animations.forEach(animation=>animation.cancel());
 animations=[];
 const targets=[$('#pipeline-diagram'),$('#pipeline-details')].filter(Boolean);
 const fade=async(from,to,duration)=>{
  animations=targets.filter(target=>typeof target.animate==='function').map(target=>target.animate(
   [{opacity:from},{opacity:to}],{duration,easing:to===0?'ease-in':'ease-out',fill:'both'}
  ));
  await Promise.all(animations.map(animation=>animation.finished.catch(()=>{})));
 };
 if(!motion?.matches)await fade(1,0,300);
 if(id!==transitionId)return;
 document.querySelectorAll('[data-stage]').forEach(b=>{const active=b.dataset.stage===value;b.classList.toggle('active',active);b.setAttribute('aria-pressed',active);});
 $('#pipeline-details').innerHTML=`<p class="pipeline-detail-label">${esc(stage.label)}</p><h3 id="pipeline-detail-title">${esc(stage.title)}</h3>${stage.paragraphs.map(p=>`<p>${esc(p).replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>')}</p>`).join('')}`;
 const img=$('#pipeline-diagram');
 if(img&&stageImages[value])img.src=stageImages[value];
 const _sp=['all','0','1','2','3'];const _pos=_sp.indexOf(value)*25;
 const _th=$('#scrubber-thumb');const _fi=$('#scrubber-fill');
 if(_th&&!_th.dataset.dragging){_th.style.left=_pos+'%';_th.setAttribute('aria-valuenow',_sp.indexOf(value));_th.setAttribute('aria-valuetext',stage.label);}
 if(_fi&&!(_th&&_th.dataset.dragging))_fi.style.width=_pos+'%';
 if(!motion?.matches)await fade(0,1,500);
 if(id!==transitionId)return;
 animations.forEach(animation=>animation.cancel());
 animations=[];
}
const controls=$('.pipeline-controls');
if(!controls)return;
const stages=['all','0','1','2','3'];
let current='all';
let timer;
let stopped=false;
let visible=false;
const motion=window.matchMedia?.('(prefers-reduced-motion: reduce)');
function cancelTimer(){
 window.clearTimeout(timer);
 timer=undefined;
}
function stopCycling(){
 stopped=true;
 cancelTimer();
}
function scheduleNext(){
 cancelTimer();
 if(stopped||!visible||document.hidden||motion?.matches)return;
 timer=window.setTimeout(async()=>{
  current=stages[(stages.indexOf(current)+1)%stages.length];
  await setStage(current);
  scheduleNext();
 },4000);
}
// Any direct interaction leaves the controls in the reader's hands.
for(const event of ['pointerdown','keydown','focusin'])controls.addEventListener(event,stopCycling);
document.querySelectorAll('[data-stage]').forEach(button=>button.addEventListener('click',()=>{
 stopCycling();
 current=button.dataset.stage;
 setStage(current);
}));
// Begin the introduction when the pipeline is on screen, and pause off screen.
if('IntersectionObserver' in window){
 const observer=new window.IntersectionObserver(entries=>{
  visible=entries.some(entry=>entry.isIntersecting);
  scheduleNext();
 },{threshold:0.15});
 observer.observe($('#method'));
}else{
 visible=true;
 scheduleNext();
}
document.addEventListener('visibilitychange',scheduleNext);
motion?.addEventListener?.('change',scheduleNext);
// Scrubber drag
const _thumb=$('#scrubber-thumb');const _rail=$('.scrubber-rail');
if(_thumb&&_rail){
 const _so=['all','0','1','2','3'];
 function _snap(clientX){const r=_rail.getBoundingClientRect();return _so[Math.round(Math.max(0,Math.min(1,(clientX-r.left)/r.width))*4)];}
 _thumb.addEventListener('pointerdown',e=>{stopCycling();_thumb.setPointerCapture(e.pointerId);_thumb.dataset.dragging='';e.preventDefault();});
 _thumb.addEventListener('pointermove',e=>{if(!('dragging' in _thumb.dataset))return;const s=_snap(e.clientX);if(s!==current){current=s;setStage(s);}});
 _thumb.addEventListener('pointerup',()=>{delete _thumb.dataset.dragging;});
 _thumb.addEventListener('keydown',e=>{
  const i=_so.indexOf(current);
  if(e.key==='ArrowRight'||e.key==='ArrowUp'){stopCycling();const n=_so[Math.min(4,i+1)];current=n;setStage(n);e.preventDefault();}
  else if(e.key==='ArrowLeft'||e.key==='ArrowDown'){stopCycling();const n=_so[Math.max(0,i-1)];current=n;setStage(n);e.preventDefault();}
 });
}
})();
