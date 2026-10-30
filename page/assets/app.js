import {setupSectionReveals} from './reveal.mjs';
import {escapeHtml as esc,makeResultsChart,makeAblationChart,makePublicChart} from './charts.mjs?v=20261028-gain';
const $=s=>document.querySelector(s);
let data;
const citation=`@article{perez2026sureon,
  title={SUREON: A Benchmark and Vision-Language Model for Surgical Reasoning},
  author={Perez, Alejandra and Rau, Anita and Mlambo, Busisiwe and White, Lee and
          Brunet, Aina and Verde, Juan and Caicedo, Juan and Lavanchy, Jo{\"e}l L. and
          Schmidt, Adam and Nwoye, Chinedu and Mohareri, Omid and Jamal, Muhammad Abdullah},
  journal={arXiv preprint arXiv:2603.06570},
  year={2026}
}`;

function renderGallery(category){
 const c=data.categories.find(x=>x.key===category),examples=data.examples.filter(x=>x.category===category);
 document.querySelectorAll('[data-category]').forEach(b=>{const active=b.dataset.category===category;b.setAttribute('aria-selected',active);b.tabIndex=active?0:-1;});
 $('#example-grid').setAttribute('aria-labelledby','qa-'+category);
 $('#gallery-title').textContent=c.label;
 $('#gallery-format').textContent={mcq:'Multiple choice',order:'Temporal sequence',summary:'Open-ended summary'}[c.format];
 $('#example-grid').querySelectorAll('video').forEach(v=>v.pause());
 $('#example-grid').innerHTML=examples.map((e,i)=>{
  const events=e.events?.length?`<ol class="event-list">${e.events.map(x=>`<li><strong>${esc(x.id)}:</strong> ${esc(x.text)}</li>`).join('')}</ol>`:'';
  let answers;
  if(c.format==='mcq'){
   answers=`<ul class="choices-list" aria-label="Answer choices">${e.options.map(option=>{
    const correct=option===e.answer;
    return `<li class="choice${correct?' correct-choice':''}"><span class="choice-marker" aria-hidden="true">${correct?'✓':'·'}</span><span>${esc(option)}${correct?'<span class="sr-only"> (correct answer)</span>':''}</span></li>`;
   }).join('')}</ul>`;
  }else{
   const answer=c.format==='order'?e.answer.replaceAll(',',' → '):e.answer;
   answers=`<div class="correct-choice open-answer"><span class="choice-marker" aria-hidden="true">✓</span><p>${esc(answer)}</p></div>`;
  }
  return `<article class="example"><video controls playsinline preload="none" poster="${esc(e.poster)}" aria-label="${esc(c.label)}, example ${i+1}"><source src="${esc(e.video)}" type="video/mp4"></video><div class="example-meta"><span>${esc(e.procedure)}</span><span>${e.duration.toFixed(1)} s</span></div><p class="example-question">${esc(e.question)}</p>${events}${answers}</article>`;
 }).join('');
 bindVideos();
}
function bindVideos(){
 document.querySelectorAll('video').forEach(v=>{
  if(v.dataset.bound)return;v.dataset.bound='true';
  v.addEventListener('play',()=>document.querySelectorAll('video').forEach(other=>{if(other!==v)other.pause();}));
 });
}
function renderDistribution(){
 // Both charts use the same percentage scale; raw counts retain their own denominators.
 const maxShare=Math.max(...data.categories.flatMap(c=>[c.train/76061,c.benchmark/340]));
 for(const [split,total,selector] of [['train',76061,'#training-distribution'],['benchmark',340,'#benchmark-distribution']]){
  const target=$(selector);
  target.setAttribute('aria-label',`${split==='train'?'Training':'Benchmark'} question category counts; denominator ${total.toLocaleString()} QA pairs`);
  target.innerHTML=`<ul class="distribution-rows">${data.categories.map(c=>{
   const share=c[split]/total;
   return `<li class="count-row"><span>${esc(c.label)}</span><div class="count-track" title="${c[split].toLocaleString()} / ${total.toLocaleString()} (${(share*100).toFixed(1)}%)"><span style="width:${share/maxShare*100}%"></span></div><b>${c[split].toLocaleString()}</b></li>`;
  }).join('')}</ul>`;
 }
}

function renderResults(){
 const category=$('#result-select').value;
 $('#main-chart').innerHTML=makeResultsChart(data,category);
 $('#result-caption').textContent='';
}

function renderPublic(){
 const key=$('#public-select').value,p=data.perception[key];$('#public-chart').innerHTML=makePublicChart(p);
 $('#public-table').innerHTML=`<caption>${esc(p.label)} · ${esc(p.metric)}</caption><thead><tr><th scope="col">Model / input</th><th scope="col">Score (%)</th></tr></thead><tbody>${p.rows.map(([name,value])=>`<tr><th scope="row">${esc(name)}</th><td>${value}</td></tr>`).join('')}</tbody>`;
 $('#public-caption').textContent='';
}
async function init(){
 setupSectionReveals();
 try{const r=await fetch('page/data/research.json?v=20261028-paper');if(!r.ok)throw Error(r.status);data=await r.json();}catch(e){$('#example-grid').innerHTML='<p>Examples could not be loaded. Please refresh the page or try again later.</p>';return;}
 $('#qa-tabs').innerHTML=data.categories.map(c=>`<button type="button" id="qa-${c.key}" role="tab" aria-controls="example-grid" aria-selected="false" tabindex="-1" data-category="${c.key}">${esc(c.label)}</button>`).join('');
 $('#qa-tabs').addEventListener('click',e=>{const b=e.target.closest('[data-category]');if(b)renderGallery(b.dataset.category);});
 $('#qa-tabs').addEventListener('keydown',e=>{
  if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;
  const tabs=[...$('#qa-tabs').querySelectorAll('[role="tab"]')],i=tabs.indexOf(document.activeElement);if(i<0)return;
  e.preventDefault();const next=e.key==='Home'?0:e.key==='End'?tabs.length-1:(i+(e.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;
  renderGallery(tabs[next].dataset.category);tabs[next].focus();
 });
 renderGallery('object_existence');
 renderDistribution();
 $('#result-select').insertAdjacentHTML('beforeend',data.categories.map(c=>`<option value="${c.key}">${esc(c.label)}</option>`).join(''));
 $('#result-select').addEventListener('change',renderResults);renderResults();
 $('#ablation-chart').innerHTML=makeAblationChart(data.ablation);
 $('#public-select').innerHTML=Object.entries(data.perception).map(([key,p])=>`<option value="${key}">${esc(p.label)}</option>`).join('');$('#public-select').addEventListener('change',renderPublic);renderPublic();
 $('#citation-text').textContent=citation;
 $('#copy-citation').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(citation);$('#copy-status').textContent='BibTeX copied.';}catch(e){$('#copy-status').textContent='Clipboard access is unavailable. Select and copy the citation above.';}});
 bindVideos();
}
init();
