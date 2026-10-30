/** One-time entrances throughout the page; scrolling and focus remain native. */
export function setupSectionReveals(){
 if(typeof window==='undefined'||!('IntersectionObserver' in window))return;
 const motion=window.matchMedia?.('(prefers-reduced-motion: reduce)');
 if(motion?.matches)return;
 // Select content blocks rather than overlapping animated parents and children.
 const targets=[...document.querySelectorAll([
  '.site-nav',
  '.paper-header > *',
  '.problem-section > .section-kicker',
  '.problem-section > .section-title',
  '.problem-section > .shortcut-thesis',
  '.problem-section > article > *',
  '.construction-bridge > *',
  'main > section:not(.paper-header):not(.problem-section):not(.abstract-section) > *',
  '.abstract-content > *',
  'footer'
 ].join(','))];
 let observer;
 let frame;
 let stopped=false;
 const reveal=element=>{
  element.classList.remove('reveal-pending');
  observer?.unobserve(element);
 };
 try{
  observer=new window.IntersectionObserver(entries=>{
   for(const entry of entries)if(entry.isIntersecting)reveal(entry.target);
  },{threshold:0,rootMargin:'0px 0px -24px 0px'});
  for(const element of targets){
   // Restored scroll positions need no entrances for content already above the viewport.
   if(element.getBoundingClientRect().bottom<=0)continue;
   const header=element.parentElement?.classList.contains('paper-header');
   const delay=header?Math.min([...element.parentElement.children].indexOf(element)*35,175):35;
   element.style.setProperty('--reveal-delay',delay+'ms');
   element.classList.add('section-reveal','reveal-pending');
  }
  // Let the hidden starting state paint, including for the initially visible header.
  frame=window.requestAnimationFrame(()=>{
   frame=window.requestAnimationFrame(()=>{
    if(stopped)return;
    try{
     for(const element of targets)if(element.classList.contains('reveal-pending'))observer.observe(element);
    }catch(error){
     targets.forEach(reveal);observer.disconnect();
    }
   });
  });
  document.addEventListener('focusin',event=>{
   const pending=event.target.closest?.('.reveal-pending');
   if(pending)reveal(pending);
  });
  motion?.addEventListener?.('change',event=>{
   if(event.matches){
    stopped=true;window.cancelAnimationFrame(frame);
    targets.forEach(reveal);observer.disconnect();
   }
  });
 }catch(error){
  // If enhancement fails, the complete page remains readable.
  stopped=true;window.cancelAnimationFrame(frame);
  targets.forEach(reveal);observer?.disconnect();
 }
}
