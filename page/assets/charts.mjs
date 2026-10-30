// Data-driven scientific figures. Axes start at zero; metrics remain separate.
export const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const e=escapeHtml;
const text=(x,y,s,size=12,fill='#657086',anchor='start',weight=400)=>`<text x="${x}" y="${y}" font-family="Arial,Helvetica,sans-serif" font-size="${size}" fill="${fill}" text-anchor="${anchor}" font-weight="${weight}">${e(s)}</text>`;
const svg=(w,h,title,content)=>`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" role="img" aria-label="${e(title)}"><title>${e(title)}</title>${content}</svg>`;
export function makeResultsChart(data,category='overall'){
 const overall=category==='overall',summary=category==='sequence_summarization';
 const models=data.models.filter(m=>m.group!=='ablation'&&(!summary||m.group!=='human'));
 const groupLabels={commercial:'Commercial',general:'Open-source',medical:'Medical'};
 const rows=[];let prevGroup=null;
 models.forEach(m=>{
  if(m.group!=='human'&&m.group!=='ours'&&m.group!==prevGroup)rows.push({type:'header',label:groupLabels[m.group]||m.group});
  rows.push({type:'model',m});
  prevGroup=m.group;
 });
 const x=230,w=610,rowH=55,headerH=22,top=58,gainX=900,gainW=320;
 let totalH=top;rows.forEach(r=>totalH+=r.type==='header'?headerH:rowH);totalH+=42;
 // Keep the same gain scale across views so bar lengths remain comparable.
 const gainLeft=gainX+30,gainAxisW=220;
 const gainPosition=value=>gainLeft+(value+20)/80*gainAxisW;
 const gainZero=gainPosition(0);
 let s='';
 // Gain column tinted background panel
 s+=`<rect x="${gainX}" y="${top-30}" width="${gainW}" height="${totalH-top+26}" rx="5" fill="#f0f3fb"/>`;
 for(let tick=0;tick<=100;tick+=20){const xx=x+w*tick/100;s+=`<line x1="${xx}" x2="${xx}" y1="${top-18}" y2="${totalH-40}" stroke="#edf0f5"/>`+text(xx,totalH-17,String(tick),10,'#9098a8','middle');}
 s+=text(x,17,summary?'LLM-judge score (%)':'Accuracy (%)',11);
 // A separate, signed scale shows both visual benefit and blind advantage.
 s+=text(gainX+gainW/2,13,'Multimodal gain',14,'#4059a7','middle',700);
 s+=text(gainX+gainW/2,30,'With video − without video · pp',10,'#697b9d','middle');
 for(const tick of [-20,0,20,40,60]){
  const xx=gainPosition(tick);
  s+=`<line x1="${xx}" x2="${xx}" y1="${top-10}" y2="${totalH-40}" stroke="${tick===0?'#97a6c4':'#e0e6f1'}" stroke-width="${tick===0?1.3:1}" ${tick===0?'stroke-dasharray="3 3"':''}/>`;
  s+=text(xx,totalH-17,tick>0?'+'+tick:String(tick),10,'#7887a2','middle');
 }
 let y=top;
 rows.forEach(row=>{
  if(row.type==='header'){s+=`<line x1="${x-60}" x2="${x+w+8}" y1="${y+headerH-4}" y2="${y+headerH-4}" stroke="#edf0f5" stroke-width="1"/>`+text(x-18,y+headerH-7,row.label,9,'#a0a8b8','end');y+=headerH;return;}
  const m=row.m,v=overall?m.video:m.categories[category],b=overall?m.blind:(m.blindCategories[category]??null);
  const ours=m.group==='ours',color=m.group==='human'?'#b28c57':ours?'#465eb4':'#8296ce';
  s+=text(x-18,y+12,m.name,13,ours?'#4059a7':'#4b566e','end',ours?700:400);
  s+=`<rect x="${x}" y="${y}" width="${w*v/100}" height="12" rx="1" fill="${color}"/>`;
  if(overall){let a=x+w*m.ci[0]/100,z=x+w*m.ci[1]/100;s+=`<path d="M${a} ${y+6}H${z}M${a} ${y+2}v8M${z} ${y+2}v8" fill="none" stroke="#374461" stroke-width="1.1"/>`;}
  s+=text(x+w*(overall?m.ci[1]:v)/100+8,y+10,`${v.toFixed(1)}%`,11,color,'start',ours?600:400);
  if(b!==null){
   s+=`<rect x="${x}" y="${y+17}" width="${w*b/100}" height="9" rx="1" fill="#d4dae9"/>`+text(x+w*b/100+6,y+25,`${b.toFixed(1)}%`,10,'#929bad');
   const gain=overall?m.gain:m.categoryGains[category];
   const gainStr=`${gain>0?'+':''}${gain.toFixed(1)}`;
   const end=gainPosition(gain),barX=Math.min(gainZero,end),barW=Math.abs(end-gainZero);
   const gainColor=gain<0?'#aa6961':ours?'#4059a7':'#8b9fc9';
   s+=`<g><title>${e(m.name)}: ${gainStr} percentage points with video versus without video</title>`;
   if(barW>0)s+=`<rect x="${barX}" y="${y+2}" width="${barW}" height="14" rx="3" fill="${gainColor}"/>`;
   else s+=`<circle cx="${gainZero}" cy="${y+9}" r="3" fill="#8b9bb9"/>`;
   s+=text(gainX+gainW-12,y+14,gainStr,14,gain<0?'#9a5c54':ours?'#304995':'#526582','end',ours?700:500);
   s+='</g>';
  }else{
   s+=text(gainX+gainW-12,y+14,'—',14,'#8b97ae','end');
  }
  y+=rowH;
 });
 return svg(1240,totalH,`Reported ${summary?'sequence-summarization score':'accuracy'} with video and without video; signed multimodal gain in percentage points; human baseline shown where available`,s);
}
export function makeAblationChart(stages){
 const w=480,h=310,base=235,scale=190/30,barw=70,positions=[90,240,390];let s='';
 for(let tick=0;tick<=30;tick+=10){const y=base-tick*scale;s+=`<line x1="35" x2="465" y1="${y}" y2="${y}" stroke="#edf0f5"/>`+text(23,y+4,String(tick),10,'#929bad','end');}
 s+=text(35,18,'Multimodal gain (percentage points)',11);
 stages.forEach((a,i)=>{const x=positions[i]-barw/2,y=base-a.gain*scale;s+=`<rect x="${x}" y="${y}" width="${barw}" height="${base-y}" rx="2" fill="${i===2?'#4b65bc':'#b8c5e6'}"/>`+text(positions[i],y-12,`+${a.gain}`,17,i===2?'#4b65bc':'#7d8eaf','middle',600)+text(positions[i],base+25,['Synthesis','Refinement','Blind curation'][i],11,'#59667d','middle')+text(positions[i],base+44,i===2?'76K pairs':'176.6K pairs',10,'#919aab','middle');});
 return svg(w,h,`Construction ablation: reported multimodal gains ${stages.map(a=>a.gain).join(', ')} points`,s);
}
export function makePublicChart(p){
 const w=480,left=175,axis=255,top=35,step=39,h=top+p.rows.length*step+30;let s='';
 for(let tick=0;tick<=100;tick+=25){const xx=left+axis*tick/100;s+=`<line x1="${xx}" x2="${xx}" y1="20" y2="${h-28}" stroke="#edf0f5"/>`+text(xx,h-8,String(tick),9,'#9098aa','middle');}
 p.rows.forEach(([name,v],i)=>{const y=top+i*step,ours=name.startsWith('SUREON-VLM');s+=text(left-12,y+13,name,11,ours?'#425ba9':'#69758c','end',ours?600:400)+`<rect x="${left}" y="${y}" width="${axis*v/100}" height="18" rx="1" fill="${ours?'#526cc3':'#cbd4e9'}"/>`+text(left+axis*v/100+8,y+13,String(v),11,ours?'#425ba9':'#8290a7');});
 return svg(w,h,`${p.label}: ${p.metric}. ${p.protocol}`,s);
}
