"""Capture review UI adapted from Task_B; see docs/CAPTURE.md."""
import base64
import cv2
from eye_detection import FONT, COL_OK, COL_WARN

def annotate(frame, cand, title, selected=False):
    """Copy of the frame with the eye oval and the sharpness / brightness / glare values printed on it."""
    img = frame.copy()
    h, w = img.shape[:2]
    scale = max(0.5, h / 720.0)
    thick = max(1, int(round(scale * 2)))

    eye = cand.get("eye")
    if eye is not None:
        o = eye["oval"]
        cv2.ellipse(img, (int(o["cx"]), int(o["cy"])), (int(o["rx"]), int(o["ry"])), o["angle"],
                    0, 360, COL_OK if cand["ok"] else COL_WARN, max(2, int(round(scale * 3))), cv2.LINE_AA)

    lines = [(title + ("  [SELECTED]" if selected else ""), COL_OK if selected else (255, 255, 255)),
             (f"Sharpness : {cand['sharpness']:.1f}", (255, 255, 255)),
             (f"Brightness: {cand['brightness']:.1f}", (255, 255, 255)),
             (f"Glare     : {cand['glare'] * 100:.2f}%", (255, 255, 255))]
    if eye is not None:
        lines.append((f"Redness (diagnostic) : {cand['coverage'] * 100:.0f}%", (255, 255, 255)))
    if eye is None:
        lines.append(("no eye found - whole-oval metrics", COL_WARN))

    sizes = [cv2.getTextSize(t, FONT, scale, thick)[0] for t, _ in lines]
    pad = int(10 * scale)
    lh = max(s[1] for s in sizes) + int(10 * scale)
    box_w = min(w, max(s[0] for s in sizes) + 2 * pad)
    box_h = min(h, lh * len(lines) + pad)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    for i, (t, col) in enumerate(lines):
        cv2.putText(img, t, (pad, int(pad / 2 + lh * (i + 1))), FONT, scale, col, thick, cv2.LINE_AA)
    return img


def _jpeg_b64(img, width=None, quality=85):
    if width and img.shape[1] > width:
        img = cv2.resize(img, (width, int(round(img.shape[0] * width / img.shape[1]))),
                         interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode()


# ----------------------------------------------------------------------------
# Browser UI
# ----------------------------------------------------------------------------
HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Eye Capture</title>
<style>
 *{box-sizing:border-box}body{margin:0;background:#0c131b;color:#e9eff5;font:15px/1.5 system-ui,sans-serif}
 .shell{max-width:1120px;margin:auto;padding:24px}header{display:flex;justify-content:space-between;align-items:center;gap:20px;margin-bottom:22px}
 h1{font-size:26px;letter-spacing:-.7px;margin:0}p{margin:4px 0;color:#9bafc2}.brand{font-size:11px;font-weight:700;letter-spacing:2px;color:#76d8ba}
 .steps{display:flex;gap:16px;color:#7d91a5;font-size:13px}.steps .active{color:#82e2c2;font-weight:700}
 .card{background:#141f2b;border:1px solid #29394a;border-radius:16px;padding:18px;margin-bottom:16px}
 .setup{display:flex;gap:20px;align-items:center;flex-wrap:wrap}.field{display:flex;flex-direction:column;gap:6px;min-width:150px;flex:1}
 label,summary{font-weight:600;font-size:13px}small,.hint{color:#9bafc2;font-size:12px}.field input,.field select{width:100%;background:#0c1621;color:#ecf5fc;border:1px solid #3d5266;border-radius:8px;padding:10px;font:inherit}
 input:focus,select:focus,button:focus-visible,summary:focus-visible{outline:2px solid #82e2c2;outline-offset:3px}
 input:disabled,select:disabled{opacity:.55}details{margin-top:14px}summary{cursor:pointer;color:#b5c8d9}details .setup{margin-top:12px}
 button{font:600 14px system-ui;padding:11px 18px;border:1px solid #3d5266;border-radius:9px;background:#243548;color:#eff7ff;cursor:pointer}
 button:hover{filter:brightness(1.15)}button:disabled{cursor:wait;opacity:.5}.primary{background:#7fe0bd;color:#062b22;border-color:#7fe0bd}.secondary{background:transparent}
 #msg{margin:0;font-size:15px;font-weight:600}#info{margin:4px 0 14px;font-size:13px;color:#acc3d7;min-height:20px}
 #stage{position:relative;aspect-ratio:16/9;background:#03080c;border-radius:12px;overflow:hidden;width:100%}
 video,img#rev{width:100%;height:100%;object-fit:cover;position:absolute;inset:0}
 canvas#guide{position:absolute;inset:0;width:100%;height:100%;touch-action:none;cursor:crosshair}
 #selection{position:absolute;border:2px dashed #82e2c2;pointer-events:none;background:#82e2c222}
 #count{position:absolute;top:16px;right:20px;font-size:40px;font-weight:700;color:#fff;text-shadow:0 2px 8px #000}
 .guidehelp{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-top:12px}.guidehelp button{font-size:12px;padding:7px 10px;flex-shrink:0}
 #reviewArea{display:flex;gap:14px;align-items:start;margin-top:14px}#cands{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;flex:3;min-width:0}
 .cand{border:1px solid #3a5066;background:#0d1722;padding:6px;border-radius:9px;font-size:11px;color:#9bb3c8;min-width:0}.cand.best{border:2px solid #7fe0bd}.cand img{display:block;width:100%;border-radius:4px;margin-bottom:6px}.cand strong{display:block;color:#e9f6ff;font-size:12px;margin-bottom:2px}
 #cropbox{flex:1;min-width:0;background:#0a131c;border:1px solid #3a5066;border-radius:9px;padding:9px;font-size:12px}#crop{display:block;width:100%;height:auto;max-height:180px;object-fit:contain;margin-top:8px}
 #btns{display:flex;gap:10px;justify-content:center;margin-top:18px}#saved{font-size:12px;color:#a5bacd;overflow-wrap:anywhere}#saved div{margin:4px 0}.savehead{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:10px}.savehead strong{font-size:14px;color:#edf7ff}#lastSave{color:#82e2c2}
 .hide{display:none!important}@media(max-width:640px){.shell{padding:12px}header{align-items:start;flex-direction:column;gap:10px}.steps{gap:14px}.card{padding:12px}.setup{gap:12px}.field{min-width:120px}#reviewArea{flex-direction:column}#cands{width:100%;grid-template-columns:repeat(2,minmax(0,1fr))}#cropbox{width:100%}#crop{max-height:140px}.guidehelp{align-items:start}.savehead{align-items:start}}
</style></head><body><main class="shell">
<header><div><div class="brand">CONJUNCTIVA IMAGING</div><h1>Eye Capture</h1><p>Capture a short video. Review the clearest frame. Save your images.</p></div><nav class="steps" aria-label="Capture progress"><span id="stepCapture" class="active">1 Capture</span><span id="stepReview">2 Review</span><span id="stepSave">3 Save</span></nav></header>
<section class="card" aria-label="Capture details"><form id="captureForm">
<div class="setup"><div class="field"><label for="patientId">Patient ID <small>(automatic)</small></label><input id="patientId" readonly aria-describedby="patientHelp"><small id="patientHelp">Assigned automatically when you save. No typing needed.</small><label id="samePatientRow" class="hide"><input id="samePatient" type="checkbox" style="width:auto;margin-right:6px">Same patient as the last save</label></div><div class="field"><label for="duration">Recording length</label><select id="duration"><option value="3">3 seconds</option><option value="5">5 seconds</option><option value="8">8 seconds</option></select><small>Hold still until the countdown ends.</small></div></div>
<details id="captureOptions"><summary>Camera &amp; optional details</summary><div class="setup">
<div class="field"><label for="cameraSelect">Camera</label><select id="cameraSelect"><option value="">Default camera</option></select></div>
<div class="field"><label for="deviceLabel">Device label</label><input id="deviceLabel" maxlength="80" pattern="[A-Za-z0-9][A-Za-z0-9_-]*" placeholder="webcam"><small>Letters, numbers, hyphens or underscores.</small></div>
<div class="field"><label for="distanceLabel">Distance</label><select id="distanceLabel"><option value="unspecified">Not recorded</option><option value="5cm">5 cm</option><option value="10cm">10 cm</option><option value="15cm">15 cm</option><option value="20cm">20 cm</option><option value="30cm">30 cm</option></select></div>
<div class="field"><label for="lightingLabel">Lighting</label><select id="lightingLabel"><option value="unspecified">Not recorded</option><option value="indoor">Indoor</option><option value="daylight">Daylight</option><option value="diffuse">Diffuse light</option><option value="flash">Flash</option></select></div>
</div><p class="hint">These details are optional labels. Distance and lighting are not measured automatically.</p></details></form></section>
<section class="card" aria-label="Camera and review"><div id="msg" role="status">Starting your camera...</div><div id="info">Allow camera access if asked.</div>
<div id="stage"><video id="cam" autoplay muted playsinline></video><img id="rev" class="hide" alt="Selected full-resolution frame"><canvas id="guide" aria-label="Drag to select the exposed lower eyelid"></canvas><div id="selection" class="hide"></div><div id="count" aria-live="polite"></div></div>
<div class="guidehelp"><span class="hint">Bring the eye close and expose the lower lid. If the guide misses it, drag a box over the exposed tissue.</span><button id="auto" class="secondary">Reset eye guide</button></div>
<div id="reviewArea"><div id="cands" class="hide"></div><div id="cropbox" class="hide"><div>Extracted tissue</div><img id="crop" alt="Conjunctiva extraction preview"></div></div>
<div id="btns"><button id="go" class="primary hide">Start capture</button><button id="keep" class="primary hide">Save capture</button><button id="disc" class="secondary hide">Retake</button><button id="retryCamera" class="secondary hide">Retry camera</button></div></section>
<section class="card" id="saved" aria-label="Save location"><div class="savehead"><strong>Your saved captures</strong><button id="openFolder" class="secondary">Open save folder</button></div><div id="outputPath"></div><div id="csvPath"></div><div id="lastSave" role="status">Each save adds an automatic ID and image paths to your CSV.</div></section>
</main>
<script>
const $=id=>document.getElementById(id);
const DURATION=%DURATION%;
const OUTPUT=%OUTPUT_JSON%;
const DETAILS=%DETAILS_JSON%;
function setSelect(id,value){
  const select=$(id);if(![...select.options].some(o=>o.value===String(value))) select.add(new Option(String(value),String(value)));
  select.value=String(value);
}
let lastPatientId='',nextPatientId=DETAILS.next_patient_id;
function showPatient(){
  $('patientId').value=DETAILS.patient_id||($('samePatient').checked?lastPatientId:nextPatientId);
  $('patientHelp').textContent=DETAILS.patient_id?'Using the existing patient ID supplied at launch.':$('samePatient').checked?'This capture will use the same patient ID.':'Assigned automatically when you save. No typing needed.';
}
$('samePatient').onchange=showPatient;
showPatient();$('deviceLabel').value=DETAILS.device;
setSelect('distanceLabel',DETAILS.distance);setSelect('lightingLabel',DETAILS.lighting);setSelect('duration',DURATION);
$('duration').selectedOptions[0].textContent=DURATION+' seconds';
$('captureForm').onsubmit=e=>e.preventDefault();
function setStage(stage){for(const name of ['Capture','Review','Save']) $('step'+name).classList.toggle('active',name===stage);}
function lockDetails(locked){document.querySelectorAll('#captureForm input,#captureForm select').forEach(e=>e.disabled=locked);}
function captureDetails(){return {patient_id:DETAILS.patient_id||($('samePatient').checked?lastPatientId:''),device:$('deviceLabel').value.trim(),distance:$('distanceLabel').value,lighting:$('lightingLabel').value};}
$('outputPath').textContent='Save folder: '+OUTPUT;
$('csvPath').textContent='CSV index: '+OUTPUT+'/dataset.csv';
const OVAL_RX=%OVAL_RX%, OVAL_RY=%OVAL_RY%;   // search hint shown until an eye is found
const TRACK_W=640;      // width of the live frames sent to Python for eye tracking
const SMOOTH=0.55;      // 0..1, higher = oval follows faster (lower = steadier)
const LOST_MAX=0;       // misses in a row before the oval is dropped
let stream, tracking=false, cur=null, lost=0;
let selecting=null, recording=false;
let loopStarted=false;
const grab=document.createElement('canvas'), gctx=grab.getContext('2d');
const sleep=ms=>new Promise(r=>setTimeout(r,ms));

function pointerPosition(e){
  const rect=$('guide').getBoundingClientRect();
  return {x:Math.max(0,Math.min(rect.width,e.clientX-rect.left)),y:Math.max(0,Math.min(rect.height,e.clientY-rect.top))};
}
$('guide').onpointerdown=e=>{
  if(recording || !tracking) return;
  selecting=pointerPosition(e); $('guide').setPointerCapture(e.pointerId);
};
$('guide').onpointermove=e=>{
  if(!selecting) return;
  const p=pointerPosition(e), box=$('selection');box.classList.remove('hide');
  Object.assign(box.style,{left:Math.min(p.x,selecting.x)+'px',top:Math.min(p.y,selecting.y)+'px',width:Math.abs(p.x-selecting.x)+'px',height:Math.abs(p.y-selecting.y)+'px'});
};
$('guide').onpointerup=async e=>{
  if(!selecting) return;
  const start=selecting,p=pointerPosition(e); selecting=null; $('selection').classList.add('hide');
  const v=$('cam'),r=$('guide').getBoundingClientRect();
  const s=Math.max(r.width/v.videoWidth,r.height/v.videoHeight),ox=(r.width-v.videoWidth*s)/2,oy=(r.height-v.videoHeight*s)/2;
  const x=(Math.min(start.x,p.x)-ox)/s/v.videoWidth,y=(Math.min(start.y,p.y)-oy)/s/v.videoHeight;
  const w=Math.abs(start.x-p.x)/s/v.videoWidth,h=Math.abs(start.y-p.y)/s/v.videoHeight;
  if(w<.02 || h<.02) return;
  const result=await window.pyRegion([x,y,w,h]);cur=null;
  $('msg').textContent=result.error || 'Selected region: hold the exposed conjunctiva inside the guide.';
};
$('guide').onpointercancel=()=>{selecting=null;$('selection').classList.add('hide');};
$('auto').onclick=async()=>{if(recording || !tracking) return;await window.pyRegion(null);cur=null;$('msg').textContent='Automatic eye detection enabled.';};

async function init(){
  $('retryCamera').classList.add('hide');$('go').classList.add('hide');tracking=false;
  try{
    if(stream) stream.getTracks().forEach(t=>t.stop());
    const selected=$('cameraSelect').value;
    stream=await navigator.mediaDevices.getUserMedia({video:{width:1280,height:720,...(selected?{deviceId:{exact:selected}}:{facingMode:'environment'})},audio:false});
    const cameraTrack=stream.getVideoTracks()[0];
    const capabilities=cameraTrack.getCapabilities ? cameraTrack.getCapabilities() : {};
    if(capabilities.focusMode && capabilities.focusMode.includes('continuous')){
      try{ await cameraTrack.applyConstraints({advanced:[{focusMode:'continuous'}]}); }catch(e){}
    }
    $('cam').srcObject=stream;
    await $('cam').play();
    const devices=await navigator.mediaDevices.enumerateDevices();
    const actual=cameraTrack.getSettings().deviceId;
    $('cameraSelect').replaceChildren(...devices.filter(d=>d.kind==='videoinput').map((d,i)=>new Option(d.label||'Camera '+(i+1),d.deviceId)));
    if(actual) $('cameraSelect').value=actual;
    $('msg').textContent='Ready. Bring the eye close and expose the lower lid.';
    $('go').classList.remove('hide');
    tracking=true;if(!loopStarted){loopStarted=true;trackLoop();}
  }catch(e){$('msg').textContent='Camera unavailable. Check that it is connected and not in use by another app.';$('info').textContent=e.message;$('retryCamera').classList.remove('hide');}
}
$('cameraSelect').onchange=async()=>{await window.pyRegion(null);cur=null;await init();};
$('retryCamera').onclick=init;
$('openFolder').onclick=async()=>{try{const r=await window.pyOpenFolder();if(r.error) throw new Error(r.error);}catch(e){$('lastSave').textContent='Could not open the folder: '+e.message;}};
function toB64(blob){return new Promise(r=>{const f=new FileReader();f.onload=()=>r(f.result.split(',')[1]);f.readAsDataURL(blob);});}

// ---- dynamic oval: green = conjunctiva clear, amber = eye found but not clear, dashed white = searching
function drawGuide(){
  const c=$('guide'), st=$('stage'), W=st.clientWidth, H=st.clientHeight, v=$('cam');
  if(!W||!H) return;
  if(c.width!==W||c.height!==H){c.width=W;c.height=H;}
  const g=c.getContext('2d'); g.clearRect(0,0,W,H);
  const vw=v.videoWidth||16, vh=v.videoHeight||9;
  const s=Math.max(W/vw,H/vh), ox=(W-vw*s)/2, oy=(H-vh*s)/2;   // same mapping as object-fit:cover
  g.lineWidth=3;
  g.beginPath();
  if(cur){
    g.setLineDash([]); g.strokeStyle=cur.ok?'#00e676':'#ffca28';
    g.ellipse(cur.cx*s+ox,cur.cy*s+oy,cur.rx*s,cur.ry*s,cur.ang,0,2*Math.PI);
  }else{
    g.setLineDash([10,6]); g.strokeStyle='rgba(255,255,255,.55)';
    g.ellipse(W/2,H/2,OVAL_RX*W,OVAL_RY*H,0,0,2*Math.PI);
  }
  g.stroke();
}
async function trackLoop(){
  const v=$('cam');
  while(true){
    if(!tracking||!v.videoWidth){ if(v.videoWidth&&$('rev').classList.contains('hide')) drawGuide(); await sleep(100); continue; }
    const tw=TRACK_W, th=Math.round(TRACK_W*v.videoHeight/v.videoWidth);
    grab.width=tw; grab.height=th; gctx.drawImage(v,0,0,tw,th);
    let r=null;
    try{ r=await window.pyTrack(grab.toDataURL('image/jpeg',0.7).split(',')[1]); }catch(e){}
    if(!tracking) continue;                       // analysis started meanwhile
    if(r&&r.found){
      const k=v.videoWidth/r.w;                   // tracked-frame px -> camera px
      const t={cx:r.cx*k,cy:r.cy*k,rx:r.rx*k,ry:r.ry*k,ang:r.angle*Math.PI/180,ok:r.ok};
      if(cur&&Math.hypot(t.cx-cur.cx,t.cy-cur.cy)<cur.rx*1.5){
        for(const q of ['cx','cy','rx','ry','ang']) cur[q]+=(t[q]-cur[q])*SMOOTH;
        cur.ok=t.ok;
      }else cur=t;                                // first lock, or jumped to the other eye
      lost=0;
    }else if(++lost>LOST_MAX) cur=null;
    if(r && r.feedback) $('info').textContent=r.feedback;
    drawGuide();
    await sleep(80);
  }
}

function showCands(list){
  const box=$('cands'); box.innerHTML='';
  list.forEach(c=>{
    const d=document.createElement('div'); d.className='cand'+(c.best?' best':'');
    const im=document.createElement('img'); im.src='data:image/jpeg;base64,'+c.image;
    const t=document.createElement('div'),title=document.createElement('strong');
    title.textContent={overall:'Selected frame',sharpness:'Best focus',exposure:'Best exposure',glare:'Least glare'}[c.name]||c.name;
    im.alt=title.textContent;
    t.textContent=`Focus ${c.sharpness.toFixed(0)} · Light ${c.brightness.toFixed(0)} · Glare ${(c.glare*100).toFixed(1)}%`;
    d.appendChild(im);d.appendChild(title);d.appendChild(t);box.appendChild(d);
  });
  box.classList.remove('hide');
}

$('go').onclick=async()=>{
  if(!$('captureForm').reportValidity()) return;
  const details=captureDetails(),duration=Number($('duration').value);
  lockDetails(true);setStage('Capture');
  recording=true; $('auto').disabled=true;
  $('go').classList.add('hide'); $('info').textContent='';
  let rec,iv;
  try{
  const chunks=[];rec=new MediaRecorder(stream,{mimeType:'video/webm;codecs=vp8',videoBitsPerSecond:8000000});
  rec.ondataavailable=e=>{if(e.data.size)chunks.push(e.data)};
  const done=new Promise(r=>rec.onstop=r);
  $('msg').textContent="Hold steady...";
  rec.start(); let t=duration; $('count').textContent=t;
  iv=setInterval(()=>{t--; $('count').textContent=t>0?t:'';},1000);
  await new Promise(r=>setTimeout(r,duration*1000));
  clearInterval(iv); rec.stop(); await done;
  tracking=false;
  $('msg').textContent="Analysing frames...";
  const res=await window.pyProcess(await toB64(new Blob(chunks,{type:'video/webm'})),details);
  if(res.error) throw new Error(res.error);
  setStage('Review');
  $('rev').src='data:image/jpeg;base64,'+res.image; $('rev').classList.remove('hide');
  $('cam').classList.add('hide'); $('guide').classList.add('hide');
  if(res.crop){
    $('crop').src='data:image/jpeg;base64,'+(res.roi||res.crop);
    $('cropbox').firstElementChild.textContent=res.roi?'Extracted conjunctiva ROI':'Eye crop — ROI extraction failed';
    $('cropbox').classList.remove('hide');
  }
  showCands(res.cands);
  let m;
  if(!res.crop) m="Warning: no eye could be found in any frame. Retake?";
  else if(res.roi_status!=='ok') m='ROI extraction failed: '+res.roi_reason+'. Move closer or select the exposed lower lid.';
  else if(!res.crop_ok) m="Quality check: "+res.feedback+". Review or retake.";
  else m='Review the selected frame and tissue preview, then save or retake.';
  $('msg').textContent=m;
  let txt=`Focus ${res.sharpness.toFixed(0)} · Brightness ${res.brightness.toFixed(0)} · Glare ${(res.glare*100).toFixed(1)}%`;
  if(res.crop) txt+=res.eye_method==='manual_region'?' · Selected area':' · Eye detected';
  $('info').textContent=txt;
  $('keep').classList.remove('hide'); $('disc').classList.remove('hide');
  }catch(e){
    clearInterval(iv);if(rec&&rec.state==='recording') rec.stop();$('count').textContent='';
    reset();$('msg').textContent='Capture failed: '+e.message+'. Please try again.';
  }
};
function reset(){
  lockDetails(false);setStage('Capture');
  recording=false; $('auto').disabled=false;
  $('rev').classList.add('hide'); $('cam').classList.remove('hide'); $('guide').classList.remove('hide');
  $('cropbox').classList.add('hide'); $('cands').classList.add('hide');
  $('keep').classList.add('hide'); $('disc').classList.add('hide'); $('go').classList.remove('hide');
  cur=null; lost=0; tracking=true;
}
$('keep').onclick=async()=>{
  $('keep').disabled=true;
  try {
    const r=await window.pyKeep(); if(r.error) throw new Error(r.error);
    lastPatientId=r.patient_id;nextPatientId=r.next_patient_id;
    if(!DETAILS.patient_id) $('samePatientRow').classList.remove('hide');
    showPatient();reset();setStage('Save');$('msg').textContent='Saved: '+r.image_id+' · Patient '+r.patient_id+'. Ready for the next capture.';
    $('csvPath').textContent='CSV index: '+r.csv_path;
    $('lastSave').textContent='Last capture: '+r.capture_dir+' | 3 best frames saved'+(r.roi_image_path?' + cropped eye + ROI.':r.eye_image_path?' + cropped eye | ROI extraction failed.':' | No eye detected: crop and ROI unavailable.');
  } catch(e) { $('msg').textContent='Save failed: '+e.message; }
  finally { $('keep').disabled=false; }
};
$('disc').onclick=async()=>{await window.pyDiscard(); reset(); $('msg').textContent="Discarded. Ready to retake.";};
init();
</script></body></html>"""
