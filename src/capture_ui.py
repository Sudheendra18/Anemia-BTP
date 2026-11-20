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
        lines.append((f"Red tissue coverage: {cand['coverage'] * 100:.0f}%", (255, 255, 255)))
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


def _png_b64(img):
    """Native-size, lossless crop preview; display scaling stays in the browser."""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Could not encode crop preview")
    return base64.b64encode(buf).decode()


HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Eye Capture</title>
<style>
:root{color-scheme:dark;--ink:#ececea;--muted:#a1a3a8;--line:#303236;--paper:#18191b;--surface:#222427;--hover:#2b2d31;--accent:#dadcde;--accent-ink:#151618;--wash:#101112;--warning:#dfb86c;--mono:Consolas,"SFMono-Regular",monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--wash);color:var(--ink);font:14px/1.5 "Segoe UI",Arial,sans-serif;-webkit-font-smoothing:antialiased;scrollbar-color:#46494f var(--wash)}
button,input,select{font:inherit}
button{display:inline-flex;justify-content:center;align-items:center;gap:9px;min-height:40px;border:1px solid #44474d;border-radius:4px;padding:9px 14px;background:var(--surface);color:var(--ink);font-size:13px;font-weight:600;cursor:pointer;transition:background .15s,border-color .15s}
button:hover:not(:disabled){background:var(--hover);border-color:#646870}button:disabled{opacity:.45;cursor:default}
.primary{background:var(--accent);border-color:var(--accent);color:var(--accent-ink)}.primary:hover:not(:disabled){background:#f1f2f3;border-color:#f1f2f3}
.secondary{background:transparent}
button:focus-visible,input:focus-visible,select:focus-visible,summary:focus-visible{outline:2px solid #9cb4e1;outline-offset:3px}
.shell{width:100%;max-width:calc(max(320px,100dvh - 210px)*var(--camera-aspect,1.777778) + 308px);margin:0 auto;padding:16px}
.workbench{display:grid;grid-template-columns:minmax(0,1fr) 260px;grid-template-rows:max-content 1fr;gap:16px;align-items:start}
.sidebar,.camera-card,#saved{background:var(--paper);border:1px solid var(--line);border-radius:6px}
.sidebar{grid-column:2;grid-row:1;padding:20px}
h2{font-size:14px;font-weight:650;letter-spacing:-.15px;margin:0 0 18px}
.section-label{display:block;color:var(--muted);font-size:10px;font-weight:650;letter-spacing:1.1px;text-transform:uppercase;margin:0 0 14px}
.setup{display:grid;gap:15px}.field{display:flex;flex-direction:column;gap:6px;min-width:0}
label{font-size:12px;font-weight:550;color:#cacbd0}
input,select{width:100%;min-width:0;min-height:40px;background:var(--surface);color:var(--ink);border:1px solid #44474d;border-radius:4px;padding:9px 10px;font-size:13px}
input::placeholder{color:#8e929a}select option{background:var(--surface);color:var(--ink)}
select{padding-right:24px;text-overflow:ellipsis}input[readonly]{background:#131416;color:#c0c3ca;font-family:var(--mono);font-size:13px;letter-spacing:.4px}
input:disabled,select:disabled{opacity:.55}
small,.hint{color:var(--muted);font-size:11px;line-height:1.6}
summary{cursor:pointer;list-style:none;font-size:12px;font-weight:550;display:flex;align-items:center;justify-content:space-between;gap:8px;min-height:28px}
summary::-webkit-details-marker{display:none}summary::after{content:'+';font:16px var(--mono);color:var(--muted)}details[open]>summary::after{content:'−'}
details .setup{margin-top:14px}details{margin-top:18px}
.sidebar-section{padding-top:22px;margin-top:22px;border-top:1px solid var(--line)}
#saved{grid-column:2;grid-row:2;padding:20px 22px;font-size:11px;color:var(--muted);overflow-wrap:anywhere}
.savehead{display:flex;align-items:center;justify-content:space-between;gap:8px}.savehead strong{color:var(--ink);font-size:13px;font-weight:600}
#openFolder{font-size:11px;min-height:30px;padding:5px 8px;gap:5px}#openFolder svg{width:13px;height:13px}
#outputPath,#csvPath{margin-top:8px;font:10px/1.6 var(--mono)}#lastSave{margin-top:12px;line-height:1.7;color:var(--muted)}
#saved details{border-top:1px solid var(--line);padding-top:12px;margin-top:14px}#saved summary{font-size:11px;font-weight:400}
.camera-card{grid-column:1;grid-row:1 / span 2;min-width:0;overflow:hidden}
.view-header{display:flex;justify-content:space-between;align-items:center;gap:16px;padding:12px 16px;border-bottom:1px solid var(--line)}
.view-heading{display:flex;align-items:center;gap:9px;min-width:0}.view-heading svg{width:17px;height:17px;color:var(--muted);flex-shrink:0}.view-header h2{margin:0;font-size:13px;font-weight:600}
#cameraQuality{font:10px/1.4 var(--mono);color:var(--muted);text-align:right;max-width:55%;overflow-wrap:anywhere}
.viewer-grid,.viewer{min-width:0}.viewer{background:#090a0b}
#stage{position:relative;width:100%;aspect-ratio:var(--preview-ratio,16/9);background:#090a0b;overflow:hidden}
video,img#rev{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}
canvas#guide{position:absolute;inset:0;width:100%;height:100%;touch-action:none;cursor:crosshair}
#selection{position:absolute;border:1px solid #a6c0ff;pointer-events:none;background:#a6c0ff22}
#count{position:absolute;inset:0;display:grid;place-items:center;pointer-events:none;font:500 68px var(--mono);color:white;text-shadow:0 2px 20px #0008}
.guidehelp{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:8px 16px;background:#141517;border-bottom:1px solid var(--line)}
.guidehelp .hint{max-width:520px}.guidehelp button{flex-shrink:0;min-height:30px;font-size:11px;padding:4px 9px;color:#c2c4ca;border-color:#3b3e44;background:var(--surface)}
body[data-stage=Review] .guidehelp{display:none}

.capture-footer{display:flex;justify-content:space-between;align-items:center;gap:20px;padding:14px 16px;min-height:80px}
.status-block{min-width:0;flex:1}#msg{font-size:13px;font-weight:600;line-height:1.6}#info{font-size:11px;color:var(--muted);margin-top:3px;line-height:1.6;font-variant-numeric:tabular-nums}
#setupWarning{margin-top:6px;overflow-wrap:anywhere}
#btns{display:flex;gap:8px;align-items:center;flex-shrink:0}#go{min-width:150px}#go::before{content:'';width:12px;height:12px;border:2px solid currentColor;border-radius:50%}#keep{order:2}#cancel{color:#eea39b}
#reviewArea{min-width:0;display:grid;grid-template-columns:minmax(0,1.35fr) minmax(230px,.8fr);border-bottom:1px solid var(--line);background:#141517}
#reviewArea:not(:has(#cropbox:not(.hide),#prediction:not(.hide))){display:none}
#cropbox{display:grid;grid-template-columns:150px minmax(0,1fr);column-gap:16px;row-gap:12px;padding:20px;min-width:0}
#cropbox>div:first-child{grid-column:1/-1;font-size:12px;font-weight:600}
#crop{display:block;width:100%;height:136px;object-fit:contain;background:#090a0b;border:1px solid var(--line);border-radius:4px;align-self:start}
.roi-controls{min-width:0}#cropSize{color:var(--muted);font:10px/1.5 var(--mono);margin:0 0 8px}
.roi-tools{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px}#cropbox button{font-size:11px;min-height:30px;padding:5px 8px}#cropbox #confirmRoi{width:100%;min-height:36px;line-height:1.5}
#prediction{align-self:stretch;padding:20px;border-left:1px solid var(--line);font-size:12px;color:#c0c3ca;line-height:1.7;overflow-wrap:anywhere}
#reviewArea:not(:has(#cropbox:not(.hide))){grid-template-columns:1fr}#reviewArea:not(:has(#cropbox:not(.hide))) #prediction{border-left:0}
.result-heading{display:block;color:var(--muted);font-size:10px;letter-spacing:.8px;text-transform:uppercase}
.result-label{display:block;color:var(--ink);font-size:26px;font-weight:650;letter-spacing:-.8px;line-height:1.2;margin:12px 0}
.result-score{font:12px/1.6 var(--mono);color:#c6c8ce}.result-note{font-size:11px;color:var(--muted);margin:12px 0 0;line-height:1.6}
#prediction[data-status=uncertain] .result-label,#prediction[data-status=review_only] .result-label{color:var(--warning)}
#prediction[data-status=running]{color:var(--accent)}
.candidate-review{border-top:1px solid var(--line);padding:12px 20px;margin:0}.candidate-review summary{font-size:12px;color:#c0c3ca;font-weight:500}
#cands{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:14px 0 8px}
.cand{border:1px solid var(--line);background:var(--paper);border-radius:4px;padding:7px;font-size:10px;line-height:1.7;color:var(--muted);min-width:0}.cand.best{border-color:#a3a7af}
.cand img{display:block;width:100%;border-radius:2px;margin-bottom:8px}.cand strong{display:block;font-size:11px;font-weight:600;color:var(--ink);margin-bottom:4px}
dialog{width:min(94vw,1160px);max-height:92vh;background:var(--paper);color:var(--ink);border:1px solid #44474d;border-radius:6px;padding:24px;box-shadow:0 18px 60px #0008}
dialog::backdrop{background:#000b}dialog strong{font-size:15px;font-weight:600}dialog p{color:var(--muted);font-size:13px}
.detailControls{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}.detailScroll{overflow:auto;max-height:65vh;background:#090a0b;border-radius:4px}#detailImage{display:block;width:auto;height:auto;max-width:none}
#detailClose{margin-left:auto}#outlineApply{background:var(--accent);border-color:var(--accent);color:var(--accent-ink)}#outlineCanvas{margin:auto;border-radius:4px}#outlineStatus{font:12px var(--mono)}
#viewModelInput{margin-top:12px}
.hide{display:none!important}

@media(max-width:1200px){.shell{padding:12px}.workbench{grid-template-columns:minmax(0,1fr) 250px;gap:12px}.sidebar{padding:20px}#saved{padding:18px 20px}#reviewArea{grid-template-columns:minmax(0,1fr) minmax(200px,.65fr)}#cropbox{grid-template-columns:110px minmax(0,1fr);gap:12px;padding:18px}#crop{height:125px}#prediction{padding:18px}.capture-footer{gap:12px}#btns{align-self:center}#cands{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:1000px){.shell{padding:12px}.workbench{grid-template-columns:minmax(0,1fr) 230px;gap:12px}.sidebar{padding:18px}#saved{padding:18px}#reviewArea{grid-template-columns:1fr}#prediction{border-left:0;border-top:1px solid var(--line)}#cropbox{grid-template-columns:150px minmax(0,1fr)}.view-header{padding:14px 16px;align-items:flex-start}#cameraQuality{font-size:9px}.guidehelp{padding:10px 16px;align-items:flex-start;gap:8px}.capture-footer{padding:16px}}
@media(max-width:900px){.shell{max-width:none;padding:12px}.workbench{display:flex;flex-direction:column;gap:16px}.sidebar{order:1;width:100%;padding:18px}.sidebar h2{margin-bottom:14px}.sidebar #captureForm>.setup{grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.sidebar .sidebar-section{margin-top:18px;padding-top:16px}.sidebar-section>.setup{grid-template-columns:minmax(0,1fr) 120px;gap:12px}.sidebar-section .section-label{display:none}#captureOptions{margin-top:12px}#captureOptions .setup{grid-template-columns:repeat(2,minmax(0,1fr))}#captureOptions .field:first-child{grid-column:1/-1}.camera-card{order:0;width:100%}#saved{order:2;width:100%;padding:18px}.capture-footer{min-height:0}.view-header{padding:13px 14px}.view-header h2{font-size:12px}#cameraQuality{max-width:52%;font-size:9px}.guidehelp{padding:10px 14px}.guidehelp .hint{font-size:10px}.capture-footer{padding:16px 14px;flex-direction:column;align-items:stretch}#btns{width:100%;justify-content:flex-end}#btns button{flex:1;min-width:0}#cropbox{padding:16px 14px;grid-template-columns:110px minmax(0,1fr)}#crop{height:125px}#prediction{padding:18px 14px}.candidate-review{padding:10px 14px}#cands{gap:8px}dialog{padding:18px}#detailTitle{display:block;font-size:13px}.detailControls button{font-size:12px}}
@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style></head><body data-stage="Capture"><main class="shell">
<div class="workbench">
<aside class="sidebar" aria-label="Capture settings"><form id="captureForm">
<h2>Session details</h2>
<div class="setup"><div class="field"><label for="patientMode">Patient</label><select id="patientMode"><option value="new">New patient</option><option value="same" disabled>Same patient as last save</option><option value="existing">Existing patient</option></select></div>
<div class="field"><label for="patientId">Patient ID</label><input id="patientId" readonly aria-describedby="patientHelp" list="patientIds" maxlength="80" pattern="[A-Za-z0-9](?:[A-Za-z0-9_]|-)*"><datalist id="patientIds"></datalist><small id="patientHelp">Assigned when you save.</small></div></div>
<section class="sidebar-section" aria-label="Camera settings"><span class="section-label">Camera setup</span><div class="setup">
<div class="field"><label for="cameraSelect">Camera</label><select id="cameraSelect"><option value="">Default camera</option></select></div>
<div class="field"><label for="duration">Recording length</label><select id="duration"><option value="3">3 seconds</option><option value="5">5 seconds</option><option value="8">8 seconds</option></select></div></div>
<details id="captureOptions"><summary>Optional capture details</summary><div class="setup">
<div class="field"><label for="deviceLabel">Device label</label><input id="deviceLabel" maxlength="80" pattern="[A-Za-z0-9](?:[A-Za-z0-9_]|-)*" placeholder="webcam"></div>
<div class="field"><label for="distanceLabel">Distance</label><select id="distanceLabel"><option value="unspecified">Not recorded</option><option value="5cm">5 cm</option><option value="10cm">10 cm</option><option value="15cm">15 cm</option><option value="20cm">20 cm</option><option value="30cm">30 cm</option></select></div>
<div class="field"><label for="lightingLabel">Lighting</label><select id="lightingLabel"><option value="unspecified">Not recorded</option><option value="indoor">Indoor</option><option value="daylight">Daylight</option><option value="diffuse">Diffuse light</option><option value="flash">Flash</option></select></div></div></details></section></form></aside>
<section class="camera-card" aria-label="Camera and review">
<div class="view-header"><div class="view-heading"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><rect x="3" y="6" width="18" height="14" rx="2"/><path d="m8 6 1.5-3h5L16 6"/><circle cx="12" cy="13" r="3.5"/></svg><h2 id="viewTitle">Camera preview</h2></div><div id="cameraQuality"></div></div>
<div class="viewer-grid"><div class="viewer">
<div id="stage"><video id="cam" autoplay muted playsinline></video><img id="rev" class="hide" alt="Selected full-resolution frame"><canvas id="guide" aria-label="Drag to select the exposed lower eyelid"></canvas><div id="selection" class="hide"></div><div id="count" aria-live="polite"></div></div>
<div class="guidehelp"><span class="hint">Expose the lower eyelid and hold still. Drag over the tissue to set the guide manually.</span><button id="auto" class="secondary">Reset guide</button></div>
</div>
<div id="reviewArea"><div id="cropbox" class="hide"><div>Extracted tissue</div><img id="crop" alt="Conjunctiva extraction preview"><div class="roi-controls"><div id="cropSize"></div><div class="roi-tools"><button id="inspect" class="secondary">View original pixels</button><button id="editRoi" class="secondary">Edit tissue outline</button></div><button id="confirmRoi" class="primary">Confirm ROI &amp; predict anemia</button></div></div><div id="prediction" class="hide" role="status" aria-live="polite"></div></div></div>
<div class="capture-footer"><div class="status-block"><div id="msg" role="status">Starting camera…</div><div id="info">Allow camera access if asked.</div><div id="setupWarning" class="hint hide" role="status"></div></div><div id="btns"><button id="go" class="primary hide">Start capture</button><button id="cancel" class="secondary hide">Cancel capture</button><button id="disc" class="secondary hide">Retake</button><button id="keep" class="primary hide">Save capture</button><button id="retryCamera" class="secondary hide">Retry camera</button></div></div>
<details id="candidateReview" class="candidate-review hide"><summary>Compare selected frames</summary><div id="cands" class="hide"></div></details>
</section>
<section id="saved" aria-label="Save location"><div class="savehead"><strong>Saved captures</strong><button id="openFolder" class="secondary"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><path d="M3 7V5h6l2 2h10v13H3V7Z"/></svg>Open folder</button></div><div id="lastSave" role="status">No captures saved this session.</div><details><summary>Storage location</summary><div id="outputPath"></div><div id="csvPath"></div></details></section>
</div></main>
<dialog id="detail" aria-labelledby="detailTitle"><strong id="detailTitle">Original pixels</strong><div class="detailControls"><button id="detailEye" class="secondary">Eye crop</button><button id="detailRoi" class="secondary">Tissue ROI</button><button id="detailModelInput" class="secondary hide">Model input</button><button id="detailClose" class="primary">Close</button></div><div class="detailScroll"><img id="detailImage" alt="Original-size crop, one image pixel per CSS pixel"></div></dialog>
<dialog id="outline" aria-labelledby="outlineTitle" aria-describedby="outlineHelp"><strong id="outlineTitle">Outline the exposed lower-lid tissue</strong><p id="outlineHelp">Click around its boundary, including pale areas. Exclude the eyeball, skin and fingers. Use at least 3 points.</p><canvas id="outlineCanvas" style="display:block;max-width:100%;max-height:60vh;touch-action:none;cursor:crosshair"></canvas><div class="detailControls"><button id="outlineUndo">Undo point</button><button id="outlineApply">Apply outline</button><button id="outlineClose">Cancel</button></div><p id="outlineStatus" role="status"></p></dialog>
<script>
const $=id=>document.getElementById(id);
const DURATION=%DURATION%;
const OUTPUT=%OUTPUT_JSON%;
const DETAILS=%DETAILS_JSON%;
if(DETAILS.setup_warning){
  $('setupWarning').textContent=DETAILS.setup_warning;
  $('setupWarning').classList.remove('hide');
}
function setSelect(id,value){
  const select=$(id);if(![...select.options].some(o=>o.value===String(value))) select.add(new Option(String(value),String(value)));
  select.value=String(value);
}
let lastPatientId='',nextPatientId=DETAILS.next_patient_id;
const patientIds=new Set(DETAILS.patient_ids||[]);
function updatePatientList(){
  $('patientIds').replaceChildren();
  for(const id of [...patientIds].sort()) $('patientIds').appendChild(new Option(id,id));
}
function showPatient(){
  const mode=$('patientMode').value;
  $('patientId').setCustomValidity('');
  $('patientId').readOnly=mode!=='existing';
  $('patientId').required=mode==='existing';
  $('patientId').value=mode==='new'?nextPatientId:mode==='same'?lastPatientId:(DETAILS.patient_id||'');
  $('patientHelp').textContent=mode==='new'?'Assigned when you save.':mode==='same'?'Using the last saved patient. Verify this is the same person.':'Select a previously saved patient ID.';
}
$('patientMode').onchange=showPatient;
$('patientId').oninput=()=>$('patientId').setCustomValidity('');
if(DETAILS.patient_id) $('patientMode').value='existing';
updatePatientList();showPatient();$('deviceLabel').value=DETAILS.device;
setSelect('distanceLabel',DETAILS.distance);setSelect('lightingLabel',DETAILS.lighting);setSelect('duration',DURATION);
$('duration').selectedOptions[0].textContent=DURATION+' seconds';
$('captureForm').onsubmit=e=>e.preventDefault();
function setStage(stage){
  document.body.dataset.stage=stage;
  $('viewTitle').textContent=stage==='Review'?'Capture review':'Camera preview';
  updatePreviewRatio();
}
function lockDetails(locked){document.querySelectorAll('#captureForm input,#captureForm select').forEach(e=>e.disabled=locked);}
function captureDetails(){
  const mode=$('patientMode').value;
  const patient=mode==='new'?'':mode==='same'?lastPatientId:$('patientId').value.trim();
  return {patient_id:patient,patient_mode:mode==='new'?'new':patient===DETAILS.patient_id?'':'existing',device:$('deviceLabel').value.trim(),distance:$('distanceLabel').value,lighting:$('lightingLabel').value};
}
$('outputPath').textContent='Save folder: '+OUTPUT;
$('csvPath').textContent='CSV index: '+OUTPUT+'/dataset.csv';
const OVAL_RX=%OVAL_RX%, OVAL_RY=%OVAL_RY%;
const SMOOTH=0.55; // Higher values follow the detected eye faster.
let stream, tracking=false, cur=null;
let selecting=null, recording=false;
let initializing=false, activeRecording=null, reviewBusy=false;
let recordedCameraSettings=null;
let regionBusy=false, trackingGeneration=0;
let reviewImages=null;
let lastFrameAt=0, frameWatch=0, frameStalled=false, cameraMuted=false;
let loopStarted=false;
const grab=document.createElement('canvas'), gctx=grab.getContext('2d');
const sleep=ms=>new Promise(r=>setTimeout(r,ms));

function updatePreviewRatio(){
  const image=document.body.dataset.stage==='Review'&&$('rev').naturalWidth?$('rev'):$('cam');
  const width=image.videoWidth||image.naturalWidth,height=image.videoHeight||image.naturalHeight;
  if(!width||!height) return;
  document.documentElement.style.setProperty('--preview-ratio',width+'/'+height);
  document.documentElement.style.setProperty('--camera-aspect',width/height);
}
$('cam').addEventListener('loadedmetadata',updatePreviewRatio);
$('cam').addEventListener('resize',updatePreviewRatio);
$('rev').addEventListener('load',updatePreviewRatio);
function frameGeometry(){
  const rect=$('guide').getBoundingClientRect(),v=$('cam');
  const w=v.videoWidth||16,h=v.videoHeight||9,s=Math.min(rect.width/w,rect.height/h);
  return {w,h,s,ox:(rect.width-w*s)/2,oy:(rect.height-h*s)/2};
}
function pointerPosition(e){
  const rect=$('guide').getBoundingClientRect(),{w,h,s,ox,oy}=frameGeometry();
  return {x:Math.max(ox,Math.min(ox+w*s,e.clientX-rect.left)),y:Math.max(oy,Math.min(oy+h*s,e.clientY-rect.top))};
}
$('guide').onpointerdown=e=>{
  if(recording || regionBusy || !tracking) return;
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
  const {w:vw,h:vh,s,ox,oy}=frameGeometry();
  const x=(Math.min(start.x,p.x)-ox)/s/vw,y=(Math.min(start.y,p.y)-oy)/s/vh;
  const w=Math.abs(start.x-p.x)/s/vw,h=Math.abs(start.y-p.y)/s/vh;
  if(w<.02 || h<.02) return;
  await applyRegion([x,y,w,h]);
};
$('guide').onpointercancel=()=>{selecting=null;$('selection').classList.add('hide');};
async function applyRegion(region){
  if(recording || initializing || regionBusy || !cameraIsLive()) return;
  regionBusy=true;trackingGeneration++;lockDetails(true);$('go').disabled=true;$('auto').disabled=true;
  try{
    const result=await window.pyRegion(region);if(result.error) throw new Error(result.error);
    cur=null;
    $('msg').textContent=region?'Selected region: hold the exposed conjunctiva inside the guide.':'Automatic eye detection enabled.';
  }catch(e){$('msg').textContent='Could not change eye region: '+e.message;}
  finally{regionBusy=false;lockDetails(false);$('go').disabled=false;$('auto').disabled=false;}
}
$('auto').onclick=()=>applyRegion(null);

async function refreshCameras(){
  const previous=$('cameraSelect').value;
  const devices=(await navigator.mediaDevices.enumerateDevices()).filter(d=>d.kind==='videoinput');
  $('cameraSelect').replaceChildren(...devices.map((d,i)=>new Option(d.label||'Camera '+(i+1),d.deviceId)));
  const present=devices.some(d=>d.deviceId===previous);
  if(present) $('cameraSelect').value=previous;
  else $('cameraSelect').value='';
  return {missing:!!previous&&!present};
}
navigator.mediaDevices.addEventListener('devicechange',()=>{
  if(!recording&&!initializing&&!regionBusy) refreshCameras().catch(()=>{});
});

async function init(){
  if(initializing || recording || regionBusy) return;
  trackingGeneration++;
  initializing=true;lockDetails(true);
  $('retryCamera').classList.add('hide');$('go').classList.add('hide');tracking=false;
  try{
    if(stream) stream.getTracks().forEach(t=>{t.onended=null;t.stop();});
    const previousCamera=$('cameraSelect').value;
    const available=await refreshCameras();
    const selected=$('cameraSelect').value;
    const videoConstraints={width:{ideal:1920},height:{ideal:1080},frameRate:{ideal:30,max:30},...(selected?{deviceId:{exact:selected}}:{facingMode:'environment'})};
    stream=await navigator.mediaDevices.getUserMedia({video:videoConstraints,audio:false});
    const cameraTrack=stream.getVideoTracks()[0];
    cameraMuted=false;frameStalled=false;lastFrameAt=performance.now();
    const watching=++frameWatch;
    function observeFrame(){
      if(watching!==frameWatch) return;
      lastFrameAt=performance.now();
      if(frameStalled){frameStalled=false;if(!recording&&!initializing) readyControls();}
      $('cam').requestVideoFrameCallback(observeFrame);
    }
    if($('cam').requestVideoFrameCallback) $('cam').requestVideoFrameCallback(observeFrame);
    cameraTrack.onmute=()=>{
      if(watching!==frameWatch) return;
      cameraMuted=true;tracking=false;cur=null;trackingGeneration++;
      if(activeRecording) activeRecording.abort(new Error('Camera paused during recording. Unlock your phone and reconnect'));
      else if(!recording){readyControls();$('msg').textContent='Camera paused. Unlock your phone or retry the camera.';}
    };
    cameraTrack.onunmute=()=>{if(watching===frameWatch){cameraMuted=false;lastFrameAt=performance.now();if(!recording&&!initializing) readyControls();}};
    cameraTrack.onended=()=>{
      tracking=false;cur=null;trackingGeneration++;
      if(activeRecording) activeRecording.abort(new Error('Camera disconnected during recording'));
      else if(!recording){
        readyControls();$('msg').textContent='Camera disconnected. Reconnect it, then click Retry camera.';
      }
    };
    const capabilities=cameraTrack.getCapabilities ? cameraTrack.getCapabilities() : {};
    if(capabilities.focusMode && capabilities.focusMode.includes('continuous')){
      try{
        const current=cameraTrack.getConstraints();
        await cameraTrack.applyConstraints({...videoConstraints,...current,advanced:[...(current.advanced||[]),{focusMode:'continuous'}]});
      }catch(e){}
    }
    $('cam').srcObject=stream;
    await $('cam').play();
    $('cameraQuality').textContent=(cameraTrack.label||'Camera')+' · '+$('cam').videoWidth+' × '+$('cam').videoHeight;
    const devices=await navigator.mediaDevices.enumerateDevices();
    const actual=cameraTrack.getSettings().deviceId;
    $('cameraSelect').replaceChildren(...devices.filter(d=>d.kind==='videoinput').map((d,i)=>new Option(d.label||'Camera '+(i+1),d.deviceId)));
    if(actual) $('cameraSelect').value=actual;
    const resetRegion=await window.pyRegion(null);if(resetRegion.error) throw new Error(resetRegion.error);
    cur=null;
    $('msg').textContent=(available.missing&&previousCamera?'Previous camera unavailable. Using '+(cameraTrack.label||'the default camera')+'. ':'')+'Ready. Bring the eye close and expose the lower lid.';
    $('go').classList.remove('hide');
    tracking=true;if(!loopStarted){loopStarted=true;trackLoop();}
  }catch(e){
    if(stream) stream.getTracks().forEach(t=>{t.onended=null;t.stop();});
    $('msg').textContent='Camera unavailable. Check that it is connected and not in use by another app.';$('info').textContent=e.message;$('retryCamera').classList.remove('hide');
  }finally{initializing=false;lockDetails(false);}
}
$('cameraSelect').onchange=init;
$('retryCamera').onclick=init;
$('openFolder').onclick=async()=>{try{const r=await window.pyOpenFolder();if(r.error) throw new Error(r.error);}catch(e){$('lastSave').textContent='Could not open the folder: '+e.message;}};
function toB64(blob){return new Promise((resolve,reject)=>{const f=new FileReader();f.onload=()=>resolve(f.result.split(',')[1]);f.onerror=()=>reject(f.error||new Error('Could not read recording'));f.onabort=()=>reject(new Error('Reading recording was interrupted'));f.readAsDataURL(blob);});}
function cameraIsLive(){return !cameraMuted && !frameStalled && !!stream && stream.getVideoTracks().some(t=>t.readyState==='live'&&!t.muted);}
setInterval(()=>{
  if(!stream||initializing||!$('cam').requestVideoFrameCallback||!stream.getVideoTracks().some(t=>t.readyState==='live')) return;
  if(performance.now()-lastFrameAt>2500 && !frameStalled){
    frameStalled=true;tracking=false;cur=null;trackingGeneration++;
    if(activeRecording) activeRecording.abort(new Error('Camera stopped delivering frames. Unlock your phone and retry'));
    else if(!recording){readyControls();$('msg').textContent='Camera stopped delivering frames. Unlock your phone or retry.';}
  }
},500);
function readyControls(){
  const live=cameraIsLive();tracking=live&&!recording;
  $('go').classList.toggle('hide',!live||recording);$('retryCamera').classList.toggle('hide',live||recording);
}
function recordClip(duration){
  return new Promise((resolve,reject)=>{
    const chunks=[];
    const settings=stream.getVideoTracks()[0].getSettings();
    const bitrate=Math.round(Math.max(8000000,Math.min(24000000,(settings.width||1280)*(settings.height||720)*(settings.frameRate||30)*.28)));
    const rec=new MediaRecorder(stream,{mimeType:'video/webm;codecs=vp8',videoBitsPerSecond:bitrate});
    recordedCameraSettings={camera_name:stream.getVideoTracks()[0].label,width:settings.width,height:settings.height,frame_rate:settings.frameRate,requested_duration_seconds:duration,video_bits_per_second:rec.videoBitsPerSecond,mime_type:rec.mimeType};
    let interval,deadline,stopTimeout,finished=false,requestedStop=false;
    function finish(error){
      if(finished) return;
      finished=true;clearInterval(interval);clearTimeout(deadline);clearTimeout(stopTimeout);
      rec.ondataavailable=rec.onerror=rec.onstop=null;
      try{if(rec.state!=='inactive') rec.stop();}catch(e){}
      activeRecording=null;$('count').textContent='';$('cancel').classList.add('hide');
      if(error) reject(error);
      else if(!chunks.length) reject(new Error('The camera returned an empty recording'));
      else resolve(new Blob(chunks,{type:'video/webm'}));
    }
    activeRecording={abort:finish};
    rec.ondataavailable=e=>{if(e.data.size) chunks.push(e.data);};
    rec.onerror=e=>finish(e.error||new Error('Camera recording failed'));
    rec.onstop=()=>finish(requestedStop?null:new Error('Camera recording stopped unexpectedly'));
    try{rec.start();}catch(e){finish(e);return;}
    $('cancel').classList.remove('hide');let remaining=duration;$('count').textContent=remaining;
    interval=setInterval(()=>{$('count').textContent=Math.max(0,--remaining)||'';},1000);
    deadline=setTimeout(()=>{
      requestedStop=true;$('cancel').classList.add('hide');
      stopTimeout=setTimeout(()=>finish(new Error('Camera did not finish the recording')),5000);
      try{rec.stop();}catch(e){finish(e);}
    },duration*1000);
  });
}
$('cancel').onclick=()=>{
  const error=new Error('Capture cancelled');error.cancelled=true;
  if(activeRecording) activeRecording.abort(error);
};

function drawGuide(){
  const c=$('guide'), st=$('stage'), W=st.clientWidth, H=st.clientHeight;
  if(!W||!H) return;
  if(c.width!==W||c.height!==H){c.width=W;c.height=H;}
  const g=c.getContext('2d'); g.clearRect(0,0,W,H);
  const {w:vw,h:vh,s,ox,oy}=frameGeometry(); // Match the full frame shown with object-fit:contain.
  g.lineWidth=3;
  g.beginPath();
  if(cur){
    g.setLineDash([]); g.strokeStyle=cur.ok?'#00e676':'#ffca28';
    g.ellipse(cur.cx*s+ox,cur.cy*s+oy,cur.rx*s,cur.ry*s,cur.ang,0,2*Math.PI);
  }else{
    g.setLineDash([10,6]); g.strokeStyle='rgba(255,255,255,.55)';
    g.ellipse(ox+vw*s/2,oy+vh*s/2,OVAL_RX*vw*s,OVAL_RY*vh*s,0,0,2*Math.PI);
  }
  g.stroke();
}
async function trackLoop(){
  const v=$('cam');
  while(true){
    if(!tracking||regionBusy||!v.videoWidth){ if(v.videoWidth&&$('rev').classList.contains('hide')) drawGuide(); await sleep(100); continue; }
    const tw=v.videoWidth, th=v.videoHeight;
    if(grab.width!==tw) grab.width=tw;
    if(grab.height!==th) grab.height=th;
    gctx.drawImage(v,0,0,tw,th);
    let r=null;const generation=trackingGeneration;
    try{ r=await window.pyTrack(grab.toDataURL('image/jpeg',0.95).split(',')[1]); }catch(e){}
    if(!tracking || regionBusy || generation!==trackingGeneration || tw!==v.videoWidth || th!==v.videoHeight) continue;
    if(r&&r.found){
      const sx=tw/r.w, sy=th/r.h;
      const t={cx:r.cx*sx,cy:r.cy*sy,rx:r.rx*sx,ry:r.ry*sy,ang:r.angle*Math.PI/180,ok:r.ok};
      if(cur&&Math.hypot(t.cx-cur.cx,t.cy-cur.cy)<cur.rx*1.5){
        for(const q of ['cx','cy','rx','ry','ang']) cur[q]+=(t[q]-cur[q])*SMOOTH;
        cur.ok=t.ok;
      }else cur=t;
    }else cur=null;
    if(r && r.feedback) $('info').textContent=r.feedback;
    drawGuide();
    await sleep(150); // One request in flight; leave CPU headroom for recording.
  }
}

function showCands(list, thumbnails){
  const box=$('cands'); box.innerHTML='';
  list.forEach(c=>{
    const d=document.createElement('div'); d.className='cand'+(c.best?' best':'');
    const im=document.createElement('img'); im.src='data:image/jpeg;base64,'+thumbnails[c.thumbnail];
    const t=document.createElement('div'),title=document.createElement('strong');
    const name={overall:'Overall quality',sharpness:'Best focus',exposure:'Best exposure',glare:'Least glare'}[c.name]||'Best tissue';
    title.textContent=(c.best?'Selected · ':'')+name;
    im.alt=title.textContent;
    t.textContent=`Focus ${c.sharpness.toFixed(0)} · Light ${c.brightness.toFixed(0)} · Glare ${(c.glare*100).toFixed(1)}%`;
    d.appendChild(im);d.appendChild(title);d.appendChild(t);box.appendChild(d);
  });
  box.classList.remove('hide');
  $('candidateReview').classList.remove('hide');
}

function showDetail(kind){
  if(!reviewImages || !reviewImages[kind]) return;
  $('detailImage').src='data:image/png;base64,'+reviewImages[kind];
  const label={roi:'Tissue ROI',crop:'Eye crop',model_input:'Model input'}[kind];
  $('detailTitle').textContent=label+' · '+reviewImages[kind+'_size'].join(' × ')+' pixels · 1:1 view';
}
$('inspect').onclick=()=>{if(!reviewImages) return;showDetail(reviewImages.roi?'roi':'crop');$('detail').showModal();};
$('detailEye').onclick=()=>showDetail('crop');
$('detailRoi').onclick=()=>showDetail('roi');
$('detailModelInput').onclick=()=>showDetail('model_input');
$('detailClose').onclick=()=>$('detail').close();

function showPrediction(prediction){
  const panel=$('prediction');
  panel.classList.remove('hide');panel.dataset.status=prediction?.status||'unavailable';
  panel.replaceChildren();
  if(['ok','uncertain','review_only'].includes(prediction?.status)){
    const heading=document.createElement('span'),label=document.createElement('strong');
    const score=document.createElement('div'),note=document.createElement('p');
    heading.className='result-heading';heading.textContent='Model prediction';
    label.className='result-label';label.textContent=prediction.label;
    score.className='result-score';score.textContent='Anemia score '+(prediction.anemia_score*100).toFixed(1)+'%';
    const warnings=['Research result only.'];
    if(prediction.status==='review_only') warnings.push(prediction.reason);
    if(prediction.status==='uncertain'||prediction.folds_disagree) warnings.push('The fold models disagree on the label.');
    note.className='result-note';note.textContent=warnings.filter(Boolean).join(' ');
    panel.append(heading,label,score,note);
    if(reviewImages?.model_input){
      const input=document.createElement('button');
      input.className='secondary';
      input.id='viewModelInput';
      input.textContent='View model input';
      input.onclick=()=>{showDetail('model_input');$('detail').showModal();};
      panel.append(input);
    }
  }else{
    panel.textContent=prediction?.reason||'Prediction unavailable.';
  }
}
function updateRoiPreview({crop, crop_size, roi, roi_size, prediction}){
  reviewImages={crop:crop??reviewImages?.crop, crop_size:crop_size??reviewImages?.crop_size, roi, roi_size};
  $('detailModelInput').classList.add('hide');
  $('crop').src='data:image/png;base64,'+(reviewImages.roi||reviewImages.crop);
  $('cropSize').textContent=(reviewImages.roi?reviewImages.roi_size:reviewImages.crop_size).join(' × ')+' original pixels · Lossless PNG';
  $('detailRoi').disabled=!reviewImages.roi;
  $('cropbox').firstElementChild.textContent=reviewImages.roi?'Extracted conjunctiva ROI':'Eye crop — ROI extraction failed';
  $('confirmRoi').disabled=!reviewImages.roi;
  $('confirmRoi').textContent=prediction.status==='disabled'?'Confirm ROI':'Confirm ROI & predict anemia';
  $('keep').textContent='Save for review';
  showPrediction(prediction);
}
let outlinePoints=[],outlineImage=new Image();
function drawOutline(){
  const canvas=$('outlineCanvas'),ctx=canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(outlineImage,0,0,canvas.width,canvas.height);
  if(outlinePoints.length){ctx.beginPath();outlinePoints.forEach((p,i)=>i?ctx.lineTo(p[0]*canvas.width,p[1]*canvas.height):ctx.moveTo(p[0]*canvas.width,p[1]*canvas.height));ctx.closePath();ctx.fillStyle='#3be5b433';ctx.fill();ctx.strokeStyle='#3be5b4';ctx.lineWidth=2;ctx.stroke();}
  $('outlineStatus').textContent=outlinePoints.length+' points selected';
}
$('editRoi').onclick=()=>{
  if(reviewBusy||!reviewImages?.crop) return;
  outlinePoints=[];outlineImage.onload=()=>{const c=$('outlineCanvas');const scale=Math.min(1,900/outlineImage.width);c.width=Math.round(outlineImage.width*scale);c.height=Math.round(outlineImage.height*scale);drawOutline();};
  outlineImage.src='data:image/png;base64,'+reviewImages.crop;$('outline').showModal();
};
$('outlineCanvas').onpointerdown=e=>{if(reviewBusy||outlinePoints.length>=200) return;const r=e.target.getBoundingClientRect();outlinePoints.push([Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))]);drawOutline();};
$('outlineUndo').onclick=()=>{if(!reviewBusy){outlinePoints.pop();drawOutline();}};
$('outlineClose').onclick=()=>{if(!reviewBusy) $('outline').close();};
$('outline').oncancel=e=>{if(reviewBusy)e.preventDefault();};
$('outlineApply').onclick=async()=>{
  if(reviewBusy) return;lockReview(true);
  try{const r=await window.pyRefine(outlinePoints);if(r.error)throw new Error(r.error);updateRoiPreview(r);$('outline').close();$('msg').textContent=r.crop_ok?'Check the corrected mask before confirming.':'Quality check: '+r.feedback;}
  catch(e){$('outlineStatus').textContent=e.message;}
  finally{lockReview(false);}
};
$('confirmRoi').onclick=async()=>{
  if(reviewBusy||!reviewImages?.roi) return;lockReview(true);
  if($('prediction').dataset.status!=='disabled'){
    showPrediction({status:'running',reason:'Running model…'});
  }
  try{
    const r=await window.pyConfirmRoi();
    if(r.error) throw new Error(r.error);
    const status=r.prediction.status;
    reviewImages.model_input=['ok','uncertain','review_only'].includes(status)
      ? (r.prediction.input_format==='masked'?reviewImages.roi:r.model_input) : null;
    reviewImages.model_input_size=reviewImages.roi_size;
    $('detailModelInput').classList.toggle('hide',!reviewImages.model_input);
    showPrediction(r.prediction);
    $('confirmRoi').textContent=status==='disabled'?'Confirm ROI':status==='unavailable'?'Retry prediction':'Run prediction again';
    $('keep').textContent=r.quality_passed?'Save approved capture':'Save for review';
    $('msg').textContent=status==='disabled'?'ROI confirmed. Inference is disabled.':
      status==='unavailable'?'ROI confirmed. No prediction was generated.':
      'ROI confirmed. Review the model result before saving.';
  }catch(e){showPrediction({status:'unavailable',reason:e.message});}
  finally{lockReview(false);}
};

$('go').onclick=async()=>{
  if(recording || initializing || regionBusy || !cameraIsLive()) return;
  const details=captureDetails(),duration=Number($('duration').value);
  $('patientId').setCustomValidity('');
  if(!$('captureForm').reportValidity()) return;
  lockDetails(true);setStage('Capture');
  recording=true; $('auto').disabled=true;
  $('go').classList.add('hide'); $('info').textContent='';
  try{
  if(details.patient_mode==='existing' && !patientIds.has(details.patient_id)){
    const r=await window.pyPatientIds();
    if(r.error) throw new Error('Could not refresh patient IDs. '+r.error);
    patientIds.clear();
    for(const id of r.ids) patientIds.add(id);
    updatePatientList();
    if(!patientIds.has(details.patient_id)){
      reset();
      $('patientId').setCustomValidity('Choose a patient ID from the saved list.');
      $('captureForm').reportValidity();
      return;
    }
    if(!cameraIsLive()) throw new Error('Camera disconnected before recording');
  }
  $('msg').textContent="Hold steady...";
  const clip=await recordClip(duration);
  tracking=false;
  $('msg').textContent="Analysing frames...";
  const res=await window.pyProcess(await toB64(clip),details,{...recordedCameraSettings,recording_bytes:clip.size});
  if(res.error) throw new Error(res.error);
  setStage('Review');
  $('rev').src='data:image/jpeg;base64,'+res.image; $('rev').classList.remove('hide');
  $('cam').classList.add('hide'); $('guide').classList.add('hide');
  if(res.crop){
    updateRoiPreview(res);
    $('cropbox').classList.remove('hide');
  }else{
    showPrediction(res.prediction);
  }
  showCands(res.cands, res.thumbnails);
  let m;
  if(!res.crop) m="Warning: no eye could be found in any frame. Retake?";
  else if(res.roi_status!=='ok') m='ROI extraction failed: '+res.roi_reason+'. Move closer or select the exposed lower lid.';
  else if(!res.crop_ok) m="Quality check: "+res.feedback+". Review or retake.";
  else m='Review the tissue preview before confirming the ROI or saving.';
  $('msg').textContent=m;
  let txt=`Focus ${res.sharpness.toFixed(0)} · Brightness ${res.brightness.toFixed(0)} · Glare ${(res.glare*100).toFixed(1)}%`;
  if(res.crop) txt+=res.eye_method==='manual_region'?' · Selected area':' · Eye detected';
  $('info').textContent=txt;
  $('keep').textContent='Save for review';
  $('keep').classList.remove('hide'); $('disc').classList.remove('hide');
  }catch(e){
    reset();$('msg').textContent=e.cancelled?'Capture cancelled. Ready to try again.':'Capture failed: '+e.message+'. '+(cameraIsLive()?'Please try again.':'Reconnect the camera, then click Retry camera.');
  }
};
function reset(){
  $('prediction').classList.add('hide');$('prediction').textContent='';
  $('candidateReview').classList.add('hide');$('candidateReview').open=false;
  reviewImages=null;
  $('detail').close();$('outline').close();
  for(const id of ['rev','crop','detailImage']) $(id).removeAttribute('src');
  $('cands').replaceChildren();
  outlineImage.onload=null;outlineImage.removeAttribute('src');
  // Resetting the backing store releases the previous crop and outline.
  $('outlineCanvas').width=300;
  $('outlineCanvas').height=150;
  outlinePoints=[];
  lockDetails(false);setStage('Capture');
  recording=false; $('auto').disabled=false;
  $('rev').classList.add('hide'); $('cam').classList.remove('hide'); $('guide').classList.remove('hide');
  $('cropbox').classList.add('hide'); $('cands').classList.add('hide');
  $('keep').classList.add('hide'); $('disc').classList.add('hide');$('cancel').classList.add('hide');
  cur=null; readyControls();
}
function lockReview(locked){reviewBusy=locked;for(const id of ['keep','disc','editRoi','outlineApply','outlineUndo','outlineClose'])$(id).disabled=locked;$('confirmRoi').disabled=locked||!reviewImages?.roi;}
$('keep').onclick=async()=>{
  if(reviewBusy) return;
  lockReview(true);$('msg').textContent='Saving capture...';
  try {
    const r=await window.pyKeep(); if(r.error) throw new Error(r.error);
    lastPatientId=r.patient_id;nextPatientId=r.next_patient_id;
    patientIds.add(r.patient_id);updatePatientList();
    $('patientMode').querySelector('option[value=same]').disabled=false;
    $('patientMode').value='new';
    showPatient();reset();setStage('Save');$('msg').textContent='Saved: '+r.image_id+' · Patient '+r.patient_id+'. '+(cameraIsLive()?'Ready for the next capture.':'Reconnect the camera, then click Retry camera.');
    $('csvPath').textContent='CSV index: '+r.csv_path;
    $('msg').textContent+=' '+(r.capture_status==='approved_for_testing'?'Quality passed and tissue confirmed.':'Saved for review—not approved for testing.');
    $('lastSave').textContent='Last capture: '+r.image_id+' · '+r.patient_id+' | 3 best frames saved'+(r.roi_image_path?' + cropped eye + ROI.':r.eye_image_path?' + cropped eye | ROI extraction failed.':' | No eye detected: crop and ROI unavailable.');
    $('lastSave').title=r.capture_dir;
    if(r.prediction_label){
      const notes=[];
      if(r.prediction_status==='review_only') notes.push('review only');
      if(r.folds_disagree) notes.push('fold models disagree');
      const qualifier=notes.length?' ('+notes.join(', ')+')':'';
      $('lastSave').textContent+=' Prediction saved: '+r.prediction_label+
        ' · Score '+(Number(r.anemia_score)*100).toFixed(1)+'%'+qualifier+'.';
    }else{
      const reason={
        needs_review:'ROI confirmation pending.',
        skipped:'Inference skipped.',
        disabled:'Inference disabled.',
        unavailable:'Prediction unavailable.',
        stale:'Prediction invalidated.',
      }[r.prediction_status]||'No prediction run.';
      $('lastSave').textContent+=' '+reason;
    }
  } catch(e) { $('msg').textContent='Save failed: '+e.message+'. Your capture is still available; retry Save or Retake.'; }
  finally { lockReview(false); }
};
$('disc').onclick=async()=>{
  if(reviewBusy) return;
  lockReview(true);
  try{
    const result=await window.pyDiscard();if(result&&result.error) throw new Error(result.error);
    reset();$('msg').textContent=cameraIsLive()?'Discarded. Ready to retake.':'Discarded. Reconnect the camera, then click Retry camera.';
  }catch(e){$('msg').textContent='Could not discard: '+e.message+'. Please try again.';}
  finally{lockReview(false);}
};
init();
</script></body></html>"""
