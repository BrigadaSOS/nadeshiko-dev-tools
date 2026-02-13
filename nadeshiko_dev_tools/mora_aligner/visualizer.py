"""Generate mora_viz.html karaoke-style visualization per episode.

Dual-mode layout per segment:
- Top: Karaoke text with mora highlighting synced to audio playback,
  accent H/L marks shown as a pitch pattern line above each word
- Bottom: F0 pitch graph with moving cursor, mora boundaries, accent tinting
"""
# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from nadeshiko_dev_tools.common.archive import discover_episodes

console = Console()

MORA_PITCH_FILE = "_mora_pitch.json"
DATA_FILE = "_data.json"
PITCH_FILE = "_pitch.json"


def _build_segment_data(
    data: dict, pitch_data: dict, mora_data: dict
) -> list[dict]:
    """Merge _data.json, _pitch.json, and _mora_pitch.json into visualization segments."""
    segments_list = data.get("segments", [])
    pitch_segments = pitch_data.get("segments", {})
    pitch_meta = pitch_data.get("metadata", {})
    default_sample_ms = pitch_meta.get("sample_ms", 10)
    mora_segments = mora_data.get("segments", {})

    viz_segments = []
    for seg in segments_list:
        seg_hash = seg.get("segment_hash", "")
        pitch_entry = pitch_segments.get(seg_hash)
        if not pitch_entry:
            continue

        f0 = pitch_entry.get("f0", [])
        sample_ms = pitch_entry.get("sample_ms", default_sample_ms)
        mora_entry = mora_segments.get(seg_hash)

        viz_segments.append({
            "hash": seg_hash,
            "index": seg.get("segment_index", 0),
            "start_ms": seg.get("start_ms", 0),
            "end_ms": seg.get("end_ms", 0),
            "duration_ms": seg.get("duration_ms", 0),
            "ja": seg.get("content_ja", ""),
            "en": seg.get("content_en", ""),
            "es": seg.get("content_es", ""),
            "f0": f0,
            "sample_ms": sample_ms,
            "mora": mora_entry,
        })

    return viz_segments


def generate_viz_html(episode_dir: Path) -> Path | None:
    """Generate mora_viz.html for one episode directory."""
    data_file = episode_dir / DATA_FILE
    pitch_file = episode_dir / PITCH_FILE
    mora_file = episode_dir / MORA_PITCH_FILE

    if not data_file.exists() or not pitch_file.exists() or not mora_file.exists():
        return None

    with open(data_file) as f:
        data = json.load(f)
    with open(pitch_file) as f:
        pitch_data = json.load(f)
    with open(mora_file) as f:
        mora_data = json.load(f)

    ep_meta = data.get("metadata", {})
    ep_num = ep_meta.get("number", episode_dir.name)
    total_segments = ep_meta.get("total_segments", 0)
    duration_s = ep_meta.get("duration_ms", 0) / 1000

    viz_segments = _build_segment_data(data, pitch_data, mora_data)
    segments_json = json.dumps(viz_segments, ensure_ascii=False, separators=(",", ":"))

    html = _HTML_TEMPLATE.replace("__EPISODE_NUM__", str(ep_num))
    html = html.replace("__TOTAL_SEGMENTS__", str(total_segments))
    html = html.replace("__DURATION_S__", f"{duration_s:.1f}")
    html = html.replace("__SEGMENTS_JSON__", segments_json)

    output_path = episode_dir / "mora_viz.html"
    with open(output_path, "w") as f:
        f.write(html)

    return output_path


def run_visualize(
    media_folder: Path,
    episode_num: int | None = None,
) -> None:
    """Generate mora visualization HTML for episodes."""
    console.print("[bold]Mora Visualization Generator[/bold]")
    console.print(f"Media: {media_folder}")
    console.print()

    all_episodes = discover_episodes(media_folder)
    if not all_episodes:
        console.print("[yellow]No episode folders found.[/yellow]")
        return

    if episode_num is not None:
        all_episodes = [ep for ep in all_episodes if ep.name == str(episode_num)]
        if not all_episodes:
            console.print(f"[red]Episode {episode_num} not found.[/red]")
            return

    generated = 0
    for ep_dir in all_episodes:
        result = generate_viz_html(ep_dir)
        if result:
            console.print(f"  ep {ep_dir.name}: [green]{result.name}[/green]")
            generated += 1
        else:
            console.print(
                f"  ep {ep_dir.name}: [yellow]skipped (missing data)[/yellow]"
            )

    console.print()
    console.print(f"[bold green]Generated {generated} visualization(s).[/bold green]")


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mora Pitch - Episode __EPISODE_NUM__</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:system-ui,-apple-system,sans-serif;background:#0a0a0f;color:#e0e0e0;padding:20px}
h1{margin-bottom:6px;font-size:1.3em;color:#fff}
.meta{color:#666;margin-bottom:16px;font-size:0.85em}

/* --- Segment card --- */
.seg{background:#111118;border:1px solid #1e1e2e;border-radius:10px;margin-bottom:14px;overflow:hidden;transition:border-color .2s}
.seg:hover{border-color:#3a3a5a}
.seg.active{border-color:#4a9eff;box-shadow:0 0 20px rgba(74,158,255,.08)}
.seg-head{display:flex;justify-content:space-between;align-items:center;padding:10px 16px;background:#0d0d14}
.seg-id{font-weight:600;color:#4a9eff;font-size:.8em;display:flex;align-items:center;gap:8px}
.seg-time{color:#555;font-size:.75em;font-family:monospace}

/* --- Karaoke text area --- */
.karaoke{padding:16px 20px 8px;min-height:80px;position:relative;cursor:pointer}
.karaoke-words{display:flex;flex-wrap:wrap;gap:2px 6px;align-items:flex-end;justify-content:center}
.k-word{display:inline-flex;flex-direction:column;align-items:center;position:relative}
.k-accent-line{height:20px;position:relative;width:100%;margin-bottom:2px}
.k-mora-row{display:flex;gap:0}
.k-mora{position:relative;padding:4px 2px;font-size:1.5em;color:#444;transition:color .15s;text-align:center;min-width:1.2em;line-height:1.2}
.k-mora.voiced{color:#e0e0e0}
.k-mora.active{color:#4a9eff;text-shadow:0 0 12px rgba(74,158,255,.5)}
.k-mora.done{color:#7ab8ff}
.k-accent-mark{position:absolute;top:-2px;left:50%;transform:translateX(-50%);font-size:.55em;font-weight:700;pointer-events:none}
.k-accent-mark.high{color:#4a9eff}
.k-accent-mark.low{color:#555}
.k-reading{font-size:.6em;color:#555;margin-top:1px;letter-spacing:.05em}
.k-en{text-align:center;color:#555;font-size:.8em;margin-top:8px;padding:0 20px}

/* --- Pitch graph --- */
.pitch-area{padding:4px 16px 10px;position:relative}
.pitch-area canvas{width:100%;height:80px;display:block;border-radius:4px}
.pitch-area .cursor-layer{position:absolute;top:4px;left:16px;right:16px;height:80px;pointer-events:none}

/* --- Play button --- */
.play-btn{background:#4a9eff;color:#fff;border:none;padding:5px 14px;border-radius:5px;cursor:pointer;font-size:.8em;transition:background .2s}
.play-btn:hover{background:#3a8eef}
.play-btn.playing{background:#ff6b6b}
.play-sm{background:none;border:1px solid #4a9eff;color:#4a9eff;padding:2px 8px;border-radius:4px;cursor:pointer;font-size:.75em;transition:all .2s}
.play-sm:hover{background:#4a9eff;color:#fff}
.play-sm.playing{border-color:#ff6b6b;color:#ff6b6b}

/* --- Detail panel --- */
.layout{display:flex;gap:16px}
.layout .main{flex:1;min-width:0}
.layout .side{width:360px;flex-shrink:0;position:sticky;top:16px;align-self:flex-start}
.detail{background:#111118;border:1px solid #1e1e2e;border-radius:10px;padding:16px}
.detail .ja{font-size:1.2em;margin-bottom:4px}
.detail .en{color:#888;font-size:.9em;margin-bottom:2px}
.detail .es{color:#666;font-size:.85em}
.detail-stats{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:10px}
.d-stat{background:#0d0d14;padding:6px 8px;border-radius:4px}
.d-stat .lbl{font-size:.65em;color:#555;text-transform:uppercase}
.d-stat .val{font-size:1em;color:#4a9eff}
.word-breakdown{margin-top:10px}
.wb-word{background:#0d0d14;border-radius:4px;padding:6px 8px;margin-bottom:4px}
.wb-head{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:3px}
.wb-surface{font-size:1em;color:#fff}
.wb-reading{font-size:.8em;color:#777;margin-left:6px}
.wb-pos{font-size:.65em;color:#555;background:#1a1a2a;padding:1px 5px;border-radius:3px}
.wb-mora{display:flex;gap:2px;flex-wrap:wrap}
.wb-m{padding:2px 5px;border-radius:3px;font-size:.75em;font-family:monospace}
.wb-m.h{background:rgba(74,158,255,.15);color:#6ab0ff;border:1px solid rgba(74,158,255,.25)}
.wb-m.l{background:rgba(80,80,80,.15);color:#888;border:1px solid rgba(80,80,80,.25)}
@media(max-width:900px){.layout{flex-direction:column}.layout .side{width:100%;position:static}}
</style>
</head>
<body>
<h1>Mora Pitch Karaoke</h1>
<p class="meta">Episode __EPISODE_NUM__ &middot; __TOTAL_SEGMENTS__ segments &middot; __DURATION_S__s</p>

<div class="layout">
<div class="main" id="segList"></div>
<div class="side">
  <div class="detail" id="detail">
    <div class="ja" style="color:#444">Click a segment</div>
  </div>
</div>
</div>

<script>
const S = __SEGMENTS_JSON__;
const dpr = window.devicePixelRatio || 1;
let curAudio = null, curBtn = null, curRaf = null, curSegIdx = -1;

// ── Helpers ──
function fmtTime(ms) {
  const s = Math.floor(ms/1000), m = Math.floor(s/60);
  return m+':'+String(s%60).padStart(2,'0')+'.'+String(ms%1000).padStart(3,'0');
}

function smoothF0(raw) {
  const f0 = raw.slice();
  let i = 0;
  while (i < f0.length) {
    if (f0[i] === 0) {
      let gs = i;
      while (i < f0.length && f0[i] === 0) i++;
      let gl = i - gs;
      if (gl <= 3 && gs > 0 && i < f0.length) {
        let sv = f0[gs-1], ev = f0[i];
        for (let j = 0; j < gl; j++) f0[gs+j] = sv + (ev-sv)*((j+1)/(gl+1));
      }
    } else i++;
  }
  const out = f0.slice();
  for (let i = 0; i < f0.length; i++) {
    if (f0[i] === 0) continue;
    let sum = 0, cnt = 0;
    for (let j = i-2; j <= i+2; j++)
      if (j >= 0 && j < f0.length && f0[j] > 0) { sum += f0[j]; cnt++; }
    if (cnt > 0) out[i] = sum / cnt;
  }
  return out;
}

// ── Flatten mora for a segment ──
function flatMora(seg) {
  if (!seg.mora || !seg.mora.words) return [];
  const flat = [];
  for (const w of seg.mora.words) {
    for (const m of w.mora) flat.push(m);
  }
  return flat;
}

// ── Draw F0 pitch graph with mora overlay ──
function drawPitch(canvas, seg) {
  const rawF0 = seg.f0, sampleMs = seg.sample_ms;
  const f0 = smoothF0(rawF0);
  const ctx = canvas.getContext('2d');
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr; canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;
  ctx.clearRect(0, 0, W, H);

  const voiced = f0.filter(v => v > 0);
  if (!voiced.length) {
    ctx.fillStyle = '#333'; ctx.font = '11px system-ui'; ctx.textAlign = 'center';
    ctx.fillText('No voiced frames', W/2, H/2); return;
  }

  const minF = Math.min(...voiced) * 0.9, maxF = Math.max(...voiced) * 1.1;
  const range = maxF - minF || 1;
  const totalMs = f0.length * sampleMs;

  // Mora background regions
  const fm = flatMora(seg);
  for (const m of fm) {
    const x1 = (m.start_ms / totalMs) * W, x2 = (m.end_ms / totalMs) * W;
    ctx.fillStyle = m.accent === 'H' ? 'rgba(74,158,255,.07)' : 'rgba(80,80,80,.04)';
    ctx.fillRect(x1, 0, x2-x1, H);
    // boundary
    ctx.strokeStyle = 'rgba(255,255,255,.08)'; ctx.lineWidth = .5;
    ctx.setLineDash([2,2]);
    ctx.beginPath(); ctx.moveTo(x2,0); ctx.lineTo(x2,H); ctx.stroke();
    ctx.setLineDash([]);
    // kana label at bottom
    const cx = (x1+x2)/2;
    ctx.fillStyle = m.accent === 'H' ? 'rgba(74,158,255,.5)' : 'rgba(150,150,150,.3)';
    ctx.font = '9px system-ui'; ctx.textAlign = 'center';
    ctx.fillText(m.kana, cx, H-3);
  }

  // Grid
  ctx.strokeStyle = '#1a1a2a'; ctx.lineWidth = .5;
  for (let i = 0; i <= 3; i++) {
    const y = (i/3)*H;
    ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke();
  }

  // F0 curve
  const runs = []; let run = null;
  for (let i = 0; i < f0.length; i++) {
    if (f0[i] > 0) { if (!run) run = []; run.push({i, v:f0[i]}); }
    else if (run) { runs.push(run); run = null; }
  }
  if (run) runs.push(run);

  ctx.strokeStyle = '#4a9eff'; ctx.lineWidth = 2;
  ctx.lineJoin = 'round'; ctx.lineCap = 'round';
  for (const run of runs) {
    if (run.length === 1) {
      const x = (run[0].i/(f0.length-1))*W, y = H-((run[0].v-minF)/range)*H;
      ctx.fillStyle = '#4a9eff'; ctx.beginPath(); ctx.arc(x,y,2,0,Math.PI*2); ctx.fill();
      continue;
    }
    const pts = run.map(p => ({x:(p.i/(f0.length-1))*W, y:H-((p.v-minF)/range)*H}));
    ctx.beginPath(); ctx.moveTo(pts[0].x, pts[0].y);
    for (let j = 0; j < pts.length-1; j++) {
      const p0=pts[Math.max(0,j-1)], p1=pts[j], p2=pts[Math.min(pts.length-1,j+1)], p3=pts[Math.min(pts.length-1,j+2)];
      const t=.2;
      ctx.bezierCurveTo(p1.x+(p2.x-p0.x)*t, p1.y+(p2.y-p0.y)*t, p2.x-(p3.x-p1.x)*t, p2.y-(p3.y-p1.y)*t, p2.x, p2.y);
    }
    ctx.stroke();
  }
}

// ── Draw accent pattern line above a word (SVG path in the accent-line div) ──
function buildAccentSVG(moraList) {
  if (!moraList.length) return '';
  const w = moraList.length * 28;
  const h = 20;
  const yH = 4, yL = 16;  // H = top, L = bottom
  let d = '';
  for (let i = 0; i < moraList.length; i++) {
    const x = i * 28 + 14;
    const y = moraList[i].accent === 'H' ? yH : yL;
    d += (i === 0 ? `M${x},${y}` : `L${x},${y}`);
  }
  return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" style="display:block;margin:0 auto">
    <path d="${d}" fill="none" stroke="#4a9eff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    ${moraList.map((m, i) => {
      const x = i * 28 + 14, y = m.accent === 'H' ? yH : yL;
      return `<circle cx="${x}" cy="${y}" r="3" fill="${m.accent === 'H' ? '#4a9eff' : '#555'}"/>`;
    }).join('')}
  </svg>`;
}

// ── Build karaoke HTML for a segment ──
function karaokeHTML(seg) {
  if (!seg.mora || !seg.mora.words) return `<div class="k-mora" style="font-size:1.2em;color:#666">${seg.ja || ''}</div>`;

  let moraGlobalIdx = 0;
  let html = '<div class="karaoke-words">';
  for (const word of seg.mora.words) {
    if (!word.mora.length) continue;
    html += '<div class="k-word">';
    // Accent line SVG
    html += buildAccentSVG(word.mora);
    // Mora characters
    html += '<div class="k-mora-row">';
    for (const m of word.mora) {
      html += `<span class="k-mora" data-midx="${moraGlobalIdx}" data-start="${m.start_ms}" data-end="${m.end_ms}">${m.kana}</span>`;
      moraGlobalIdx++;
    }
    html += '</div>';
    // Reading under word
    html += `<span class="k-reading">${word.surface}</span>`;
    html += '</div>';
  }
  html += '</div>';
  if (seg.en) html += `<div class="k-en">${seg.en}</div>`;
  return html;
}

// ── Audio & sync ──
function stopAudio() {
  if (curAudio) { curAudio.pause(); curAudio.src = ''; curAudio = null; }
  if (curRaf) { cancelAnimationFrame(curRaf); curRaf = null; }
  if (curBtn) { curBtn.textContent = curBtn.dataset.label; curBtn.classList.remove('playing'); curBtn = null; }
  // Reset mora highlights
  document.querySelectorAll('.k-mora.active,.k-mora.done').forEach(el => {
    el.classList.remove('active','done');
  });
  // Clear cursor canvases
  document.querySelectorAll('.cursor-layer').forEach(c => {
    const ctx = c.getContext('2d');
    ctx.clearRect(0, 0, c.width, c.height);
  });
  curSegIdx = -1;
}

function playAudio(hash, btn, segIdx) {
  if (curAudio && curBtn === btn) { stopAudio(); return; }
  stopAudio();

  const audio = new Audio(hash + '.mp3');
  curAudio = audio; curBtn = btn; curSegIdx = segIdx;
  btn.textContent = '\u25A0'; btn.classList.add('playing');
  audio.addEventListener('ended', stopAudio);

  const seg = S[segIdx];
  const totalMs = seg.f0.length * seg.sample_ms;
  const moraEls = document.querySelectorAll(`#seg-${segIdx} .k-mora`);
  const cursorCanvas = document.getElementById('cursor-' + segIdx);

  audio.play().then(() => {
    // Setup cursor canvas
    let cCtx, cW, cH;
    if (cursorCanvas) {
      cCtx = cursorCanvas.getContext('2d');
      const rect = cursorCanvas.getBoundingClientRect();
      cursorCanvas.width = rect.width * dpr; cursorCanvas.height = rect.height * dpr;
      cCtx.scale(dpr, dpr);
      cW = rect.width; cH = rect.height;
    }

    function frame() {
      if (audio.paused || audio.ended) return;
      const pct = audio.currentTime / audio.duration;
      const currentMs = pct * totalMs;

      // Highlight mora
      moraEls.forEach(el => {
        const st = +el.dataset.start, en = +el.dataset.end;
        el.classList.toggle('active', currentMs >= st && currentMs < en);
        el.classList.toggle('done', currentMs >= en);
      });

      // Draw cursor on pitch graph
      if (cCtx) {
        cCtx.clearRect(0, 0, cW, cH);
        const x = pct * cW;
        cCtx.strokeStyle = '#ff6b6b'; cCtx.lineWidth = 1.5;
        cCtx.beginPath(); cCtx.moveTo(x, 0); cCtx.lineTo(x, cH); cCtx.stroke();

        // Glow at cursor position
        const grad = cCtx.createLinearGradient(x-8, 0, x+8, 0);
        grad.addColorStop(0, 'rgba(255,107,107,0)');
        grad.addColorStop(.5, 'rgba(255,107,107,.15)');
        grad.addColorStop(1, 'rgba(255,107,107,0)');
        cCtx.fillStyle = grad;
        cCtx.fillRect(x-8, 0, 16, cH);
      }

      curRaf = requestAnimationFrame(frame);
    }
    frame();
  });
}

// ── Render segments ──
function render() {
  const list = document.getElementById('segList');
  let html = '';

  S.forEach((seg, i) => {
    const hasMora = seg.mora && seg.mora.words && seg.mora.words.length > 0;
    html += `
    <div class="seg" id="seg-${i}" onclick="select(${i})">
      <div class="seg-head">
        <span class="seg-id">#${seg.index}
          <button class="play-sm" data-label="\u25B6" onclick="event.stopPropagation();playAudio('${seg.hash}',this,${i})">\u25B6</button>
        </span>
        <span class="seg-time">${fmtTime(seg.start_ms)} \u2192 ${fmtTime(seg.end_ms)}</span>
      </div>
      <div class="karaoke" id="karaoke-${i}">
        ${hasMora ? karaokeHTML(seg) : `<div style="font-size:1.2em;color:#555;text-align:center;padding:8px">${seg.ja || '(no text)'}</div>`}
      </div>
      <div class="pitch-area">
        <canvas id="pitch-${i}"></canvas>
        <canvas id="cursor-${i}" class="cursor-layer"></canvas>
      </div>
    </div>`;
  });

  list.innerHTML = html;

  // Draw all pitch graphs
  S.forEach((seg, i) => {
    drawPitch(document.getElementById('pitch-'+i), seg);
  });
}

function select(i) {
  document.querySelectorAll('.seg').forEach(el => el.classList.remove('active'));
  document.getElementById('seg-'+i)?.classList.add('active');

  const seg = S[i];
  const voiced = seg.f0.filter(v => v > 0);
  const avg = voiced.length ? Math.round(voiced.reduce((a,b)=>a+b)/voiced.length) : 0;
  const mn = voiced.length ? Math.min(...voiced) : 0;
  const mx = voiced.length ? Math.max(...voiced) : 0;

  let wbHtml = '';
  if (seg.mora && seg.mora.words) {
    wbHtml = '<div class="word-breakdown">';
    for (const w of seg.mora.words) {
      if (!w.mora.length) continue;
      const chips = w.mora.map(m => {
        const cls = m.accent === 'H' ? 'h' : 'l';
        const hz = m.f0_mean !== null ? ` ${Math.round(m.f0_mean)}Hz` : '';
        return `<span class="wb-m ${cls}" title="${m.start_ms}-${m.end_ms}ms${hz}">${m.kana}<sup>${m.accent}</sup></span>`;
      }).join('');
      wbHtml += `<div class="wb-word">
        <div class="wb-head">
          <span><span class="wb-surface">${w.surface}</span><span class="wb-reading">${w.reading}</span></span>
          <span class="wb-pos">${w.pos}</span>
        </div>
        <div class="wb-mora">${chips}</div>
      </div>`;
    }
    wbHtml += '</div>';
  }

  document.getElementById('detail').innerHTML = `
    <div class="ja">${seg.ja||''}</div>
    <div class="en">${seg.en||''}</div>
    <div class="es">${seg.es||''}</div>
    <div style="margin-top:8px">
      <button class="play-btn" data-label="\u25B6 Play" onclick="playAudio('${seg.hash}',this,${i})">\u25B6 Play</button>
    </div>
    <div class="detail-stats">
      <div class="d-stat"><div class="lbl">Duration</div><div class="val">${(seg.duration_ms/1000).toFixed(2)}s</div></div>
      <div class="d-stat"><div class="lbl">Avg F0</div><div class="val">${avg} Hz</div></div>
      <div class="d-stat"><div class="lbl">Range</div><div class="val">${mn}-${mx} Hz</div></div>
      <div class="d-stat"><div class="lbl">Voiced</div><div class="val">${voiced.length}/${seg.f0.length}</div></div>
    </div>
    ${wbHtml}`;
}

render();
window.addEventListener('resize', () => {
  S.forEach((seg, i) => {
    const c = document.getElementById('pitch-'+i);
    if (c) drawPitch(c, seg);
  });
});
</script>
</body>
</html>
"""
