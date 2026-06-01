/* ════════════════════════════════════════════════════════
   Farmer Advisory System — main.js
   Modules: Upload/Scan · Ask AI · Camera · History
════════════════════════════════════════════════════════ */

// ── Animated stat counters ──────────────────────────────
document.querySelectorAll('.stat-n[data-target]').forEach(el => {
  const target = +el.dataset.target;
  let cur = 0;
  const step = Math.max(1, Math.floor(target / 45));
  const t = setInterval(() => {
    cur = Math.min(cur + step, target);
    el.textContent = cur;
    if (cur >= target) clearInterval(t);
  }, 28);
});

// ══════════════════════════════════════════════════════════
// SCAN PAGE
// ══════════════════════════════════════════════════════════
(function scanModule() {
  const dz         = document.getElementById('drop-zone');
  const fi         = document.getElementById('file-input');
  const prevWrap   = document.getElementById('prev-wrap');
  const prevImg    = document.getElementById('prev-img');
  const scanBtn    = document.getElementById('scan-btn');
  const resultWrap = document.getElementById('result-wrap');
  const alertEl    = document.getElementById('alert-box');
  const loader     = document.getElementById('loading-overlay');

  if (!dz) return;   // not on scan page

  let currentFile = null;

  /* ─ helpers ─ */
  function alert_(msg, type = 'err') {
    if (!alertEl) return;
    alertEl.textContent = msg;
    alertEl.className = `alert ${type} show`;
    setTimeout(() => alertEl.classList.remove('show'), 5000);
  }

  function setFile(file) {
    if (!file) return;
    const ok = ['image/jpeg','image/png','image/webp','image/jpg'];
    if (!ok.includes(file.type)) {
      alert_('Invalid file type. Use JPG, PNG, or WebP.'); return;
    }
    if (file.size > 16 * 1024 * 1024) {
      alert_('File too large — max 16 MB.'); return;
    }
    currentFile = file;
    const r = new FileReader();
    r.onload = e => {
      prevImg.src = e.target.result;
      dz.style.display = 'none';
      prevWrap.style.display = 'block';
      scanBtn.disabled = false;
      resultWrap.style.display = 'none';
    };
    r.readAsDataURL(file);
  }

  /* ─ events ─ */
  dz.addEventListener('click', () => fi.click());
  fi.addEventListener('change', e => setFile(e.target.files[0]));

  dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('over'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('over'));
  dz.addEventListener('drop', e => {
    e.preventDefault(); dz.classList.remove('over');
    setFile(e.dataTransfer.files[0]);
  });

  const changeBtn = document.getElementById('change-btn');
  if (changeBtn) changeBtn.addEventListener('click', () => {
    currentFile = null;
    prevWrap.style.display = 'none';
    dz.style.display = '';
    scanBtn.disabled = true;
    fi.value = '';
    resultWrap.style.display = 'none';
  });

  /* ─ scan ─ */
  scanBtn && scanBtn.addEventListener('click', async () => {
    // check for camera-captured file too
    const file = currentFile || window._camFile;
    if (!file) { alert_('Please select or capture an image first.', 'warn'); return; }

    loader && loader.classList.add('on');
    scanBtn.disabled = true;

    const fd = new FormData();
    fd.append('image', file);

    try {
      const res  = await fetch('/scan', { method: 'POST', body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Server error');
      renderResult(data);
    } catch (err) {
      alert_('Error: ' + err.message);
    } finally {
      loader && loader.classList.remove('on');
      scanBtn.disabled = false;
    }
  });

  /* ─ render result ─ */
  function renderResult(d) {
    resultWrap.style.display = 'block';
    resultWrap.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // model-not-loaded inline warning
    const mw = document.getElementById('model-warn-inline');
    if (mw) mw.style.display = d.status === 'model_not_loaded' ? 'flex' : 'none';

    setText('res-disease',  d.disease || 'Unknown');
    setText('res-desc',     d.description || '');

    const sev = document.getElementById('res-sev');
    if (sev) { sev.textContent = d.severity || 'Unknown'; sev.className = `sev-badge ${d.severity||'Unknown'}`; }

    // confidence bar
    const pct = d.confidence || 0;
    setText('conf-pct', pct.toFixed(1) + '%');
    const fill = document.getElementById('conf-fill');
    if (fill) {
      setTimeout(() => { fill.style.width = pct + '%'; }, 120);
      if      (pct >= 75) fill.style.background = 'linear-gradient(90deg,#2d6a4f,#52b788)';
      else if (pct >= 45) fill.style.background = 'linear-gradient(90deg,#b45309,#d97706)';
      else                fill.style.background = 'linear-gradient(90deg,#991b1b,#dc2626)';
    }

    setText('cnn-info', d.status === 'model_not_loaded'
      ? 'CNN (not loaded — run train_image_model.py --demo) • Input: 224×224 RGB'
      : 'CNN MobileNetV2 transfer learning • Input: 224×224 RGB');

    if (d.treatment) {
      setText('treat-fertilizer', d.treatment.fertilizer || '—');
      setText('treat-pesticide',  d.treatment.pesticide  || '—');
      setText('treat-irrigation', d.treatment.irrigation || '—');
      const ul = document.getElementById('treat-prevention');
      if (ul && Array.isArray(d.treatment.prevention)) {
        ul.innerHTML = d.treatment.prevention.map(p => `<li>${p}</li>`).join('');
      }
    }
  }

  function setText(id, v) {
    const el = document.getElementById(id);
    if (el) el.textContent = v;
  }
})();

// ══════════════════════════════════════════════════════════
// ASK AI PAGE
// ══════════════════════════════════════════════════════════
(function askModule() {
  const form    = document.getElementById('ask-form');
  const ta      = document.getElementById('query-ta');
  const counter = document.getElementById('char-ct');
  const aiCard  = document.getElementById('ai-card');
  const loader  = document.getElementById('loading-overlay');

  if (!ta) return;   // not on ask page

  ta.addEventListener('input', () => {
    if (counter) counter.textContent = ta.value.length + '/500';
  });

  // suggestion chips
  document.querySelectorAll('.chip').forEach(c => {
    c.addEventListener('click', () => {
      ta.value = c.dataset.q || c.textContent;
      ta.dispatchEvent(new Event('input'));
      ta.focus();
    });
  });

  form && form.addEventListener('submit', async e => {
    e.preventDefault();
    const query = ta.value.trim();
    if (query.length < 3) return;

    loader && loader.classList.add('on');

    try {
      const res  = await fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);
      renderAI(data);
    } catch (err) {
      alert('Error: ' + err.message);
    } finally {
      loader && loader.classList.remove('on');
    }
  });

  function renderAI(data) {
    if (!aiCard) return;
    aiCard.style.display = 'block';
    aiCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    const titleEl = document.getElementById('ai-title');
    const bodyEl  = document.getElementById('ai-body');
    if (titleEl) titleEl.textContent = data.title || 'Advisory';
    if (bodyEl) {
      let html = (data.response || '').replace(/\n/g, '<br>');
      html = html.replace(/\((\d+)\)/g, '<br><strong>($1)</strong>');
      bodyEl.innerHTML = html;
    }
  }
})();

// ══════════════════════════════════════════════════════════
// CAMERA MODULE
// ══════════════════════════════════════════════════════════
(function cameraModule() {
  const openBtn   = document.getElementById('cam-open');
  const closeBtn  = document.getElementById('cam-close');
  const captureBtn= document.getElementById('cam-capture');
  const video     = document.getElementById('cam-vid');
  const camPanel  = document.getElementById('cam-panel');
  const prevWrap  = document.getElementById('prev-wrap');
  const prevImg   = document.getElementById('prev-img');
  const dz        = document.getElementById('drop-zone');
  const scanBtn   = document.getElementById('scan-btn');

  if (!openBtn || !video) return;

  let stream = null;

  openBtn.addEventListener('click', async () => {
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: 640, height: 480 }
      });
      video.srcObject = stream;
      video.style.display = 'block';
      video.play();
      camPanel.style.display = 'block';
      openBtn.style.display = 'none';
    } catch (err) {
      alert('Camera not available: ' + err.message);
    }
  });

  closeBtn && closeBtn.addEventListener('click', stopCam);

  captureBtn && captureBtn.addEventListener('click', () => {
    const c = document.createElement('canvas');
    c.width = video.videoWidth || 640;
    c.height = video.videoHeight || 480;
    c.getContext('2d').drawImage(video, 0, 0);
    c.toBlob(blob => {
      window._camFile = new File([blob], 'capture.jpg', { type: 'image/jpeg' });
      const r = new FileReader();
      r.onload = ev => {
        if (prevImg) prevImg.src = ev.target.result;
        if (dz) dz.style.display = 'none';
        if (prevWrap) prevWrap.style.display = 'block';
        if (scanBtn) scanBtn.disabled = false;
      };
      r.readAsDataURL(window._camFile);
      stopCam();
    }, 'image/jpeg', 0.92);
  });

  function stopCam() {
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
    video.style.display = 'none';
    if (camPanel) camPanel.style.display = 'none';
    if (openBtn) openBtn.style.display = '';
  }
})();

// ══════════════════════════════════════════════════════════
// HISTORY: clear button
// ══════════════════════════════════════════════════════════
(function historyModule() {
  const btn = document.getElementById('clear-btn');
  if (!btn) return;
  btn.addEventListener('click', async () => {
    if (!confirm('Clear all history? This cannot be undone.')) return;
    const r = await fetch('/api/history/clear', { method: 'POST' });
    if (r.ok) location.reload();
  });
})();
