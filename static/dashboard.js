"use strict";
const readJSON = id => JSON.parse(document.getElementById(id).textContent);
const report = readJSON("report");
const features = readJSON("features");
const byId = id => document.getElementById(id);
const colors = {cart: "#8a929b", forest: "#e5e3de", xgb: "#c9a978"};
const fmt = new Intl.NumberFormat();
const percent = value => (100 * value).toFixed(2) + "%";

function canvasContext(canvas) {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  const ctx = canvas.getContext("2d");
  ctx.scale(scale, scale);
  return {ctx, width, height};
}

function drawSignal() {
  const {ctx, width, height} = canvasContext(byId("signal"));
  const x = width * .5;
  const y = height * .5;
  const r = Math.min(height * .365, width * .4);
  ctx.strokeStyle = "#202020";
  ctx.lineWidth = 1;
  for (let i = x % 24; i < width; i += 24) {
    ctx.beginPath(); ctx.moveTo(i, 38); ctx.lineTo(i, height - 38); ctx.stroke();
  }
  for (let i = 38; i < height - 37; i += 24) {
    ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(width, i); ctx.stroke();
  }
  ctx.strokeStyle = "#36302a";
  ctx.setLineDash([2, 5]);
  for (const radius of [r + 9, r + 24]) {
    ctx.beginPath(); ctx.arc(x, y, radius, 0, Math.PI * 2); ctx.stroke();
  }
  ctx.setLineDash([]);
  ctx.strokeStyle = "#504637";
  ctx.beginPath(); ctx.moveTo(x - r - 35, y); ctx.lineTo(x + r + 35, y); ctx.stroke();
  ctx.fillStyle = "#0b0b0b";
  ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();

  for (let lat = -89; lat <= 89; lat += 2.05) {
    const a = lat * Math.PI / 180;
    const samples = Math.round(171 * Math.cos(a)) + 1;
    for (let j = 0; j < samples; j++) {
      const b = (j / samples - .5) * Math.PI * 2;
      const depth = Math.cos(a) * Math.cos(b);
      const px = Math.cos(a) * Math.sin(b);
      const py = Math.sin(a);
      if (depth <= 0) continue;
      const nx = .95 * px + .312 * py;
      const ny = -.312 * px + .95 * py;
      const wave = Math.cos(b * 5 + Math.sin(a * 8)) + .75 * Math.sin(a * 8 - b * 4)
        + .34 * Math.sin(b * 22 + a * 19);
      const lit = wave > .26;
      const sweep = Math.abs(ny + .08) < .028;
      const opacity = lit ? .25 + .52 * depth : .1 + .11 * depth;
      ctx.fillStyle = sweep ? "rgba(201,169,120,.9)" : "rgba(241,233,219," + opacity + ")";
      ctx.beginPath();
      ctx.arc(x + r * nx, y + r * ny, (lit ? 1.07 : .65) + depth * .38, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.strokeStyle = "#827158";
  ctx.lineWidth = .7;
  ctx.beginPath();
  ctx.ellipse(x, y, r * 1.22, r * .36, -.25, .16, Math.PI + .38);
  ctx.stroke();
  ctx.fillStyle = "#d3b68d";
  for (const [dx, dy] of [[-.87, .26], [.69, -.48], [.23, .53]]) {
    ctx.beginPath(); ctx.arc(x + dx * r, y + dy * r, 2.6, 0, Math.PI * 2); ctx.fill();
    ctx.strokeRect(x + dx * r - 6, y + dy * r - 6, 12, 12);
  }
}

function line(ctx, x1, y1, x2, y2) {
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
}

function drawPR() {
  const {ctx, width, height} = canvasContext(byId("pr-chart"));
  const left = 50;
  const top = 12;
  const w = width - left - 12;
  const h = height - top - 45;
  if (w <= 0) return;
  ctx.font = "10px Consolas, monospace";
  ctx.textBaseline = "middle";
  for (let i = 0; i <= 4; i++) {
    const value = i / 4;
    const x = left + w * value;
    const y = top + h * (1 - value);
    ctx.strokeStyle = "#2d2d2d";
    ctx.fillStyle = "#9c9c9c";
    line(ctx, left, y, left + w, y);
    line(ctx, x, top, x, top + h);
    ctx.textAlign = "right";
    ctx.fillText(value.toFixed(2), left - 7, y);
    ctx.textAlign = "center";
    ctx.fillText(value.toFixed(2), x, top + h + 12);
  }
  ctx.fillText("Recall", left + w / 2, height - 8);
  ctx.save();
  ctx.translate(10, top + h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Precision", 0, 0);
  ctx.restore();

  const prevalence = report.partitions.test.fraud / report.partitions.test.rows;
  ctx.strokeStyle = "#77716a";
  ctx.setLineDash([4, 4]);
  line(ctx, left, top + h * (1 - prevalence), left + w, top + h * (1 - prevalence));
  ctx.setLineDash([]);
  for (const [id, model] of Object.entries(report.models)) {
    const x = recall => left + w * recall;
    const y = precision => top + h * (1 - precision);
    ctx.strokeStyle = colors[id];
    ctx.lineWidth = id === report.recommended ? 2.2 : 1.4;
    ctx.beginPath();
    model.pr.forEach(([r, p], i) => {
      if (i === 0) ctx.moveTo(x(r), y(p));
      else {
        ctx.lineTo(x(r), y(model.pr[i - 1][1]));
        ctx.lineTo(x(r), y(p));
      }
    });
    ctx.stroke();
  }
}

function populateLegend() {
  const legend = byId("legend");
  for (const [id, model] of Object.entries(report.models)) {
    const item = document.createElement("span");
    const swatch = document.createElement("i");
    swatch.style.backgroundColor = colors[id];
    item.append(swatch, model.name);
    legend.append(item);
  }
  const baseline = document.createElement("span");
  baseline.textContent = "— Prevalence baseline";
  legend.append(baseline);
}

function updateMatrix() {
  const id = byId("matrix-model").value;
  const model = report.models[id];
  const labels = ["True negatives", "False alerts", "Missed fraud", "Caught fraud"];
  byId("matrix").innerHTML = model.matrix.flat().map((count, index) =>
    '<div class="matrix-cell ' + (index === 0 || index === 3 ? "correct" : "") + '"><span>'
    + labels[index] + "</span><strong>" + fmt.format(count) + "</strong></div>"
  ).join("");
  byId("importance-method").textContent = id === "xgb"
    ? "Relative average split gain within XGBoost."
    : "Relative impurity reduction within this classifier.";
  byId("importance").innerHTML = model.importance.map(f =>
    "<li>" + f.name + " · " + percent(f.value) + "</li>"
  ).join("");
}

function loadExample() {
  const caseStudy = report.examples[byId("example").value];
  if (caseStudy) {
    for (const feature of features) byId("field-" + feature).value = caseStudy.values[feature];
  }
}

function showResult(result, truth, modelName) {
  const ex = result.explanation;
  const top = ex.features.slice(0, 6);
  const max = Math.max(...top.map(f => Math.abs(f.value)), 1e-9);
  const rows = top.map(f => {
    const direction = f.value < 0 ? "minus" : "";
    const signed = (f.value >= 0 ? "+" : "") + f.value.toFixed(3);
    return '<li><span>' + f.name + '</span><span class="factor-track"><i class="factor-fill '
      + direction + '" style="width:' + percent(Math.abs(f.value) / max) + '"></i></span>'
      + '<span class="factor-value">' + signed + "</span></li>";
  }).join("");
  byId("result").innerHTML = '<div class="result-status">' + (result.flagged ? "Flag for review" : "Below alert cutoff")
    + '</div><div class="result-score">' + percent(result.score) + '</div><p class="result-info">'
    + modelName + " score · cutoff " + percent(result.threshold) + '</p><div class="score-track" aria-hidden="true">'
    + '<span class="score-fill" style="width:' + percent(result.score) + '"></span>'
    + '<span class="score-cut" style="left:' + percent(result.threshold) + '"></span></div>'
    + '<p class="result-info">' + (truth === undefined ? "Edited or custom transaction." : "Recorded label: " + (truth ? "fraud." : "legitimate."))
    + ' Scores are uncalibrated.</p><h3 class="explanation-head">Largest score drivers</h3><p class="result-info">'
    + ex.method + " · " + ex.unit + " · baseline " + ex.base.toFixed(3)
    + ". Positive values move the score up. Negative values move it down.</p>"
    + (rows ? '<ul class="factors">' + rows + "</ul>" : '<p class="result-info">No nonzero contributions for this input.</p>');
}

const form = byId("score-form");
const submit = form.querySelector("button");
let sequence = 0;
function markStale() {
  sequence += 1;
  submit.disabled = false;
  byId("error").textContent = "";
  byId("result").innerHTML = "<p>Run the assessment to update the result.</p>";
}

form.addEventListener("input", event => {
  if (event.target.matches("input")) byId("example").value = "custom";
  markStale();
});
form.addEventListener("invalid", event => {
  if (event.target.closest(".feature-details")) {
    form.querySelector(".feature-details").open = true;
  }
}, true);
byId("example").addEventListener("change", loadExample);
byId("matrix-model").addEventListener("change", updateMatrix);
form.addEventListener("submit", async event => {
  event.preventDefault();
  const ticket = ++sequence;
  const model = byId("score-model").value;
  const values = Object.fromEntries(features.map(name => [name, byId("field-" + name).valueAsNumber]));
  const truth = report.examples[byId("example").value]?.truth;
  submit.disabled = true;
  byId("error").textContent = "";
  byId("result").innerHTML = "<p>Scoring transaction…</p>";
  try {
    const response = await fetch("/api/score", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({model, values})
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Unable to score this transaction.");
    if (ticket === sequence) showResult(result, truth, report.models[model].name);
  } catch (error) {
    if (ticket === sequence) {
      byId("error").textContent = error.message;
      byId("result").innerHTML = "<p>Assessment unavailable. Review the inputs and try again.</p>";
    }
  } finally {
    if (ticket === sequence) submit.disabled = false;
  }
});

populateLegend();
updateMatrix();
loadExample();
new ResizeObserver(drawPR).observe(byId("pr-chart"));
new ResizeObserver(drawSignal).observe(byId("signal"));

