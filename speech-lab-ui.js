/* Parliamentary speech teaching lab, runs fully in the browser. */
(() => {
"use strict";
const E = id => document.getElementById(id);
const engine = window.SpeechEngine;
if (!engine) { E("dataStatus").textContent = "Text preprocessing library failed to load."; return; }
const labels = ["Labour", "Conservative"];
const palette = { train: "#0f766e", validation: "#d97706" };
let sample = null;
let model = null;
let vocabulary = null;
let split = null;
let layers = [16];
let epochsHistory = [];
let lastPoint = null;
let busy = false;
let metric = "accuracy";
let heldOutChoice = null;
let selectedEpoch = null;

const textStatus = message => { E("trainingStatus").textContent = message; };
const formatPercent = number => Number.isFinite(number) ? (number * 100).toFixed(1) + "%" : "—";
const addNode = (parent, tag, options = {}) => {
  const el = document.createElement(tag);
  for (const [key,value] of Object.entries(options)) {
    if (key === "text") el.textContent = value;
    else if (key === "class") el.className = value;
    else if (key === "value") el.value = value;
    else el.setAttribute(key, String(value));
  }
  parent.appendChild(el);
  return el;
};
const setBusy = value => {
  busy = value;
  E("train").disabled = value || !sample;
  E("addLayer").disabled = value || layers.length >= 4;
  E("activation").disabled = value;
  E("epochs").disabled = value;
  E("earlyStop").disabled = value;
  E("vocabulary").disabled = value;
  E("trainingSet").disabled = value;
  E("localData").disabled = value;
  E("randomSpeech").disabled = value || !model;
  E("predict").disabled = value || !model;
  document.querySelectorAll(".layerSelect,.removeLayer").forEach(el => el.disabled = value || (el.classList.contains("removeLayer") && layers.length === 1));
};

function resetModel() {
  if (model) model.dispose();
  model = null; vocabulary = null; split = null; lastPoint = null; heldOutChoice = null; selectedEpoch=null;
  epochsHistory = [];
  ["trainAccuracy","validationAccuracy","testAccuracy"].forEach(id => E(id).textContent = "—");
  E("splitInfo").textContent = "A majority-class baseline is useful for checking whether the network is actually learning.";
  E("baselineInfo").textContent = "Train both models to see how they perform on new MPs.";
  E("evaluationReport").replaceChildren();
  E("vocabularyReport").replaceChildren();
  E("chartArea").innerHTML = '<p class="py-20 text-center text-sm text-slate-400">Train a network to display its learning curves.</p>';
  E("predictionResult").replaceChildren();
  E("selectedSpeechMeta").textContent = "";
  E("randomSpeech").disabled = true;
  E("predict").disabled = true;
}
function newConfig() {
  if (busy) return;
  resetModel();
  textStatus("Architecture changed. Train a new model to compare results.");
  drawDiagram();
  setBusy(false);
}
function layersUI() {
  const parent = E("layerControls");
  parent.replaceChildren();
  layers.forEach((units, i) => {
    const wrapper = addNode(parent, "div", { class: "flex items-center gap-2 rounded-xl bg-slate-50 p-3" });
    addNode(wrapper, "label", { class: "flex-1 text-sm font-semibold", for: "hidden-" + i, text: "Layer " + (i+1) });
    const select = addNode(wrapper, "select", { id: "hidden-" + i, class: "layerSelect rounded-lg border border-slate-300 p-2" });
    [4,8,10,16,32,64,128].forEach(n => addNode(select, "option", { value: n, text: n + " neurons" }));
    select.value = String(units);
    select.addEventListener("change", () => { layers[i] = Number(select.value); newConfig(); });
    const remove = addNode(wrapper, "button", { type: "button", class: "removeLayer text-slate-600 rounded px-2 py-1 hover:bg-slate-200", "aria-label": "Remove layer " + (i + 1), text: "×" });
    remove.disabled = busy || layers.length === 1;
    remove.addEventListener("click", () => {
      if (busy || layers.length <= 1) return;
      layers.splice(i,1); layersUI(); newConfig();
    });
  });
  E("layerCount").textContent = layers.length + " of 4";
  E("addLayer").disabled = busy || layers.length >= 4;
  drawDiagram();
}

function drawDiagram() {
  const counts = [vocabulary ? vocabulary.words.length : Number(E("vocabulary").value), ...layers, 2];
  const host = E("diagram");
  host.replaceChildren();
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 700 196");
  svg.setAttribute("class", "w-full rounded-xl border border-slate-200 bg-slate-50");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Illustration of a neural network with " + layers.length + " hidden layers");
  const xpos = i => 40 + 620 * i / (counts.length - 1);
  const visible = counts.map(n => Math.min(6, n));
  const ypos = (i,j) => 22 + j * 116 / Math.max(1,visible[i]-1);
  const shape = (tag, attributes) => {
    const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attributes).forEach(([key,value]) => n.setAttribute(key, String(value)));
    svg.appendChild(n);
    return n;
  };
  for(let i=0;i<counts.length-1;i++)
    for(let j=0;j<visible[i];j++)
      for(let k=0;k<visible[i+1];k++)
        shape("line",{x1:xpos(i),y1:ypos(i,j),x2:xpos(i+1),y2:ypos(i+1,k)});
  for(let i=0;i<counts.length;i++) {
    for(let j=0;j<visible[i];j++)
      shape("circle",{cx:xpos(i),cy:ypos(i,j),r:5,fill:i===0?"#0f766e":i===counts.length-1?"#2563eb":"#d97706"});
    const title = shape("text",{x:xpos(i),y:161,fill:"#64748b","font-size":11,"text-anchor":"middle"});
    title.textContent = i===0?"TF-IDF":i===counts.length-1?"Output":"Hidden "+i;
    const number = shape("text",{x:xpos(i),y:178,fill:"#0f172a","font-size":12,"font-weight":700,"text-anchor":"middle"});
    number.textContent = String(counts[i]);
  }
  host.appendChild(svg);
  addNode(host,"p",{class:"mt-1 text-xs text-slate-500",text:"Up to six nodes per layer are shown; labels give actual neuron counts."});
}

function drawChart() {
  if (!epochsHistory.length) return;
  const host = E("chartArea");
  host.replaceChildren();
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox","0 0 690 230"); svg.setAttribute("class","w-full");
  svg.setAttribute("role","img"); svg.setAttribute("aria-label","Training and validation "+metric+" by epoch");
  const keyTrain = metric === "accuracy"?"accuracy":"loss";
  const keyVal = metric === "accuracy"?"valAccuracy":"valLoss";
  const values = epochsHistory.flatMap(row=>[row[keyTrain],row[keyVal]]).filter(Number.isFinite);
  const bound = metric === "accuracy"?1:Math.max(.1,...values)*1.05;
  const firstEpoch=epochsHistory[0].epoch,lastEpoch=epochsHistory[epochsHistory.length-1].epoch;
  const xEpoch=e=>48+(lastEpoch===firstEpoch?0:605*(e-firstEpoch)/(lastEpoch-firstEpoch));
  const x=i=>xEpoch(epochsHistory[i].epoch);
  const y = n => 195 - 167*(Math.max(0,Math.min(bound,Number(n)||0))/bound);
  const shape = (tag,attrs,text) => {
    const element = document.createElementNS("http://www.w3.org/2000/svg",tag);
    Object.entries(attrs).forEach(([name,value]) => element.setAttribute(name,String(value)));
    if(text !== undefined) element.textContent = text;
    svg.appendChild(element);
    return element;
  };
  for(const part of [0,.25,.5,.75,1]) {
    shape("line",{x1:48,x2:653,y1:y(bound*part),y2:y(bound*part),stroke:"#e2e8f0"});
    shape("text",{x:43,y:y(bound*part)+4,"text-anchor":"end","font-size":11,fill:"#64748b"},
      metric==="accuracy"?Math.round(part*100)+"%":(part*bound).toFixed(2));
  }
  const points = key => epochsHistory.map((item,i)=>x(i)+","+y(item[key])).join(" ");
  shape("polyline",{points:points(keyTrain),stroke:palette.train,fill:"none","stroke-width":2.5});
  shape("polyline",{points:points(keyVal),stroke:palette.validation,fill:"none","stroke-width":2.5});
  if(selectedEpoch!==null && selectedEpoch>=firstEpoch && selectedEpoch<=lastEpoch) {
    const px=xEpoch(selectedEpoch);
    shape("line",{x1:px,x2:px,y1:28,y2:195,stroke:"#475569","stroke-width":1.5,"stroke-dasharray":"4 4"});
    shape("text",{x:Math.max(65,Math.min(590,px+7)),y:17,fill:"#334155","font-size":11},"Kept epoch "+selectedEpoch);
    const best=epochsHistory.find(row=>row.epoch===selectedEpoch);
    if(best)shape("circle",{cx:px,cy:y(best[keyVal]),r:4,fill:palette.validation,stroke:"#fff","stroke-width":1.5});
  }
  shape("text",{x:48,y:216,"font-size":11,fill:"#64748b"},"Epoch "+epochsHistory[0].epoch);
  shape("text",{x:653,y:216,"font-size":11,fill:"#64748b","text-anchor":"end"},"Epoch "+epochsHistory[epochsHistory.length-1].epoch);
  host.appendChild(svg);
  const legend=addNode(host,"div",{class:"mt-1 flex flex-wrap gap-4 text-xs text-slate-600"});
  const trainLegend=addNode(legend,"span",{text:"● Training"});trainLegend.style.color=palette.train;
  const valLegend=addNode(legend,"span",{text:"● Validation"});valLegend.style.color=palette.validation;
  if(selectedEpoch!==null)addNode(legend,"span",{text:"Dashed line = saved model (epoch "+selectedEpoch+")"});
}

function showVocabularyClues(vocab) {
  const host=E("vocabularyReport");host.replaceChildren();
  addNode(host,"h4",{class:"font-semibold text-sm",text:"Which word clues did the model select?"});
  addNode(host,"p",{class:"mt-1 mb-3 text-xs text-slate-600",
    text:"These counts show how many different training MPs used each term. They are correlations, not political opinions or proof of why the model predicts a party."});
  const table=addNode(host,"table",{class:"w-full text-sm text-left"});
  const head=addNode(table,"thead");
  const h=addNode(head,"tr",{class:"border-b border-slate-200 text-xs text-slate-500"});
  ["Word or phrase","Labour MPs","Conservative MPs"].forEach(x=>addNode(h,"th",{class:"py-2 pr-2",text:x}));
  const body=addNode(table,"tbody");
  for(const clue of vocab.clues || []) {
    const tr=addNode(body,"tr",{class:"border-b border-slate-100"});
    [clue.term,String(clue.labourMPs),String(clue.conservativeMPs)].forEach(
      (v,i)=>addNode(tr,"td",{class:"py-2 pr-2"+(i===0?" font-medium":""),text:v}));
  }
}
function showEvaluation(rows, predictions) {
  const stats=engine.evaluatePredictions(rows,predictions);
  const host=E("evaluationReport");host.replaceChildren();
  addNode(host,"h4",{class:"font-semibold text-base",text:"Where does the model make mistakes?"});
  addNode(host,"p",{class:"mt-1 text-xs text-slate-600",
    text:"Unseen test speeches only. Each row is the MP's actual party; each column is the prediction. Use validation results, not repeated test scores, to select settings."});
  const table=addNode(host,"table",{class:"w-full mt-3 text-sm text-left"});
  const head=addNode(table,"thead"),h=addNode(head,"tr",{class:"border-b border-slate-200"});
  ["Actual ↓ / Predicted →","Labour","Conservative"].forEach(v=>addNode(h,"th",{class:"py-2 pr-2",text:v}));
  const body=addNode(table,"tbody");
  for(const party of labels) {
    const tr=addNode(body,"tr",{class:"border-b border-slate-100"});
    addNode(tr,"th",{class:"py-2 pr-2 font-semibold",text:party});
    for(const predicted of labels) addNode(tr,"td",{class:"py-2 pr-2",text:String(stats.confusion[party][predicted])});
  }
  addNode(host,"h4",{class:"font-semibold text-sm mt-5",text:"Accuracy by decade"});
  addNode(host,"p",{class:"text-xs text-slate-600 mt-1",
    text:"Speeches from different periods may use different language. Small groups are less reliable."});
  const dt=addNode(host,"table",{class:"w-full text-sm text-left mt-2"});
  const dh=addNode(dt,"thead"),dr=addNode(dh,"tr",{class:"border-b border-slate-200"});
  ["Period","Test speeches","Accuracy"].forEach(v=>addNode(dr,"th",{class:"py-2 pr-2",text:v}));
  const db=addNode(dt,"tbody");
  for(const decade of stats.decades) {
    const tr=addNode(db,"tr",{class:"border-b border-slate-100"});
    [decade.period,String(decade.total),formatPercent(decade.correct/decade.total)].forEach(
      v=>addNode(tr,"td",{class:"py-2 pr-2",text:v}));
  }
}
function validateSample(data) {
  if (!data || !Array.isArray(data.examples)) throw new Error("The file must contain a prepared CommonsSpeech examples array.");
  if (data.examples.length < 100) throw new Error("The sample is too small for this classroom lab.");
  for(const item of data.examples) {
    if (!item || !labels.includes(item.party) || !String(item.mp||"").trim() || !String(item.text||"").trim())
      throw new Error("Invalid speech, speaker or party label in sample.");
  }
  const speakers = new Set(data.examples.map(item => item.party + "|" + item.mp));
  return { speeches:data.examples.length, mps:speakers.size };
}
function loadSample(data) {
  const info = validateSample(data);
  resetModel();
  sample = data;
  E("speechCount").textContent = info.speeches.toLocaleString();
  E("speakerCount").textContent = info.mps.toLocaleString();
  const counts = labels.map(party=>data.examples.filter(x=>x.party===party).length);
  E("dataStatus").textContent = "Ready: "+counts[0]+" Labour and "+counts[1]+" Conservative speeches."+
    (info.speeches>5200?" Expanded sample; the overfitting demo retains its original data.":"");
  textStatus("Choose an architecture and train. The split will keep MPs out of multiple sets.");
  setBusy(false);
}
async function initialiseData() {
  try {
    const response = await fetch("./data/commons-speech-sample.json", {cache:"no-store"});
    if (!response.ok) throw new Error("HTTP " + response.status);
    const data = await response.json();
    loadSample(data);
  } catch(error) {
    E("dataStatus").textContent = "Prepared speech sample not available (" + error.message + "). The source-data build may still be pending; you can load a prepared JSON sample below.";
    textStatus("Speech data is needed before training. The numerical CSV classifier remains available.");
    setBusy(false);
  }
}
async function train() {
  if(busy || !sample) return;
  const epochs = Number(E("epochs").value);
  const stopEarly = E("earlyStop").checked;
  if (!Number.isSafeInteger(epochs) || epochs < 1 || epochs > 1000) {
    textStatus("Enter a whole number of epochs between 1 and 1,000.");
    return;
  }
  resetModel();
  setBusy(true);
  const tensors = [];
  let created = null;
  try {
    textStatus("Preparing speaker-grouped train, validation and test sets…");
    await tf.ready();
    const heldOutParts = engine.splitBySpeaker(sample.examples);
    const originalTrainingCount = heldOutParts.train.length;
    const qualityMode=E("trainingSet").value;
    const chosenTrain=engine.chooseTrainingSpeeches(heldOutParts.train,qualityMode);
    const parts={...heldOutParts,train:chosenTrain};
    if(parts.train.length<200 || labels.some(party=>parts.train.filter(row=>row.party===party).length<80))
      throw new Error("Not enough speeches from both parties after filtering. Choose All speeches.");
    const vocab = engine.buildVocabulary(parts.train, Number(E("vocabulary").value));
    showVocabularyClues(vocab);
    // Each feature and the IDF weight are learned from training speakers only.
    const tensorPair = rows => {
      const input = tf.tensor2d(engine.encodeRows(rows, vocab),[rows.length,vocab.words.length]);
      const output = tf.tensor2d(rows.map(row=>labels.map(party=>Number(row.party === party))));
      tensors.push(input,output);
      return [input,output];
    };
    const [trainX,trainY] = tensorPair(parts.train);
    const [validationX,validationY] = tensorPair(parts.validation);
    const [testX,testY] = tensorPair(parts.test);
    drawDiagram();
    created = tf.sequential();
    layers.forEach((count,i) => {
      const options = {
        units:count, activation:E("activation").value,
        kernelRegularizer:tf.regularizers.l2({l2:.002})
      };
      if(i===0) options.inputShape = [vocab.words.length];
      created.add(tf.layers.dense(options));
      created.add(tf.layers.dropout({rate:.35}));
    });
    created.add(tf.layers.dense({units:2,activation:"softmax"}));
    created.compile({optimizer:tf.train.adam(.001),loss:"categoricalCrossentropy",metrics:["accuracy"]});
    const interval = Math.max(1,Math.ceil(epochs / 120));
    let bestLoss=Infinity,bestWeights=null,bestEpoch=0,staleEpochs=0,stoppedEarly=false;
    try {
    await created.fit(trainX,trainY,{
      batchSize:32, epochs, shuffle:true,
      validationData: [validationX, validationY],
      callbacks:{onEpochEnd:async (epoch, logs) => {
        lastPoint = {
          epoch:epoch+1, accuracy:logs.acc ?? logs.accuracy,
          valAccuracy:logs.val_acc ?? logs.val_accuracy,
          loss:logs.loss,valLoss:logs.val_loss
        };
        if(epoch===0 || (epoch+1)%interval===0 || epoch===epochs-1) {
          epochsHistory.push(lastPoint);
          textStatus("Epoch "+(epoch+1)+"/"+epochs+" · training "+formatPercent(lastPoint.accuracy)+
            " · validation "+formatPercent(lastPoint.valAccuracy));
          if(epochsHistory.length===1 || epochsHistory.length%5===0 || epoch===epochs-1) drawChart();
          await tf.nextFrame();
        }
        if(Number.isFinite(lastPoint.valLoss) && lastPoint.valLoss < bestLoss-.001) {
          bestLoss=lastPoint.valLoss;
          bestEpoch=epoch+1;
          if(bestWeights) bestWeights.forEach(w=>w.dispose());
          bestWeights=created.getWeights().map(w=>w.clone());
          staleEpochs=0;
        } else {
          staleEpochs++;
          // Only stop if explicitly enabled; give the model at least 12 epochs.
          if(stopEarly && epoch+1 >= 12 && staleEpochs >= 6) {
            stoppedEarly=true;
            created.stopTraining=true;
          }
        }
      }}
    });
    if(bestWeights) created.setWeights(bestWeights);
    selectedEpoch=bestEpoch||null;
    drawChart();
    } finally {
      if(bestWeights) bestWeights.forEach(w=>w.dispose());
    }
    textStatus("Comparing with logistic regression…");
    const linear=tf.sequential();
    let baselineAccuracy;
    try {
      linear.add(tf.layers.dense({units:2,inputShape:[vocab.words.length],activation:"softmax",
        kernelRegularizer:tf.regularizers.l2({l2:.01})}));
      linear.compile({optimizer:tf.train.adam(.002),loss:"categoricalCrossentropy",metrics:["accuracy"]});
      await linear.fit(trainX,trainY,{batchSize:32,epochs:Math.min(epochs,25),shuffle:true,
        validationData:[validationX,validationY],
        callbacks:[tf.callbacks.earlyStopping({monitor:"val_loss",patience:4})]});
      const baselineResult=linear.evaluate(testX,testY);
      const baselineTensors=Array.isArray(baselineResult)?baselineResult:[baselineResult];
      baselineAccuracy=(await baselineTensors[1].data())[0];
      baselineTensors.forEach(t=>t.dispose());
    } finally {linear.dispose();}
    textStatus("Evaluating held-out MPs…");
    // Predict once for the confusion matrix and decade breakdown.
    const predictedTensor=created.predict(testX);
    try {
      const scores=await predictedTensor.data();
      const guessed=parts.test.map((_,i)=>labels[scores[i*2]>=scores[i*2+1]?0:1]);
      showEvaluation(parts.test,guessed);
    } finally { predictedTensor.dispose(); }
    const result = created.evaluate(testX,testY);
    const outputs = Array.isArray(result)?result:[result];
    const testAcc = (await outputs[1].data())[0];
    outputs.forEach(x=>x.dispose());
    model = created;
    created = null;
    vocabulary = vocab;
    split = parts;
    // Show accuracy for the restored best-validation checkpoint.
    const trainResult=model.evaluate(trainX,trainY);
    const valResult=model.evaluate(validationX,validationY);
    const trainT=Array.isArray(trainResult)?trainResult:[trainResult];
    const valT=Array.isArray(valResult)?valResult:[valResult];
    E("trainAccuracy").textContent=formatPercent((await trainT[1].data())[0]);
    E("validationAccuracy").textContent=formatPercent((await valT[1].data())[0]);
    [...trainT,...valT].forEach(t=>t.dispose());
    E("testAccuracy").textContent = formatPercent(testAcc);
    // Choose the majority-party baseline from training data only, never from the test labels.
    const trainingCounts=labels.map(party=>parts.train.filter(x=>x.party===party).length);
    const majorityParty=labels[trainingCounts[1]>trainingCounts[0]?1:0];
    const baseline=parts.test.filter(x=>x.party===majorityParty).length/parts.test.length;
    E("splitInfo").textContent =
      "Speaker-grouped split · "+parts.train.length+" training / "+parts.validation.length+" validation / "+parts.test.length+
      " unseen test speeches. "+vocab.words.length+" TF-IDF word and phrase features."+
      (qualityMode==="substantial"?" Filter kept "+parts.train.length+" of "+originalTrainingCount+
      " training speeches; validation and test sets are unchanged.":" All eligible training speeches used.");
    E("baselineInfo").textContent="Neural network: "+formatPercent(testAcc)+
      " · Logistic regression: "+formatPercent(baselineAccuracy)+
      " · Always predict "+majorityParty+": "+formatPercent(baseline)+".";
    textStatus((stoppedEarly
      ? "Stopped after "+lastPoint.epoch+" of "+epochs+" epochs because validation loss stopped improving."
      : "Completed all "+epochs+" requested epochs.")+
      (bestEpoch ? " Restored epoch "+bestEpoch+" (lowest validation loss). The cards show that saved model; the graph shows all epochs." : "")+
      " Test MPs were not present in training or validation.");
    E("randomSpeech").disabled=false;
    E("predict").disabled=false;
    drawDiagram();
  } catch(error) {
    textStatus("Training failed: "+error.message);
  } finally {
    tensors.forEach(t=>t.dispose());
    if(created) created.dispose();
    setBusy(false);
  }
}
async function predict() {
  if(!model||!vocabulary||busy) return;
  const text = E("speechText").value.trim();
  if (text.length < 20) {
    E("predictionResult").textContent="Enter a longer paragraph before predicting.";
    return;
  }
  const vec = engine.modelInputsFromText(text,vocabulary);
  if (!vec.some(x=>x>0)) {
    E("predictionResult").textContent="None of the model's learned words appear in this speech. Try a longer, more relevant paragraph.";
    return;
  }
  let x, y;
  try {
    x = tf.tensor2d([vec]);
    y = model.predict(x);
    const scores = (await y.array())[0];
    const best = scores[0]>=scores[1]?0:1;
    const box = E("predictionResult");
    box.replaceChildren();
    const result = addNode(box,"div",{class:"rounded-xl bg-slate-50 border border-slate-200 p-5"});
    addNode(result,"p",{class:"text-xs font-bold uppercase tracking-wider text-slate-500",text:"Prediction"});
    addNode(result,"p",{class:"mt-1 text-2xl font-bold",text:labels[best]});
    addNode(result,"p",{class:"mt-1 text-sm text-slate-600",text:"Model score: "+formatPercent(scores[best])+" (not a calibrated probability)"});
    for(let i=0;i<2;i++) {
      const line=addNode(result,"div",{class:"mt-4"});
      const header=addNode(line,"div",{class:"flex justify-between text-sm font-medium"});
      addNode(header,"span",{text:labels[i]});
      addNode(header,"span",{text:formatPercent(scores[i])});
      const track=addNode(line,"div",{class:"mt-1 h-2.5 overflow-hidden rounded-full bg-slate-200"});
      const bar=addNode(track,"div",{class:"h-full rounded-full"});
      bar.style.width=Math.min(100,Math.max(0,scores[i]*100))+"%";
      bar.style.background=i===0?"#0f766e":"#475569";
    }
    if (heldOutChoice && text===heldOutChoice.text) {
      addNode(result,"p",{class:"mt-4 text-sm font-medium",text:
        "Source label: "+heldOutChoice.party+" · "+(labels[best]===heldOutChoice.party?"Correct":"Incorrect")+" on this example."});
    }
    const tokenHits = engine.featureTokens(text).filter(word=>vocabulary.words.includes(word));
    addNode(result,"p",{class:"mt-3 text-xs text-slate-500",text:"Recognised "+new Set(tokenHits).size+" distinct vocabulary terms."});
  } catch(error) {
    E("predictionResult").textContent="Prediction failed: "+error.message;
  } finally {
    if (x) x.dispose();
    if (y) y.dispose();
  }
}
function showHeldout() {
  if(!split || !split.test.length) return;
  const row=split.test[Math.floor(Math.random()*split.test.length)];
  heldOutChoice=row;
  E("speechText").value=row.text;
  E("selectedSpeechMeta").textContent="Held-out Commons example · "+row.year+" · party hidden until prediction";
  E("predictionResult").replaceChildren();
}
E("addLayer").addEventListener("click",()=>{if(busy||layers.length>=4)return;layers.push(16);layersUI();newConfig();});
E("activation").addEventListener("change",newConfig);
E("vocabulary").addEventListener("change",newConfig);
E("trainingSet").addEventListener("change",newConfig);
E("epochs").addEventListener("change",newConfig);
E("earlyStop").addEventListener("change",newConfig);
E("train").addEventListener("click",train);
E("predict").addEventListener("click",predict);
E("randomSpeech").addEventListener("click",showHeldout);
E("speechText").addEventListener("input",()=>{if(heldOutChoice && E("speechText").value!==heldOutChoice.text) heldOutChoice=null;});
E("showAccuracy").addEventListener("click",()=>{
  metric="accuracy"; E("showAccuracy").setAttribute("aria-pressed","true"); E("showLoss").setAttribute("aria-pressed","false");
  drawChart();
});
E("showLoss").addEventListener("click",()=>{
  metric="loss"; E("showAccuracy").setAttribute("aria-pressed","false"); E("showLoss").setAttribute("aria-pressed","true");
  drawChart();
});
E("localData").addEventListener("change",async e=>{
  const file=e.target.files[0];
  if(!file)return;
  if(file.size>15*1024*1024) {E("dataStatus").textContent="The prepared sample must be under 15 MB, not the 1 GB source ZIP.";return;}
  try {loadSample(JSON.parse(await file.text()));}
  catch(error) {E("dataStatus").textContent="Cannot load sample: "+error.message;}
});
layersUI();
initialiseData();
})();
