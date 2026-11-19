const assert = require("node:assert/strict");
const fs = require("node:fs");
const engine = require("../speech-engine.js");

const examples = [];
for (let p = 0; p < 2; p++) {
  const party = p === 0 ? "Labour" : "Conservative";
  for (let person = 0; person < 15; person++)
    for (let s = 0; s < 4; s++)
      examples.push({
        text: ("housing education environment transport economy taxation childcare industry policy " +
               (p === 0 ? "investment public service provision " : "enterprise market trade development ") +
               "local regional national neighbourhood science planning framework "
               ).repeat(2),
        mp: party + "-" + person, party, year: 2000 + s
      });
}
// Same individual can appear with two historical party labels; this must not leak.
examples.push({ ...examples[0], party: "Conservative", mp: "Labour-0" });
const first = engine.splitBySpeaker(examples);
const second = engine.splitBySpeaker(examples);
assert.deepEqual(first, second);
const speakerSets = [first.train, first.validation, first.test].map(rows => new Set(rows.map(row => row.mp)));
assert.equal(first.train.length + first.validation.length + first.test.length, examples.length);
for (let a = 0; a < speakerSets.length; a++) {
  for (let b = a + 1; b < speakerSets.length; b++) {
    for (const mp of speakerSets[a]) assert.ok(!speakerSets[b].has(mp), "no MP leakage");
  }
}
for (const part of [first.train, first.validation, first.test]) {
  assert.deepEqual([...new Set(part.map(row => row.party))].sort(), ["Conservative", "Labour"]);
}
const vocab = engine.buildVocabulary(first.train, 100);
assert.ok(vocab.words.length >= 20);
assert.ok(!vocab.words.includes("labour") && !vocab.words.includes("conservative"), "party names masked");
assert.ok(engine.featureTokens("public ownership of property").includes("public ownership"),"adjacent phrase");
assert.ok(!engine.featureTokens("public and private").includes("public private"),"no crossing stopword");
const compact=engine.encodeRows(first.train.slice(0,3),vocab);
assert.ok(compact instanceof Float32Array,"compact training matrix");
assert.equal(compact.length,3*vocab.words.length);
assert.equal(engine.isSubstantialSpeech({text:"A brief question."}),false,"short speech removed");
const longText=("Housing education health transport communities investment public services jobs local government " +
  "manufacturing technology research agriculture schools hospitals childcare environment climate energy " +
  "infrastructure apprenticeships skills taxation industry planning employment innovation science universities " +
  "libraries regeneration neighbourhood development security justice. ").repeat(5);
assert.equal(engine.isSubstantialSpeech({text:longText}),true,"substantial speech kept");
assert.equal(engine.isSubstantialSpeech({text:"I beg to move, That, at the sitting "+longText}),false,"procedural motion removed");
assert.equal(engine.chooseTrainingSpeeches([{text:"A brief question."},{text:longText}],"substantial").length,1);
assert.equal(engine.chooseTrainingSpeeches([{text:"A brief question."},{text:longText}],"all").length,2);
assert.throws(()=>engine.chooseTrainingSpeeches([],"invalid"),/quality mode/);
const diag=engine.evaluatePredictions([
 {party:"Labour",year:1982},{party:"Conservative",year:1984},
 {party:"Conservative",year:2008},{party:"Labour",year:2019}
],["Labour","Labour","Conservative","Conservative"]);
assert.deepEqual(diag.confusion,{Labour:{Labour:1,Conservative:1},Conservative:{Labour:1,Conservative:1}});
assert.deepEqual(diag.decades.map(x=>[x.period,x.total,x.correct]),[
 ["1979–1989",2,1],["2000–2009",1,1],["2010–2019",1,0]
]);
assert.throws(()=>engine.evaluatePredictions([{party:"Labour",year:2001}],[]),/Prediction count/);
assert.ok(Array.isArray(vocab.clues) && vocab.clues.length>0,"vocabulary clues documented");
const x = engine.makeVector(examples[0].text, vocab);
assert.equal(x.length, vocab.words.length);
assert.ok(x.some(n => n > 0));
const len = Math.sqrt(x.reduce((a,b) => a + b*b, 0));
assert.ok(Math.abs(len - 1) < 1e-5, "TF-IDF L2 normalised");
const y = engine.modelInputsFromText("unheard vocabulary xyzxyzxyz", vocab);
assert.ok(y.every(n => n === 0), "OOV only gives empty vector");
assert.throws(() => engine.splitBySpeaker(examples.slice(0, 12)), /distinct speakers/);
const html = fs.readFileSync("speech-lab.html", "utf8");
const ui = fs.readFileSync("speech-lab-ui.js", "utf8");
assert.match(html, /speech-lab-ui\.js/);
assert.match(html, /speech-engine\.js/);
assert.match(ui, /tf\.sequential\(\)/);
assert.match(ui, /validationData: \[validationX, validationY\]/);
assert.match(ui, /evaluate\(testX,\s*testY\)/);
assert.match(ui, /grouped split/i);
assert.match(ui, /staleEpochs=0/);
assert.match(html, /id="earlyStop" type="checkbox"/, "early stopping is optional");
assert.match(ui, /const stopEarly = E\("earlyStop"\)\.checked;/, "toggle controls stopping");
assert.match(ui, /stopEarly && epoch\+1 >= 12 && staleEpochs >= 6/, "stop rule includes a minimum");
assert.match(ui, /Completed all /, "full epoch completion is explained");
assert.match(ui, /const trainingCounts=labels\.map\(party=>parts\.train/, "baseline selected only from training");
assert.doesNotMatch(ui, /Guessing baseline:/, "majority class is not random guessing");
assert.match(html, /What is logistic regression\?/, "student explanation is present");
assert.match(ui, /tf\.layers\.dropout/);
assert.match(ui, /Logistic regression/);
assert.match(ui, /showEvaluation\(parts\.test,guessed\)/,"test predictions drive diagnostics");
assert.match(ui, /heldOutParts\.train/,"filter comes after speaker split");
assert.match(html, /id="trainingSet"/,"quality experiment selection");
assert.match(html, /id="evaluationReport"/,"evaluation visible");
assert.match(html, /id="vocabularyReport"/,"vocabulary audit visible");
assert.match(ui, /validation and test sets are unchanged/,"students see fair comparison");
assert.match(ui,/selectedEpoch=bestEpoch/,"selected epoch is marked");
assert.match(ui,/tensor2d\(engine\.encodeRows\(rows, vocab\),\[rows\.length,vocab\.words\.length\]\)/,"correct typed array tensor shape");
assert.match(html,/value="1600" selected/,"expanded vocabulary option");
assert.match(html,/A dashed line marks the best epoch/,"graph teaches selected model");
assert.match(fs.readFileSync("speech-lab-overfitting-ui.js","utf8"),/commons-speech-original\.json/,"original uses frozen data");
assert.match(html, /speech-lab-overfitting\.html/);
const oldHtml=fs.readFileSync("speech-lab-overfitting.html","utf8");
const oldUi=fs.readFileSync("speech-lab-overfitting-ui.js","utf8");
assert.match(oldHtml,/speech-engine-original\.js/);
assert.match(oldUi,/created\.fit\(trainX,trainY/);
assert.doesNotMatch(oldUi,/staleEpochs=0/);
console.log("PASS: speaker-disjoint split, balanced labels, masked TF-IDF, vectors, UI wiring");
