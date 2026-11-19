// Regression test: the enhanced speech classifier must generalise to unseen MPs.
// The held-out test set is NOT used to select features, models or thresholds here.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const engine = require("../speech-engine.js");

const dataset=JSON.parse(fs.readFileSync("data/commons-speech-sample.json","utf8"));
const parts=engine.splitBySpeaker(dataset.examples);
const trainSpeakers=new Set(parts.train.map(r=>String(r.mp)));
const validationSpeakers=new Set(parts.validation.map(r=>String(r.mp)));
const testSpeakers=new Set(parts.test.map(r=>String(r.mp)));
for(const mp of validationSpeakers)assert.ok(!trainSpeakers.has(mp),"validation MP leaked into training");
for(const mp of testSpeakers)assert.ok(!trainSpeakers.has(mp) && !validationSpeakers.has(mp),"test MP leaked");

const model=engine.trainNaiveBayes(parts.train,8000);
const validationSpeeches=parts.validation.map(r=>engine.predictNaiveBayes(model,r.text).label);
const perSpeech=validationSpeeches.filter((label,i)=>label===parts.validation[i].party).length/parts.validation.length;
const speaker=engine.evaluateNaiveBayesSpeakers(parts.validation,model);
assert.equal(speaker.total,validationSpeakers.size,"each validation MP counted once");
assert.equal(speaker.predictions.reduce((n,x)=>n+x.speechCount,0),parts.validation.length,
  "all speeches belong to exactly one evaluated MP");
assert.ok(speaker.accuracy>=0.70,
  "MP-level validation accuracy must meet the 70% target without speaker leakage");
assert.ok(speaker.accuracy>perSpeech,
  "aggregated MP-level classification must beat single-speech classification");
console.log("PASS: full-data speaker-disjoint Naive Bayes validation: speech "+
  (100*perSpeech).toFixed(1)+"%, MP "+(100*speaker.accuracy).toFixed(1)+
  "% across "+speaker.total+" previously unseen MPs");
