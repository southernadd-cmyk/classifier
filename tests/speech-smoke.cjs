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
console.log("PASS: speaker-disjoint split, balanced labels, masked TF-IDF, vectors, UI wiring");
