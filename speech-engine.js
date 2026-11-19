/* CommonsSpeech educational text features and grouped split.
   Shared pure functions: browser (SpeechEngine global) and Node smoke tests.
   Original data: Blumenau (2021), UK Data Service 854292. */
(function (root, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.SpeechEngine = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";
  const STOP = new Set(("a about above after again against all am an and any are aren't as at be because been before being below between both but by can cannot could did do does doing down during each few for from further had has have having he her here hers herself him himself his how i if in into is it its itself just let's me more most must my myself no nor not of off on once only or other our ours ourselves out over own same she should so some such than that the their theirs them themselves then there these they this those through to too under until up very was we were what when where which while who whom why will with would you your yours yourself yourselves also quite really said says say get got one two three four five six seven eight nine ten mr mrs ms sir madam honourable honorable hon right rt member house commons minister secretary state government opposition speaker gentleman gentlemen lady ladies friend colleagues members question answer debate committee time thank ask asked asking member's").split(/\s+/));
  const MASK = new Set(["labour", "conservative", "conservatives", "tory", "tories", "lab", "tories'", "con"]);
  function tokenize(text) {
    return String(text || "").toLowerCase().replace(/&[a-z]+;/g, " ").match(/[a-z]+(?:'[a-z]+)?/g)?.filter(word =>
      word.length > 2 && word.length < 24 && !STOP.has(word) && !MASK.has(word)
    ) || [];
  }
  function shuffled(list, random) {
    const values = list.slice();
    for (let i = values.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [values[i], values[j]] = [values[j], values[i]];
    }
    return values;
  }
  function seededRandom(seed = 20261119) {
    let x = seed >>> 0;
    return () => ((x = (Math.imul(x, 1664525) + 1013904223) >>> 0) / 4294967296);
  }
  // Split by real person, globally: even MPs who changed parties never leak between sets.
  // Stratify the SPEAKER groups by the party with most sampled speeches.
  function splitBySpeaker(examples, seed = 20261119) {
    const random = seededRandom(seed);
    const speakers = new Map();
    examples.forEach(row => {
      if (!row || !["Labour","Conservative"].includes(row.party) ||
          !String(row.text || "").trim() || !String(row.mp || "").trim()) return;
      const key = String(row.mp);
      if (!speakers.has(key)) speakers.set(key, []);
      speakers.get(key).push(row);
    });
    const groups = { Labour: [], Conservative: [] };
    speakers.forEach((records, person) => {
      const labourCount = records.filter(row => row.party === "Labour").length;
      const conservativeCount = records.length - labourCount;
      const primary = labourCount >= conservativeCount ? "Labour" : "Conservative";
      groups[primary].push([person, records]);
    });
    const split = { train: [], validation: [], test: [] };
    for (const party of ["Labour", "Conservative"]) {
      const all = shuffled(groups[party], random);
      if (all.length < 8) throw new Error(party + " needs at least eight distinct speakers for speaker-held-out evaluation.");
      const testSpeakers = Math.max(1, Math.round(all.length * .15));
      const validationSpeakers = Math.max(1, Math.round(all.length * .15));
      all.forEach(([_, records], index) => {
        const target = index < testSpeakers ? "test" : index < testSpeakers + validationSpeakers ? "validation" : "train";
        split[target].push(...records);
      });
    }
    for (const name of ["train", "validation", "test"]) {
      split[name] = shuffled(split[name], random);
      const classes = new Set(split[name].map(row => row.party));
      if (classes.size < 2) throw new Error("Both party labels are needed in the " + name + " partition.");
    }
    return split;
  }
  function buildVocabulary(trainingDocs, limit = 650) {
    const df = new Map();
    for (const doc of trainingDocs) {
      const unique = new Set(tokenize(doc.text));
      unique.forEach(word => df.set(word, (df.get(word) || 0) + 1));
    }
    const sorted = [...df.entries()].filter(([, n]) => n >= 2)
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).slice(0, limit);
    if (sorted.length < 20) throw new Error("Too little meaningful speech text to build the vocabulary.");
    const n = trainingDocs.length;
    return { words: sorted.map(([term]) => term),
      idf: sorted.map(([, freq]) => Math.log((1 + n) / (1 + freq)) + 1),
      trainDocs: n };
  }
  function makeVector(text, vocabulary) {
    const words = tokenize(text);
    const lookup = new Map(vocabulary.words.map((word, i) => [word, i]));
    const counts = new Map();
    for (const word of words) {
      const index = lookup.get(word);
      if (index !== undefined) counts.set(index, (counts.get(index) || 0) + 1);
    }
    const vec = new Float32Array(vocabulary.words.length);
    let squared = 0;
    for (const [index, frequency] of counts) {
      const val = (1 + Math.log(frequency)) * vocabulary.idf[index];
      vec[index] = val;
      squared += val * val;
    }
    if (squared > 0) {
      const norm = Math.sqrt(squared);
      for (const index of counts.keys()) vec[index] /= norm;
    }
    return vec;
  }
  function encodeRows(rows, vocab) {
    return rows.map(row => Array.from(makeVector(row.text, vocab)));
  }
  function modelInputsFromText(text, vocabulary) {
    return Array.from(makeVector(text, vocabulary));
  }
  return { tokenize, splitBySpeaker, buildVocabulary, makeVector, encodeRows, modelInputsFromText, seededRandom };
});
