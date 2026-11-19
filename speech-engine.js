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
  // Include adjacent phrases without joining words across ignored filler.
  function featureTokens(text) {
    const raw = String(text || "").toLowerCase().replace(/&[a-z]+;/g, " ").match(/[a-z]+(?:'[a-z]+)?/g) || [];
    const valid = raw.map(w => w.length > 2 && w.length < 24 && !STOP.has(w) && !MASK.has(w) ? w : null);
    const terms = valid.filter(Boolean);
    for (let i=0; i<valid.length-1; i++) if(valid[i] && valid[i+1]) terms.push(valid[i]+" "+valid[i+1]);
    return terms;
  }
  // A simple quality filter for our teaching experiment. It does not label
  // political opinions: it only checks whether a passage has enough substance.
  function isSubstantialSpeech(row) {
    const text=String(row && row.text || "").trim();
    const words=text.match(/[a-z]+(?:['’][a-z]+)?/gi) || [];
    if(words.length < 100) return false;
    if(new Set(tokenize(text)).size < 22) return false;
    // Exclude recognisable procedural openings, not all speeches mentioning procedure.
    if(/^(?:i beg to move,?\s*that,?\s*at the sitting\b|the house met at\b|the house proceeded to\b|to ask the secretary of state\b)/i.test(text)) return false;
    return true;
  }
  function chooseTrainingSpeeches(rows, mode="all") {
    if(mode!=="all" && mode!=="substantial") throw new Error("Unknown speech-quality mode.");
    return mode==="all" ? rows.slice() : rows.filter(isSubstantialSpeech);
  }
  function evaluatePredictions(rows, predictions) {
    if(rows.length!==predictions.length) throw new Error("Prediction count must match test speeches.");
    const labels=["Labour","Conservative"];
    const confusion={Labour:{Labour:0,Conservative:0},Conservative:{Labour:0,Conservative:0}};
    const decades=new Map();
    let correct=0;
    rows.forEach((row,i)=>{
      const actual=row.party,predicted=predictions[i];
      if(!labels.includes(actual)||!labels.includes(predicted))throw new Error("Invalid class label in evaluation.");
      confusion[actual][predicted]++;
      const good=actual===predicted;
      if(good)correct++;
      const year=Number(row.year);
      const decade=year<1990?"1979–1989":year<2000?"1990–1999":year<2010?"2000–2009":"2010–2019";
      if(!decades.has(decade))decades.set(decade,{period:decade,total:0,correct:0});
      const entry=decades.get(decade);
      entry.total++;if(good)entry.correct++;
    });
    return {confusion,total:rows.length,correct,
      decades:[...decades.values()].sort((a,b)=>a.period.localeCompare(b.period))};
  }
  function buildVocabulary(trainingDocs,limit=650) {
    // Count the MPs using each term so repeated speeches by one MP don't dominate.
    const speakerTerms=new Map(),docFreq=new Map();
    let nDocs=0;
    for(const doc of trainingDocs) {
      if(doc.party!=="Labour"&&doc.party!=="Conservative")continue;
      nDocs++;
      const key=String(doc.mp)+"|"+doc.party;
      if(!speakerTerms.has(key))speakerTerms.set(key,{party:doc.party,terms:new Set()});
      const group=speakerTerms.get(key);
      for(const word of new Set(featureTokens(doc.text))) {
        group.terms.add(word);
        docFreq.set(word,(docFreq.get(word)||0)+1);
      }
    }
    const totals={Labour:0,Conservative:0},bySpeaker=new Map();
    for(const group of speakerTerms.values()) {
      totals[group.party]++;
      for(const word of group.terms) {
        if(!bySpeaker.has(word))bySpeaker.set(word,{Labour:0,Conservative:0});
        bySpeaker.get(word)[group.party]++;
      }
    }
    const n=totals.Labour+totals.Conservative;
    const ranked=[...bySpeaker].map(([term,t])=>{
      const a=t.Labour,b=t.Conservative,c=totals.Labour-a,d=totals.Conservative-b;
      const count=a+b,denom=(a+b)*(c+d)*(a+c)*(b+d);
      return [term,count,denom>0?n*(a*d-b*c)**2/denom:0,a,b];
    }).filter(([,count])=>count>=3 && count<=n*.85)
      .sort((a,b)=>b[2]-a[2]||b[1]-a[1]||a[0].localeCompare(b[0]));
    const selected=ranked.slice(0,limit);
    // Keep tiny classroom fixtures usable.
    if(selected.length<Math.min(20,limit)) {
      const chosen=new Set(selected.map(row=>row[0]));
      for(const [term] of [...docFreq].filter(([,v])=>v>=2)
          .sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0]))) {
        if(selected.length>=Math.min(20,limit))break;
        if(!chosen.has(term)){selected.push([term,0,0]);chosen.add(term);}
      }
    }
    if(selected.length<20)throw new Error("Too little meaningful speech text to build the vocabulary.");
    return {words:selected.map(([term])=>term),
      idf:selected.map(([term])=>Math.log((1+nDocs)/(1+docFreq.get(term)))+1),
      trainDocs:nDocs,
      clues:selected.slice(0,10).map(([term,_,score,labourMPs,conservativeMPs])=>({
        term,score,labourMPs:labourMPs||0,conservativeMPs:conservativeMPs||0
      }))};
  }

  // Sparse Multinomial Naive Bayes: speaker-held-out, vocabulary learned from
  // training only; party names are masked before adjacent phrases are formed.
  function bayesFeatureTokens(text) {
    const raw = String(text || "").toLowerCase().replace(/&[a-z]+;/g," ")
      .match(/[a-z]+(?:['’][a-z]+)?/g) || [];
    const words = raw.map(s => {
      const w=s.replace(/['’]s$/,"");
      return w.length>=2 && w.length<24 && !MASK.has(w) ? w : null;
    });
    const result=words.filter(Boolean);
    for(let i=0;i<words.length-1;i++)
      if(words[i] && words[i+1]) result.push(words[i]+" "+words[i+1]);
    return result;
  }
  function bayesTermCounts(text) {
    const counts=new Map();
    for(const word of bayesFeatureTokens(text))
      counts.set(word,Math.min(3,(counts.get(word)||0)+1));
    return counts;
  }
  function trainNaiveBayes(rows,limit=8000) {
    const df=new Map(),partyTotals=[0,0];
    for(const row of rows) {
      const cls=row.party==="Labour"?0:row.party==="Conservative"?1:-1;
      if(cls<0)throw Error("Unexpected class");
      partyTotals[cls]++;
      for(const word of new Set(bayesFeatureTokens(row.text)))
        df.set(word,(df.get(word)||0)+1);
    }
    if(!partyTotals[0] || !partyTotals[1])throw Error("Two parties required");
    const vocab=[...df].filter(([,count])=>count>=3)
      .sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0]))
      .slice(0,limit).map(([word])=>word);
    if(vocab.length<20)throw Error("Insufficient vocabulary");
    const index=new Map(vocab.map((word,i)=>[word,i]));
    const counts=[new Float64Array(vocab.length),new Float64Array(vocab.length)],sum=[0,0];
    for(const row of rows) {
      const cls=row.party==="Labour"?0:1;
      for(const [word,n] of bayesTermCounts(row.text)) {
        const idx=index.get(word);
        if(idx!==undefined){counts[cls][idx]+=n;sum[cls]+=n;}
      }
    }
    const n=vocab.length, weights=new Map();
    for(let i=0;i<n;i++)weights.set(vocab[i],
      Math.log((counts[0][i]+1)/(sum[0]+n))-Math.log((counts[1][i]+1)/(sum[1]+n)));
    return {weights,termCount:n,bias:Math.log(partyTotals[0]/partyTotals[1])};
  }
  function predictNaiveBayes(fitted,text) {
    if(!fitted)throw Error("Train Naive Bayes first");
    let logOdds=fitted.bias,matchedTerms=0;
    for(const [word,n] of bayesTermCounts(text)) {
      const weight=fitted.weights.get(word);
      if(weight!==undefined){logOdds+=n*weight;matchedTerms++;}
    }
    const p=1/(1+Math.exp(-Math.max(-50,Math.min(50,logOdds))));
    return {label:p>=.5?"Labour":"Conservative",scores:[p,1-p],matchedTerms};
  }
  const indexCache=new WeakMap();
  function vocabularyIndex(vocab) {
    if(!indexCache.has(vocab))
      indexCache.set(vocab,new Map(vocab.words.map((word,i)=>[word,i])));
    return indexCache.get(vocab);
  }
  function makeVector(text, vocabulary) {
    const words = featureTokens(text);
    const lookup = vocabularyIndex(vocabulary);
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
    const width=vocab.words.length,flat=new Float32Array(rows.length*width);
    rows.forEach((row,i)=>flat.set(makeVector(row.text,vocab),i*width));
    return flat;
  }
  function modelInputsFromText(text, vocabulary) {
    return Array.from(makeVector(text, vocabulary));
  }
  return { tokenize, featureTokens, splitBySpeaker, isSubstantialSpeech, chooseTrainingSpeeches,
    evaluatePredictions, buildVocabulary, makeVector, encodeRows, modelInputsFromText, seededRandom,
    bayesFeatureTokens,trainNaiveBayes,predictNaiveBayes };
});
