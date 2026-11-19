const fs = require('node:fs');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const parser = require('@babel/parser');

const html = fs.readFileSync('index.html', 'utf8');
const match = html.match(/<script type="text\/babel">([\s\S]*?)<\/script>/);
assert.ok(match, 'React JSX script is present');
const script = match[1];
parser.parse(script, { sourceType: 'script', plugins: ['jsx'] });
assert.ok(script.includes('validationData: [valX, valY]'), 'validation is separate');
assert.ok(script.includes('newModel.evaluate(testX, testY)'), 'held-out testing is separate');
assert.ok(script.includes("metadata ="), 'normalisation metadata is exported');

const helpers = script.slice(0, script.indexOf('function Architecture('));
assert.ok(helpers.length > 1000, 'pure preprocessing helpers found');
const context = { React: {}, module: { exports: {} } };
vm.runInNewContext(helpers + '\nmodule.exports = { splitData, buildSchema, encode, parseEpochs };', context);
const { splitData, buildSchema, encode, parseEpochs } = context.module.exports;

const rows = Array.from({ length: 120 }, (_, i) => ({
  id: String(i), amount: String(i + 5), category: i % 3 ? 'small' : 'large',
  target: i % 2 ? 'genuine' : 'forged'
}));
const first = splitData(rows, 'target');
const second = splitData(rows, 'target');
assert.equal(first.train.length + first.val.length + first.test.length, rows.length);
assert.deepEqual(Array.from(first.test, r => r.id), Array.from(second.test, r => r.id), 'repeated splits match');
const ids = [...first.train, ...first.val, ...first.test].map(row => row.id);
assert.equal(new Set(ids).size, rows.length, 'splits never share examples');
for (const part of [first.train, first.val, first.test]) {
  assert.equal(new Set(part.map(r => r.target)).size, 2, 'both labels represented in every split');
}
const schema = buildSchema(first.train, ['amount', 'category']);
assert.equal(schema[0].type, 'number');
assert.equal(schema[1].type, 'category');
const encoded = encode(first.train[0], schema);
assert.equal(encoded.length, 3);
assert.ok(Number.isFinite(encoded[0]), 'normalised values stay finite');
assert.equal(encoded.slice(1).reduce((a,b) => a+b, 0), 1, 'one hot encoding');
assert.equal(parseEpochs('150'), 150);
for (const bad of ['0', '1.5', '10001', 'abc', '']) {
  assert.throws(() => parseEpochs(bad));
}
assert.throws(() => splitData(rows.slice(0, 3), 'target'), /fewer than 5/);
console.log('PASS: JSX parser, separate evaluation, stratified split, normalisation, categories, epochs');
