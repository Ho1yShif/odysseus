// static/js/model/matchKey.js
//
// Pure helper for matching a model name against a set of known keys. No DOM —
// safe to import anywhere and to unit-test under node.

// Return the most specific (longest) key that is a substring of `name`, or null.
// Returning the first match instead made "gpt-5.6-sol-mini" match the shorter
// "gpt-5.6-sol" key — billing it at gpt-5.6-sol rates (~16x) and showing the wrong
// context window.
export function matchModelKey(name, keys) {
  const n = (name || '').toLowerCase();
  let best = null;
  for (const key of keys) {
    if (n.includes(key) && (best === null || key.length > best.length)) {
      best = key;
    }
  }
  return best;
}
