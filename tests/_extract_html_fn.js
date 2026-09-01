// Shared helper for the tests that exercise DOM-free renderers living inside index.html's main
// <script> (the "Explain this" sheet, the decision log, the risk-posture toggle). Brace-matches
// a `function NAME(...) { ... }` out of the source, correctly stepping over ' " ` strings,
// template `${...}` interpolations, // and /* */ comments, and /regex/ literals (the codebase
// uses `.replace(/"/g, "&quot;")` all over, and a naive matcher treats the `"` inside `/"/` as
// an unterminated string).

const fs = require("node:fs");
const path = require("node:path");

const HTML = fs.readFileSync(path.join(__dirname, "..", "index.html"), "utf8");

// Chars after which a `/` begins a regex literal, not a division. Covers the real cases in this
// file (`= /re/`, `(/re/`, `, /re/`, `return /re/`, `.replace(/re/`).
const _REGEX_PRECEDERS = new Set(["=", "(", ",", "{", "[", ":", ";", "!", "&", "|", "?", "+", "-", "*", "%", "^", "~", "<", ">"]);

function _lastCodeChar(s, i) {
  for (let j = i - 1; j >= 0; j--) {
    if (!/\s/.test(s[j])) return s[j];
  }
  return "";
}

// Single-arg on purpose: these are commonly used as `names.map(extractHtmlFn)`, and Array.map
// passes (item, index, array) -- a second parameter here would get clobbered by the index.
function extractHtmlFn(name) {
  const html = HTML;
  const start = html.search(new RegExp("function\\s+" + name + "\\s*\\("));
  if (start < 0) throw new Error("function not found: " + name);
  let i = html.indexOf("{", start) + 1; // step INSIDE the function body
  // Each entry is a mode. A "code" frame tracks its own brace depth (braces INSIDE this frame),
  // so `{a,b}` object literals inside a template `${...}` don't prematurely close it.
  const stack = [{ mode: "code", depth: 0 }];
  const top = () => stack[stack.length - 1];
  for (; i < html.length; i++) {
    const c = html[i], p = html[i - 1], m = top().mode;
    if (m === "'" || m === '"') { if (c === m && p !== "\\") stack.pop(); continue; }
    if (m === "`") {
      if (c === "`" && p !== "\\") stack.pop();
      else if (c === "$" && html[i + 1] === "{" && p !== "\\") { stack.push({ mode: "code", depth: 0 }); i++; }
      continue;
    }
    // code mode
    if (c === "/" && html[i + 1] === "/") { const nl = html.indexOf("\n", i); i = nl < 0 ? html.length : nl; continue; }
    if (c === "/" && html[i + 1] === "*") { const end = html.indexOf("*/", i + 2); i = end < 0 ? html.length : end + 1; continue; }
    if (c === "/" && (_REGEX_PRECEDERS.has(_lastCodeChar(html, i)) || _lastCodeChar(html, i) === "")) {
      i++;
      let inClass = false;
      for (; i < html.length; i++) {
        const rc = html[i], rp = html[i - 1];
        if (rp === "\\") continue;
        if (rc === "[") inClass = true;
        else if (rc === "]") inClass = false;
        else if (rc === "/" && !inClass) break;
      }
      while (i + 1 < html.length && /[a-z]/.test(html[i + 1])) i++; // flags
      continue;
    }
    if (c === "'" || c === '"' || c === "`") { stack.push({ mode: c }); continue; }
    if (c === "{") { top().depth++; continue; }
    if (c === "}") {
      if (top().depth > 0) { top().depth--; continue; }
      // depth 0 in this frame: this "}" closes the frame itself
      if (stack.length > 1) { stack.pop(); continue; }   // closes a ${...} interpolation
      return html.slice(start, i + 1);                    // closes the function body
    }
  }
  throw new Error("unbalanced braces extracting: " + name);
}

function extractHtmlConst(name) {
  const html = HTML;
  // `const NAME = { ... \n};` (multi-line object) or `const NAME = <expr>;` (single line)
  const objMatch = html.match(new RegExp("const " + name + "\\s*=\\s*\\{[\\s\\S]*?\\n\\};"));
  if (objMatch) return objMatch[0];
  const lineMatch = html.match(new RegExp("const " + name + "\\s*=\\s*[^;\\n]+;"));
  if (lineMatch) return lineMatch[0];
  throw new Error("const not found: " + name);
}

module.exports = { HTML, extractHtmlFn, extractHtmlConst };
