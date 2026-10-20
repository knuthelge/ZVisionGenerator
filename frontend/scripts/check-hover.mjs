// Fail when a `:hover` rule in the frontend's CSS is not inside `@media (hover: hover)`.
//
// A touch screen keeps `:hover` on whatever was tapped last, so a bare hover style leaves buttons looking pressed after
// a tap. Tailwind's `hover:` variant already carries the media query; this covers global.css and <style> blocks.
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';

const root = new URL('../src/', import.meta.url).pathname;

function files(dir) {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) return files(path);
    return /\.(svelte|css)$/.test(name) ? [path] : [];
  });
}

/** Return the 1-based lines of `:hover` selectors that no enclosing block limits to hover devices. */
function bareHoverLines(css, lineOffset = 0) {
  const hits = [];
  const stack = [];
  let preludeStart = 0;
  for (let i = 0; i < css.length; i++) {
    if (css.startsWith('/*', i)) {
      const end = css.indexOf('*/', i + 2);
      i = end < 0 ? css.length : end + 1;
      if (preludeStart <= i) preludeStart = i + 1;
      continue;
    }
    const ch = css[i];
    if (ch === '"' || ch === "'") {
      const end = css.indexOf(ch, i + 1);
      i = end < 0 ? css.length : end;
      continue;
    }
    if (ch === '{') {
      const prelude = css.slice(preludeStart, i);
      const guarded = stack.some((p) => /hover:\s*hover/.test(p));
      if (prelude.includes(':hover') && !prelude.trim().startsWith('@') && !guarded) {
        hits.push(lineOffset + css.slice(0, i).split('\n').length);
      }
      stack.push(prelude);
      preludeStart = i + 1;
    } else if (ch === '}') {
      stack.pop();
      preludeStart = i + 1;
    } else if (ch === ';') {
      preludeStart = i + 1;
    }
  }
  return hits;
}

const problems = [];
for (const path of files(root)) {
  const text = readFileSync(path, 'utf8');
  const blocks = path.endsWith('.css')
    ? [{ css: text, offset: 0 }]
    : [...text.matchAll(/<style[^>]*>([\s\S]*?)<\/style>/g)].map((m) => ({ css: m[1], offset: text.slice(0, m.index + m[0].indexOf('>') + 1).split('\n').length - 1 }));
  for (const { css, offset } of blocks) {
    for (const line of bareHoverLines(css, offset)) problems.push(`${relative(process.cwd(), path)}:${line}`);
  }
}

if (problems.length) {
  console.error(`${problems.length} :hover rule(s) outside @media (hover: hover); a tap leaves them on for a finger:`);
  for (const problem of problems) console.error(`  ${problem}`);
  process.exit(1);
}
