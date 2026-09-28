import * as esbuild from 'esbuild';

const entries = ['shader-gradient-bg', 'liquid-logo', 'liquid-glass'];

await esbuild.build({
  entryPoints: entries.map((name) => `src/${name}.js`),
  outdir: '../../assets/effects',
  bundle: true,
  format: 'esm',
  minify: true,
  target: 'es2020',
  logLevel: 'info',
});
