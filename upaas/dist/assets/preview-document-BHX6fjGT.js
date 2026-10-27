function e(e){return e.replace(/<\/(script)/gi,`<\\/$1`).replace(/<!--/g,`<\\x21--`)}function t(e){return e.replace(/&/g,`&amp;`).replace(/</g,`&lt;`).replace(/>/g,`&gt;`).replace(/"/g,`&quot;`)}function n(e){return JSON.stringify(e).replace(/</g,`\\u003c`)}function r(e){return/\bclassName\s*[=:]/.test(e)}function i(e){return`(() => {
  "use strict";
  const reg = (globalThis.__unslothModules ??= Object.create(null));
  const AVAILABLE = ${n([...e].sort())};
  const lookup = (id) => {
    if (!Object.prototype.hasOwnProperty.call(reg, id)) {
      throw new Error(JSON.stringify(String(id)) + " isn't available in previews. Available: " + AVAILABLE.join(", "));
    }
    return reg[id];
  };
  window.__unslothPreview = {
    import: async (id, meta) => {
      const mod = lookup(id);
      const names = meta && meta.importedNames;
      if (names) {
        for (const name of names) {
          if (!Object.prototype.hasOwnProperty.call(mod, name)) {
            throw new SyntaxError("The requested module " + JSON.stringify(id) + " does not provide an export named " + JSON.stringify(name));
          }
        }
      }
      return mod;
    },
    dynamicImport: async () => {
      throw new Error("Dynamic import() isn't supported in previews.");
    },
    exportAll: (exports, source) => {
      if (source === exports || source == null || typeof source !== "object") return;
      for (const key in source) {
        if (key === "default" || key === "__esModule" || key in exports) continue;
        Object.defineProperty(exports, key, { enumerable: true, configurable: true, get: () => source[key] });
      }
    },
    importMeta: (file) => Object.freeze({ url: "preview:///" + file, env: Object.freeze({ MODE: "production", DEV: false, PROD: true, SSR: false }) }),
  };
})();`}function a(e,t){return`(async () => {
  "use strict";
  const P = window.__unslothPreview;
  const __vite_ssr_exports__ = Object.create(null);
  const __vite_ssr_import__ = P.import;
  const __vite_ssr_dynamic_import__ = P.dynamicImport;
  const __vite_ssr_exportAll__ = (source) => P.exportAll(__vite_ssr_exports__, source);
  const __vite_ssr_import_meta__ = P.importMeta(${n(t)});
  const __unsloth_local_app__ = await (async () => {
${e}
;return typeof App === "undefined" ? undefined : App;
  })();
  const Component = __vite_ssr_exports__.default ?? __vite_ssr_exports__.App ?? __unsloth_local_app__;
  const isComponent = typeof Component === "function" || (Component !== null && typeof Component === "object" && "$$typeof" in Component);
  if (!isComponent) {
    throw new Error("The preview needs a default export (or an App component) that is a React component.");
  }
  const React = globalThis.__unslothModules["react"];
  const { createRoot } = globalThis.__unslothModules["react-dom/client"];
  createRoot(document.getElementById("root"), {
    onUncaughtError: (error) => {
      const message = error && error.name ? error.name + ": " + error.message : String(error);
      window.dispatchEvent(new ErrorEvent("error", { error, message: "Uncaught " + message }));
    },
    onCaughtError: (error) => console.error(error),
  }).render(React.isValidElement(Component) ? Component : React.createElement(Component));
})();`}var o=`<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />`,s=t=>`<script>${e(t)}<\/script>`;function c({code:e,runtimes:n,tailwind:r,available:c,title:l,file:u}){return`${o}
<title>${t(l)}</title>
${r===null?``:s(r)}
</head>
<body>
<div id="root"></div>
${n.map(e=>s(e.code)).join(`
`)}
${s(i(c))}
${s(a(e,u))}
</body>
</html>`}function l(e,r){let i=e.map(({message:e,line:t,column:n})=>{let r=t>0?n>0?`line ${t}, column ${n}`:`line ${t}`:``;return r?`Compile error at ${r}: ${e}`:`Compile error: ${e}`});return`${o}
<title>${t(r)}</title>
</head>
<body>
<script>
(() => {
  for (const message of ${n(i)}) {
    window.dispatchEvent(new ErrorEvent("error", { message }));
  }
})();
<\/script>
</body>
</html>`}var u={files:{react:{url:`assets/preview-runtime/react.49615431d7.js`,bytes:220381,version:`19.3.1`},"lucide-react":{url:`assets/preview-runtime/lucide-react.77023e07e8.js`,bytes:782524,version:`1.61.0`},recharts:{url:`assets/preview-runtime/recharts.fb4cab7689.js`,bytes:499385,version:`3.7.0`},motion:{url:`assets/preview-runtime/motion.5363b66e98.js`,bytes:191886,version:`12.43.0`},tailwind:{url:`assets/preview-runtime/tailwind.f41327cfea.js`,bytes:271380,version:`4.2.4`}},modules:{react:`react`,"react/jsx-runtime":`react`,"react/jsx-dev-runtime":`react`,"react-dom":`react`,"react-dom/client":`react`,"lucide-react":`lucide-react`,recharts:`recharts`,motion:`motion`,"motion/react":`motion`,"framer-motion":`motion`},tailwind:`tailwind`},d=`react`;function f(e,t){let n=new Map,r=Object.keys(e.files).filter(t=>t!==e.tailwind),i=t=>{let n=new Set([d]),i=[];for(let r of t){let t=Object.hasOwn(e.modules,r)?e.modules[r]:void 0;t?n.add(t):i.includes(r)||i.push(r)}return{names:r.filter(e=>n.has(e)),unknown:i}},a=async n=>{let r=e.files[n];if(!r)throw Error(`Unknown preview library "${n}"`);let i=await t(r.url);if(i.byteLength!==r.bytes||i[0]===60)throw Error(`Preview library "${n}" didn't load correctly`);return new TextDecoder().decode(i)};return{resolve:i,unknownImports:t=>{let n=Object.keys(e.modules).sort().join(`, `);return i(t).unknown.map(e=>`${JSON.stringify(e)} isn't available in previews. Available: ${n}`)},load:e=>{let t=n.get(e);return t||(t=a(e),t.catch(()=>n.delete(e)),n.set(e,t)),t},available:Object.keys(e.modules),tailwind:e.tailwind}}var p=f(u,async e=>{let t=await fetch(new URL(`/${e}`,window.location.href));if(!t.ok)throw Error(`HTTP ${t.status}`);return new Uint8Array(await t.arrayBuffer())});async function m({code:e,deps:t,source:n,lang:i,title:a}){let{names:o}=p.resolve(t),[s,l]=await Promise.all([Promise.all(o.map(async e=>({name:e,code:await p.load(e)}))),r(n)?p.load(p.tailwind):Promise.resolve(null)]);return c({code:e,runtimes:s,tailwind:l,available:p.available,title:a,file:`App.${i}`})}function h(e){return p.unknownImports(e).map(e=>({message:e,line:0,column:0}))}function g(e,t){return l(e,t)}export{g as buildCompileErrorDocument,m as buildPreviewDocument,h as unknownImportDiagnostics};