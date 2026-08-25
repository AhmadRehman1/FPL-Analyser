// App-shell-only service worker: caches the static PWA shell (HTML/CSS/JS/icons/manifest) for
// installability and offline load, but never touches data/dashboard/*.json -- those are fetched
// directly from raw.githubusercontent.com with cache: "no-store" by index.html itself, and must
// always hit the network for fresh gameweek data. Bumping CACHE_NAME is the only thing needed to
// invalidate old shells on a deploy.
const CACHE_NAME = "fq-shell-v5";
const SHELL_URLS = [
  "./",
  "./index.html",
  "./landing.html",
  "./track-record.html",
  "./manifest.json",
  "./icons/icon-192.png",
  "./icons/icon-512.png",
  "./icons/icon-512-maskable.png",
  "./icons/apple-touch-icon.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(SHELL_URLS)).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((names) => Promise.all(names.filter((n) => n !== CACHE_NAME).map((n) => caches.delete(n))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  // Only ever intercept same-origin GETs for the app shell -- everything else (in particular
  // every data/dashboard/*.json fetch, which lives on a different origin entirely) passes
  // straight through to the network untouched.
  if (event.request.method !== "GET" || url.origin !== self.location.origin) return;

  event.respondWith(
    caches.match(event.request).then((cached) => {
      const network = fetch(event.request)
        .then((response) => {
          if (response.ok) {
            const copy = response.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(event.request, copy));
          }
          return response;
        })
        .catch(() => cached);
      return cached || network;
    })
  );
});
