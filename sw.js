// App-shell service worker: caches the static PWA shell (HTML/CSS/JS/icons/manifest) for
// installability and offline load, but never touches data/dashboard/*.json -- those are fetched
// directly from raw.githubusercontent.com with cache: "no-store" by index.html itself, and must
// always hit the network for fresh gameweek data. Bumping CACHE_NAME is the only thing needed to
// invalidate old shells on a deploy.
//
// App gap 1: also handles real Web Push -- `push` renders the deadline/injury alert the
// scheduled pipeline sent via scripts/push_notify.py; `notificationclick` opens the app.
const CACHE_NAME = "fq-shell-v6";
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

// ---- Web Push (app gap 1) --------------------------------------------------
// The payload is the JSON scripts/push_alerts.build_push_payload() produced, sent by
// scripts/push_notify.py. Defensive: a malformed / bodyless push still shows something rather
// than the browser's own "This site has been updated in the background" fallback.
self.addEventListener("push", (event) => {
  let data = {};
  try { data = event.data ? event.data.json() : {}; } catch (e) { data = { body: event.data && event.data.text() }; }
  const title = data.title || "FPL Quant";
  const options = {
    body: data.body || "You have a new deadline alert.",
    tag: data.tag || "fpl-quant-deadline",
    renotify: true,
    icon: "./icons/icon-192.png",
    badge: "./icons/icon-192.png",
    data: { url: data.url || "./" },
  };
  event.waitUntil(self.registration.showNotification(title, options));
});

self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  const target = (event.notification.data && event.notification.data.url) || "./";
  event.waitUntil(
    self.clients.matchAll({ type: "window", includeUncontrolled: true }).then((clients) => {
      for (const c of clients) {
        if ("focus" in c) { c.focus(); if ("navigate" in c) c.navigate(target); return; }
      }
      return self.clients.openWindow(target);
    })
  );
});
