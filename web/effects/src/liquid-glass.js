// liquid-glass-js, self-hosted, plus a helper that anchors the lens inside a
// section instead of the library's default position:fixed overlay.
import LiquidGlass from 'liquid-glass-js';

export default LiquidGlass;

// Places a non-draggable glass lens over `target`, refracting `decor`.
// `decor` must be an absolutely positioned inset:0 child of `section`, and must be
// plain DOM/CSS/SVG (the library clones it, so a canvas would clone blank).
// The lens lives inside `section`, so it scrolls with the page with no JS.
export function mountInSection(section, decor, target, params = {}) {
  const glass = new LiquidGlass({ ...params, background: decor, draggable: false, zIndex: 1 });

  // The library aligns its clone against viewport coords; inside the section the
  // decor always sits at (0, 0), so pin that and size from the decor itself.
  glass._syncBg = function () {
    this.bgX = 0;
    this.bgY = 0;
    this.lensInner.style.width = decor.offsetWidth + 'px';
    this.lensInner.style.height = decor.offsetHeight + 'px';
  };
  for (const el of [glass.lensEl, glass.glassEl]) {
    el.style.position = 'absolute';
    el.style.pointerEvents = 'none';
    section.appendChild(el);
  }

  function place() {
    glass.set({ width: target.offsetWidth, height: target.offsetHeight });
    glass._syncBg();
    glass.moveTo(target.offsetLeft, target.offsetTop);
  }
  place();
  glass.refresh();
  const ro = new ResizeObserver(() => {
    place();
    glass.refresh();
  });
  ro.observe(section);

  return {
    destroy() {
      ro.disconnect();
      glass.destroy();
    },
  };
}
