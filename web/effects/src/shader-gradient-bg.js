// Hero background: animated shader gradient (react-three-fiber via shadergradient).
// Not self-mounting -- the page calls mount()/destroy() so it can lazy-load and
// pause this off-screen. See assets/effects/README in landing.html's inline script.
import React from 'react';
import { createRoot } from 'react-dom/client';
import { ShaderGradientCanvas, ShaderGradient } from 'shadergradient';

// Matches the site's green/gold theme (manifest.json theme_color, index.html accents).
const DEFAULT_PROPS = {
  type: 'waterPlane',
  animate: 'on',
  color1: '#0b3d24',
  color2: '#2b5230',
  color3: '#e8b93b',
  reflection: 0.1,
  wireframe: false,
  grain: 'off',
  uSpeed: 0.2,
  uStrength: 3,
  uDensity: 1.3,
  uFrequency: 5.5,
  uAmplitude: 3,
  positionY: 0,
  rotationX: 0,
  rotationY: 0,
  rotationZ: 60,
  cAzimuthAngle: 180,
  cPolarAngle: 90,
  cDistance: 4,
  cameraZoom: 1,
  lightType: 'env',
  envPreset: 'city',
  brightness: 1,
};

export function mount(container, { pixelDensity = 1, props = {} } = {}) {
  const root = createRoot(container);
  root.render(
    React.createElement(
      ShaderGradientCanvas,
      { pixelDensity, style: { width: '100%', height: '100%', pointerEvents: 'none' } },
      React.createElement(ShaderGradient, { ...DEFAULT_PROPS, ...props })
    )
  );
  return {
    destroy() {
      root.unmount();
    },
  };
}
