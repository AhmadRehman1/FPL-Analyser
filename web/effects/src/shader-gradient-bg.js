// Hero background: animated shader gradient (react-three-fiber via shadergradient).
// Not self-mounting -- the page calls mount()/destroy() so it can lazy-load and
// tear it down off-screen (see the inline script at the end of landing.html).
import React from 'react';
import { createRoot } from 'react-dom/client';
import { ShaderGradientCanvas, ShaderGradient } from 'shadergradient';

// Matches the site's green/gold theme (manifest.json theme_color, index.html accents).
const DEFAULT_PROPS = {
  type: 'plane',
  animate: 'on',
  // Greens only: gold mixed into the gradient goes olive. Gold lives in the UI accents.
  color1: '#1b8a57', // pitch green
  color2: '#0a3322', // deep night green
  color3: '#5cc98e', // mint highlight
  reflection: 0.1,
  wireframe: false,
  grain: 'off',
  uSpeed: 0.18,
  uStrength: 3.4,
  uDensity: 1.2,
  uFrequency: 5.5,
  uAmplitude: 1,
  positionX: -1.4,
  positionY: 0,
  positionZ: 0,
  rotationX: 0,
  rotationY: 10,
  rotationZ: 50,
  cAzimuthAngle: 180,
  cPolarAngle: 90,
  cDistance: 3.6,
  cameraZoom: 1,
  lightType: '3d',
  envPreset: 'city',
  brightness: 1.05,
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
