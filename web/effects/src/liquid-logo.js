// Liquid metal effect for the header logo mark. Vanilla (no React/three) --
// @paper-design/shaders ships a plain ShaderMount class the react wrapper
// itself just calls into, so we call it directly.
import {
  ShaderMount,
  liquidMetalFragmentShader,
  toProcessedLiquidMetal,
  getShaderColorFromString,
  defaultObjectSizing,
  ShaderFitOptions,
  LiquidMetalShapes,
} from '@paper-design/shaders';

const DEFAULT_PARAMS = {
  colorBack: '#00000000', // transparent, so it sits on whatever is behind
  colorTint: '#f3d27a', // warm gold, matches the site accent
  contour: 0.45,
  distortion: 0.08,
  softness: 0.12,
  repetition: 2.2,
  shiftRed: 0.25,
  shiftBlue: 0.25,
  angle: 70,
  shape: 'none',
  scale: 0.92,
};

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Failed to load processed logo image'));
    img.src = src;
  });
}

// image: URL or data: URI for the logo (an inline SVG data URI works fine).
export async function mount(container, { image, speed = 1, minPixelRatio = 1, maxPixelCount = 1920 * 1080, params = {} } = {}) {
  const merged = { ...DEFAULT_PARAMS, ...params };
  const { pngBlob } = await toProcessedLiquidMetal(image);
  const objectUrl = URL.createObjectURL(pngBlob);
  const imageEl = await loadImage(objectUrl);

  const uniforms = {
    u_colorBack: getShaderColorFromString(merged.colorBack),
    u_colorTint: getShaderColorFromString(merged.colorTint),
    u_image: imageEl,
    u_contour: merged.contour,
    u_distortion: merged.distortion,
    u_softness: merged.softness,
    u_repetition: merged.repetition,
    u_shiftRed: merged.shiftRed,
    u_shiftBlue: merged.shiftBlue,
    u_angle: merged.angle,
    u_isImage: true,
    u_shape: LiquidMetalShapes[merged.shape],
    u_fit: ShaderFitOptions[defaultObjectSizing.fit],
    u_scale: merged.scale,
    u_rotation: defaultObjectSizing.rotation,
    u_offsetX: defaultObjectSizing.offsetX,
    u_offsetY: defaultObjectSizing.offsetY,
    u_originX: defaultObjectSizing.originX,
    u_originY: defaultObjectSizing.originY,
    u_worldWidth: defaultObjectSizing.worldWidth,
    u_worldHeight: defaultObjectSizing.worldHeight,
  };

  const shaderMount = new ShaderMount(
    container,
    liquidMetalFragmentShader,
    uniforms,
    undefined,
    speed,
    0,
    minPixelRatio,
    maxPixelCount,
    ['u_image']
  );

  return {
    destroy() {
      shaderMount.dispose();
      URL.revokeObjectURL(objectUrl);
    },
  };
}
