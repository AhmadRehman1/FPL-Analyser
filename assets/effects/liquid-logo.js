var ce=Object.defineProperty;var ue=(t,e,o)=>e in t?ce(t,e,{enumerable:!0,configurable:!0,writable:!0,value:o}):t[e]=o;var s=(t,e,o)=>ue(t,typeof e!="symbol"?e+"":e,o);var J=`#version 300 es
precision mediump float;

layout(location = 0) in vec4 a_position;

uniform vec2 u_resolution;
uniform float u_pixelRatio;
uniform float u_imageAspectRatio;
uniform float u_originX;
uniform float u_originY;
uniform float u_worldWidth;
uniform float u_worldHeight;
uniform float u_fit;
uniform float u_scale;
uniform float u_rotation;
uniform float u_offsetX;
uniform float u_offsetY;

out vec2 v_objectUV;
out vec2 v_objectBoxSize;
out vec2 v_responsiveUV;
out vec2 v_responsiveBoxGivenSize;
out vec2 v_patternUV;
out vec2 v_patternBoxSize;
out vec2 v_imageUV;

vec3 getBoxSize(float boxRatio, vec2 givenBoxSize) {
  vec2 box = vec2(0.);
  // fit = none
  box.x = boxRatio * min(givenBoxSize.x / boxRatio, givenBoxSize.y);
  float noFitBoxWidth = box.x;
  if (u_fit == 1.) { // fit = contain
    box.x = boxRatio * min(u_resolution.x / boxRatio, u_resolution.y);
  } else if (u_fit == 2.) { // fit = cover
    box.x = boxRatio * max(u_resolution.x / boxRatio, u_resolution.y);
  }
  box.y = box.x / boxRatio;
  return vec3(box, noFitBoxWidth);
}

void main() {
  gl_Position = a_position;

  vec2 uv = gl_Position.xy * .5;
  vec2 boxOrigin = vec2(.5 - u_originX, u_originY - .5);
  vec2 givenBoxSize = vec2(u_worldWidth, u_worldHeight);
  givenBoxSize = max(givenBoxSize, vec2(1.)) * u_pixelRatio;
  float r = u_rotation * 3.14159265358979323846 / 180.;
  mat2 graphicRotation = mat2(cos(r), sin(r), -sin(r), cos(r));
  vec2 graphicOffset = vec2(-u_offsetX, u_offsetY);


  // ===================================================

  float fixedRatio = 1.;
  vec2 fixedRatioBoxGivenSize = vec2(
  (u_worldWidth == 0.) ? u_resolution.x : givenBoxSize.x,
  (u_worldHeight == 0.) ? u_resolution.y : givenBoxSize.y
  );

  v_objectBoxSize = getBoxSize(fixedRatio, fixedRatioBoxGivenSize).xy;
  vec2 objectWorldScale = u_resolution.xy / v_objectBoxSize;

  v_objectUV = uv;
  v_objectUV *= objectWorldScale;
  v_objectUV += boxOrigin * (objectWorldScale - 1.);
  v_objectUV += graphicOffset;
  v_objectUV /= u_scale;
  v_objectUV = graphicRotation * v_objectUV;

  // ===================================================

  v_responsiveBoxGivenSize = vec2(
  (u_worldWidth == 0.) ? u_resolution.x : givenBoxSize.x,
  (u_worldHeight == 0.) ? u_resolution.y : givenBoxSize.y
  );
  float responsiveRatio = v_responsiveBoxGivenSize.x / v_responsiveBoxGivenSize.y;
  vec2 responsiveBoxSize = getBoxSize(responsiveRatio, v_responsiveBoxGivenSize).xy;
  vec2 responsiveBoxScale = u_resolution.xy / responsiveBoxSize;

  #ifdef ADD_HELPERS
  v_responsiveHelperBox = uv;
  v_responsiveHelperBox *= responsiveBoxScale;
  v_responsiveHelperBox += boxOrigin * (responsiveBoxScale - 1.);
  #endif

  v_responsiveUV = uv;
  v_responsiveUV *= responsiveBoxScale;
  v_responsiveUV += boxOrigin * (responsiveBoxScale - 1.);
  v_responsiveUV += graphicOffset;
  v_responsiveUV /= u_scale;
  v_responsiveUV.x *= responsiveRatio;
  v_responsiveUV = graphicRotation * v_responsiveUV;
  v_responsiveUV.x /= responsiveRatio;

  // ===================================================

  float patternBoxRatio = givenBoxSize.x / givenBoxSize.y;
  vec2 patternBoxGivenSize = vec2(
  (u_worldWidth == 0.) ? u_resolution.x : givenBoxSize.x,
  (u_worldHeight == 0.) ? u_resolution.y : givenBoxSize.y
  );
  patternBoxRatio = patternBoxGivenSize.x / patternBoxGivenSize.y;

  vec3 boxSizeData = getBoxSize(patternBoxRatio, patternBoxGivenSize);
  v_patternBoxSize = boxSizeData.xy;
  float patternBoxNoFitBoxWidth = boxSizeData.z;
  vec2 patternBoxScale = u_resolution.xy / v_patternBoxSize;

  v_patternUV = uv;
  v_patternUV += graphicOffset / patternBoxScale;
  v_patternUV += boxOrigin;
  v_patternUV -= boxOrigin / patternBoxScale;
  v_patternUV *= u_resolution.xy;
  v_patternUV /= u_pixelRatio;
  if (u_fit > 0.) {
    v_patternUV *= (patternBoxNoFitBoxWidth / v_patternBoxSize.x);
  }
  v_patternUV /= u_scale;
  v_patternUV = graphicRotation * v_patternUV;
  v_patternUV += boxOrigin / patternBoxScale;
  v_patternUV -= boxOrigin;
  // x100 is a default multiplier between vertex and fragmant shaders
  // we use it to avoid UV presision issues
  v_patternUV *= .01;

  // ===================================================

  vec2 imageBoxSize;
  if (u_fit == 1.) { // contain
    imageBoxSize.x = min(u_resolution.x / u_imageAspectRatio, u_resolution.y) * u_imageAspectRatio;
  } else if (u_fit == 2.) { // cover
    imageBoxSize.x = max(u_resolution.x / u_imageAspectRatio, u_resolution.y) * u_imageAspectRatio;
  } else {
    imageBoxSize.x = min(10.0, 10.0 / u_imageAspectRatio * u_imageAspectRatio);
  }
  imageBoxSize.y = imageBoxSize.x / u_imageAspectRatio;
  vec2 imageBoxScale = u_resolution.xy / imageBoxSize;

  v_imageUV = uv;
  v_imageUV *= imageBoxScale;
  v_imageUV += boxOrigin * (imageBoxScale - 1.);
  v_imageUV += graphicOffset;
  v_imageUV /= u_scale;
  v_imageUV.x *= u_imageAspectRatio;
  v_imageUV = graphicRotation * v_imageUV;
  v_imageUV.x /= u_imageAspectRatio;

  v_imageUV += .5;
  v_imageUV.y = 1. - v_imageUV.y;
}`;var ee=1920*1080*4,A=class{constructor(e,o,i,c,n=0,a=0,r=2,m=ee,u=[]){s(this,"parentElement");s(this,"canvasElement");s(this,"gl");s(this,"program",null);s(this,"uniformLocations",{});s(this,"fragmentShader");s(this,"rafId",null);s(this,"lastRenderTime",0);s(this,"currentFrame",0);s(this,"speed",0);s(this,"currentSpeed",0);s(this,"providedUniforms");s(this,"mipmaps",[]);s(this,"hasBeenDisposed",!1);s(this,"resolutionChanged",!0);s(this,"textures",new Map);s(this,"minPixelRatio");s(this,"maxPixelCount");s(this,"isSafari",me());s(this,"uniformCache",{});s(this,"textureUnitMap",new Map);s(this,"ownerDocument");s(this,"initProgram",()=>{let e=de(this.gl,J,this.fragmentShader);e&&(this.program=e)});s(this,"setupPositionAttribute",()=>{let e=this.gl.getAttribLocation(this.program,"a_position"),o=this.gl.createBuffer();this.gl.bindBuffer(this.gl.ARRAY_BUFFER,o);let i=[-1,-1,1,-1,-1,1,-1,1,1,-1,1,1];this.gl.bufferData(this.gl.ARRAY_BUFFER,new Float32Array(i),this.gl.STATIC_DRAW),this.gl.enableVertexAttribArray(e),this.gl.vertexAttribPointer(e,2,this.gl.FLOAT,!1,0,0)});s(this,"setupUniforms",()=>{let e={u_time:this.gl.getUniformLocation(this.program,"u_time"),u_pixelRatio:this.gl.getUniformLocation(this.program,"u_pixelRatio"),u_resolution:this.gl.getUniformLocation(this.program,"u_resolution")};Object.entries(this.providedUniforms).forEach(([o,i])=>{if(e[o]=this.gl.getUniformLocation(this.program,o),i instanceof HTMLImageElement){let c=`${o}AspectRatio`;e[c]=this.gl.getUniformLocation(this.program,c)}}),this.uniformLocations=e});s(this,"renderScale",1);s(this,"parentWidth",0);s(this,"parentHeight",0);s(this,"parentDevicePixelWidth",0);s(this,"parentDevicePixelHeight",0);s(this,"devicePixelsSupported",!1);s(this,"intersectionObserver",null);s(this,"isInViewport",!0);s(this,"resizeObserver",null);s(this,"setupResizeObserver",()=>{this.resizeObserver=new ResizeObserver(([e])=>{if(e?.borderBoxSize[0]){let o=e.devicePixelContentBoxSize?.[0];o!==void 0&&(this.devicePixelsSupported=!0,this.parentDevicePixelWidth=o.inlineSize,this.parentDevicePixelHeight=o.blockSize),this.parentWidth=e.borderBoxSize[0].inlineSize,this.parentHeight=e.borderBoxSize[0].blockSize}this.handleResize()}),this.resizeObserver.observe(this.parentElement)});s(this,"setupIntersectionObserver",()=>{let e=this.ownerDocument.defaultView;e?.IntersectionObserver&&(this.intersectionObserver=new e.IntersectionObserver(([o])=>{this.isInViewport=o?.isIntersecting??!0,this.updateCurrentSpeed()}),this.intersectionObserver.observe(this.parentElement))});s(this,"handleVisualViewportChange",()=>{this.resizeObserver?.disconnect(),this.setupResizeObserver()});s(this,"handleResize",()=>{let e=0,o=0,i=Math.max(1,window.devicePixelRatio),c=visualViewport?.scale??1;if(this.devicePixelsSupported){let f=Math.max(1,this.minPixelRatio/i);e=this.parentDevicePixelWidth*f*c,o=this.parentDevicePixelHeight*f*c}else{let f=Math.max(i,this.minPixelRatio)*c;if(this.isSafari){let h=pe(this.ownerDocument);f*=Math.max(1,h)}e=Math.round(this.parentWidth)*f,o=Math.round(this.parentHeight)*f}let n=Math.sqrt(this.maxPixelCount)/Math.sqrt(e*o),a=Math.min(1,n),r=Math.round(e*a),m=Math.round(o*a),u=r/Math.round(this.parentWidth);(this.canvasElement.width!==r||this.canvasElement.height!==m||this.renderScale!==u)&&(this.renderScale=u,this.canvasElement.width=r,this.canvasElement.height=m,this.resolutionChanged=!0,this.gl.viewport(0,0,this.gl.canvas.width,this.gl.canvas.height),this.render(performance.now()))});s(this,"render",e=>{if(this.hasBeenDisposed)return;if(this.program===null){console.warn("Tried to render before program or gl was initialized");return}let o=e-this.lastRenderTime;this.lastRenderTime=e,this.currentSpeed!==0&&(this.currentFrame+=o*this.currentSpeed),this.gl.clear(this.gl.COLOR_BUFFER_BIT),this.gl.useProgram(this.program),this.gl.uniform1f(this.uniformLocations.u_time,this.currentFrame*.001),this.resolutionChanged&&(this.gl.uniform2f(this.uniformLocations.u_resolution,this.gl.canvas.width,this.gl.canvas.height),this.gl.uniform1f(this.uniformLocations.u_pixelRatio,this.renderScale),this.resolutionChanged=!1),this.gl.drawArrays(this.gl.TRIANGLES,0,6),this.currentSpeed!==0?this.requestRender():this.rafId=null});s(this,"requestRender",()=>{this.rafId!==null&&cancelAnimationFrame(this.rafId),this.rafId=requestAnimationFrame(this.render)});s(this,"setTextureUniform",(e,o)=>{if(!o.complete||o.naturalWidth===0)throw new Error(`Paper Shaders: image for uniform ${e} must be fully loaded`);let i=this.textures.get(e);i&&this.gl.deleteTexture(i),this.textureUnitMap.has(e)||this.textureUnitMap.set(e,this.textureUnitMap.size);let c=this.textureUnitMap.get(e);this.gl.activeTexture(this.gl.TEXTURE0+c);let n=this.gl.createTexture();this.gl.bindTexture(this.gl.TEXTURE_2D,n),this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_WRAP_S,this.gl.CLAMP_TO_EDGE),this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_WRAP_T,this.gl.CLAMP_TO_EDGE),this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_MIN_FILTER,this.gl.LINEAR),this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_MAG_FILTER,this.gl.LINEAR),this.gl.texImage2D(this.gl.TEXTURE_2D,0,this.gl.RGBA,this.gl.RGBA,this.gl.UNSIGNED_BYTE,o),this.mipmaps.includes(e)&&(this.gl.generateMipmap(this.gl.TEXTURE_2D),this.gl.texParameteri(this.gl.TEXTURE_2D,this.gl.TEXTURE_MIN_FILTER,this.gl.LINEAR_MIPMAP_LINEAR));let a=this.gl.getError();if(a!==this.gl.NO_ERROR||n===null){console.error("Paper Shaders: WebGL error when uploading texture:",a);return}this.textures.set(e,n);let r=this.uniformLocations[e];if(r){this.gl.uniform1i(r,c);let m=`${e}AspectRatio`,u=this.uniformLocations[m];if(u){let f=o.naturalWidth/o.naturalHeight;this.gl.uniform1f(u,f)}}});s(this,"areUniformValuesEqual",(e,o)=>e===o?!0:Array.isArray(e)&&Array.isArray(o)&&e.length===o.length?e.every((i,c)=>this.areUniformValuesEqual(i,o[c])):!1);s(this,"setUniformValues",e=>{this.gl.useProgram(this.program),Object.entries(e).forEach(([o,i])=>{let c=i;if(i instanceof HTMLImageElement&&(c=`${i.src.slice(0,200)}|${i.naturalWidth}x${i.naturalHeight}`),this.areUniformValuesEqual(this.uniformCache[o],c))return;this.uniformCache[o]=c;let n=this.uniformLocations[o];if(!n){console.warn(`Uniform location for ${o} not found`);return}if(i instanceof HTMLImageElement)this.setTextureUniform(o,i);else if(Array.isArray(i)){let a=null,r=null;if(i[0]!==void 0&&Array.isArray(i[0])){let m=i[0].length;if(i.every(u=>u.length===m))a=i.flat(),r=m;else{console.warn(`All child arrays must be the same length for ${o}`);return}}else a=i,r=a.length;switch(r){case 2:this.gl.uniform2fv(n,a);break;case 3:this.gl.uniform3fv(n,a);break;case 4:this.gl.uniform4fv(n,a);break;case 9:this.gl.uniformMatrix3fv(n,!1,a);break;case 16:this.gl.uniformMatrix4fv(n,!1,a);break;default:console.warn(`Unsupported uniform array length: ${r}`)}}else typeof i=="number"?this.gl.uniform1f(n,i):typeof i=="boolean"?this.gl.uniform1i(n,i?1:0):console.warn(`Unsupported uniform type for ${o}: ${typeof i}`)})});s(this,"getCurrentFrame",()=>this.currentFrame);s(this,"setFrame",e=>{this.currentFrame=e,this.lastRenderTime=performance.now(),this.render(performance.now())});s(this,"setSpeed",(e=1)=>{this.speed=e,this.updateCurrentSpeed()});s(this,"updateCurrentSpeed",()=>{this.setCurrentSpeed(this.ownerDocument.hidden||!this.isInViewport?0:this.speed)});s(this,"setCurrentSpeed",e=>{this.currentSpeed=e,this.rafId===null&&e!==0&&(this.lastRenderTime=performance.now(),this.rafId=requestAnimationFrame(this.render)),this.rafId!==null&&e===0&&(cancelAnimationFrame(this.rafId),this.rafId=null)});s(this,"setMaxPixelCount",(e=ee)=>{this.maxPixelCount=e,this.handleResize()});s(this,"setMinPixelRatio",(e=2)=>{this.minPixelRatio=e,this.handleResize()});s(this,"setUniforms",e=>{this.setUniformValues(e),this.providedUniforms={...this.providedUniforms,...e},this.render(performance.now())});s(this,"handleDocumentVisibilityChange",()=>{this.updateCurrentSpeed()});s(this,"dispose",()=>{this.hasBeenDisposed=!0,this.rafId!==null&&(cancelAnimationFrame(this.rafId),this.rafId=null),this.gl&&this.program&&(this.textures.forEach(e=>{this.gl.deleteTexture(e)}),this.textures.clear(),this.gl.deleteProgram(this.program),this.program=null,this.gl.bindBuffer(this.gl.ARRAY_BUFFER,null),this.gl.bindBuffer(this.gl.ELEMENT_ARRAY_BUFFER,null),this.gl.bindRenderbuffer(this.gl.RENDERBUFFER,null),this.gl.bindFramebuffer(this.gl.FRAMEBUFFER,null),this.gl.getError()),this.resizeObserver&&(this.resizeObserver.disconnect(),this.resizeObserver=null),this.intersectionObserver&&(this.intersectionObserver.disconnect(),this.intersectionObserver=null),visualViewport?.removeEventListener("resize",this.handleVisualViewportChange),this.ownerDocument.removeEventListener("visibilitychange",this.handleDocumentVisibilityChange),this.uniformLocations={},this.canvasElement.remove(),delete this.parentElement.paperShaderMount});if(e?.nodeType===1)this.parentElement=e;else throw new Error("Paper Shaders: parent element must be an HTMLElement");if(this.ownerDocument=e.ownerDocument,!this.ownerDocument.querySelector("style[data-paper-shader]")){let p=this.ownerDocument.createElement("style");p.innerHTML=he,p.setAttribute("data-paper-shader",""),this.ownerDocument.head.prepend(p)}let f=this.ownerDocument.createElement("canvas");this.canvasElement=f,this.parentElement.prepend(f),this.fragmentShader=o,this.providedUniforms=i,this.mipmaps=u,this.currentFrame=a,this.minPixelRatio=r,this.maxPixelCount=m;let h=f.getContext("webgl2",c);if(!h)throw new Error("Paper Shaders: WebGL is not supported in this browser");this.gl=h,this.initProgram(),this.setupPositionAttribute(),this.setupUniforms(),this.setUniformValues(this.providedUniforms),this.setupResizeObserver(),visualViewport?.addEventListener("resize",this.handleVisualViewportChange),this.setupIntersectionObserver(),this.setSpeed(n),this.parentElement.setAttribute("data-paper-shader",""),this.parentElement.paperShaderMount=this,this.ownerDocument.addEventListener("visibilitychange",this.handleDocumentVisibilityChange)}};function te(t,e,o){let i=t.createShader(e);return i?(t.shaderSource(i,o),t.compileShader(i),t.getShaderParameter(i,t.COMPILE_STATUS)?i:(console.error("An error occurred compiling the shaders: "+t.getShaderInfoLog(i)),t.deleteShader(i),null)):null}function de(t,e,o){let i=t.getShaderPrecisionFormat(t.FRAGMENT_SHADER,t.MEDIUM_FLOAT),c=i?i.precision:null;c&&c<23&&(e=e.replace(/precision\s+(lowp|mediump)\s+float;/g,"precision highp float;"),o=o.replace(/precision\s+(lowp|mediump)\s+float/g,"precision highp float").replace(/\b(uniform|varying|attribute)\s+(lowp|mediump)\s+(\w+)/g,"$1 highp $3"));let n=te(t,t.VERTEX_SHADER,e),a=te(t,t.FRAGMENT_SHADER,o);if(!n||!a)return null;let r=t.createProgram();return r?(t.attachShader(r,n),t.attachShader(r,a),t.linkProgram(r),t.getProgramParameter(r,t.LINK_STATUS)?(t.detachShader(r,n),t.detachShader(r,a),t.deleteShader(n),t.deleteShader(a),r):(console.error("Unable to initialize the shader program: "+t.getProgramInfoLog(r)),t.deleteProgram(r),t.deleteShader(n),t.deleteShader(a),null)):null}var he=`@layer paper-shaders {
  :where([data-paper-shader]) {
    isolation: isolate;
    position: relative;

    & canvas {
      contain: strict;
      display: block;
      position: absolute;
      inset: 0;
      z-index: -1;
      width: 100%;
      height: 100%;
      border-radius: inherit;
      corner-shape: inherit;
    }
  }
}`;function me(){let t=navigator.userAgent.toLowerCase();return t.includes("safari")&&!t.includes("chrome")&&!t.includes("android")}function pe(t){let e=visualViewport?.scale??1,o=visualViewport?.width??window.innerWidth,i=window.innerWidth-t.documentElement.clientWidth,c=e*o+i,n=outerWidth/c,a=Math.round(100*n);return a%5===0?a/100:a===33?1/3:a===67?2/3:a===133?4/3:n}var U={fit:"contain",scale:1,rotation:0,offsetX:0,offsetY:0,originX:.5,originY:.5,worldWidth:0,worldHeight:0};var N={none:0,contain:1,cover:2};var oe=`
#define TWO_PI 6.28318530718
#define PI 3.14159265358979323846
`,ie=`
vec2 rotate(vec2 uv, float th) {
  return mat2(cos(th), sin(th), -sin(th), cos(th)) * uv;
}
`;var re=`
  color += 1. / 256. * (fract(sin(dot(.014 * gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453123) - .5);
`,se=`
vec3 permute(vec3 x) { return mod(((x * 34.0) + 1.0) * x, 289.0); }
float snoise(vec2 v) {
  const vec4 C = vec4(0.211324865405187, 0.366025403784439,
    -0.577350269189626, 0.024390243902439);
  vec2 i = floor(v + dot(v, C.yy));
  vec2 x0 = v - i + dot(i, C.xx);
  vec2 i1;
  i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;
  i = mod(i, 289.0);
  vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0))
    + i.x + vec3(0.0, i1.x, 1.0));
  vec3 m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy),
      dot(x12.zw, x12.zw)), 0.0);
  m = m * m;
  m = m * m;
  vec3 x = 2.0 * fract(p * C.www) - 1.0;
  vec3 h = abs(x) - 0.5;
  vec3 ox = floor(x + 0.5);
  vec3 a0 = x - ox;
  m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);
  vec3 g;
  g.x = a0.x * x0.x + h.x * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130.0 * dot(m, g);
}
`;var j=`#version 300 es
precision mediump float;

uniform sampler2D u_image;

uniform vec2 u_resolution;
uniform float u_time;

uniform vec4 u_colorBack;
uniform vec4 u_colorTint;

uniform float u_softness;
uniform float u_repetition;
uniform float u_shiftRed;
uniform float u_shiftBlue;
uniform float u_distortion;
uniform float u_contour;
uniform float u_angle;

uniform float u_shape;
uniform bool u_isImage;

in vec2 v_objectUV;
in vec2 v_responsiveUV;
in vec2 v_responsiveBoxGivenSize;
in vec2 v_imageUV;

out vec4 fragColor;

${oe}
${ie}
${se}

float getColorChanges(float c1, float c2, float stripe_p, vec3 w, float blur, float bump, float tint) {

  float ch = mix(c2, c1, smoothstep(.0, 2. * blur, stripe_p));

  float border = w[0];
  ch = mix(ch, c2, smoothstep(border, border + 2. * blur, stripe_p));

  if (u_isImage == true) {
    bump = smoothstep(.2, .8, bump);
  }
  border = w[0] + .4 * (1. - bump) * w[1];
  ch = mix(ch, c1, smoothstep(border, border + 2. * blur, stripe_p));

  border = w[0] + .5 * (1. - bump) * w[1];
  ch = mix(ch, c2, smoothstep(border, border + 2. * blur, stripe_p));

  border = w[0] + w[1];
  ch = mix(ch, c1, smoothstep(border, border + 2. * blur, stripe_p));

  float gradient_t = (stripe_p - w[0] - w[1]) / w[2];
  float gradient = mix(c1, c2, smoothstep(0., 1., gradient_t));
  ch = mix(ch, gradient, smoothstep(border, border + .5 * blur, stripe_p));

  // Tint color is applied with color burn blending
  ch = mix(ch, 1. - min(1., (1. - ch) / max(tint, 0.0001)), u_colorTint.a);
  return ch;
}

float getImgFrame(vec2 uv, float th) {
  float frame = 1.;
  frame *= smoothstep(0., th, uv.y);
  frame *= 1.0 - smoothstep(1. - th, 1., uv.y);
  frame *= smoothstep(0., th, uv.x);
  frame *= 1.0 - smoothstep(1. - th, 1., uv.x);
  return frame;
}

float blurEdge3x3(sampler2D tex, vec2 uv, vec2 dudx, vec2 dudy, float radius, float centerSample) {
  vec2 texel = 1.0 / vec2(textureSize(tex, 0));
  vec2 r = radius * texel;

  float w1 = 1.0, w2 = 2.0, w4 = 4.0;
  float norm = 16.0;
  float sum = w4 * centerSample;

  sum += w2 * textureGrad(tex, uv + vec2(0.0, -r.y), dudx, dudy).r;
  sum += w2 * textureGrad(tex, uv + vec2(0.0, r.y), dudx, dudy).r;
  sum += w2 * textureGrad(tex, uv + vec2(-r.x, 0.0), dudx, dudy).r;
  sum += w2 * textureGrad(tex, uv + vec2(r.x, 0.0), dudx, dudy).r;

  sum += w1 * textureGrad(tex, uv + vec2(-r.x, -r.y), dudx, dudy).r;
  sum += w1 * textureGrad(tex, uv + vec2(r.x, -r.y), dudx, dudy).r;
  sum += w1 * textureGrad(tex, uv + vec2(-r.x, r.y), dudx, dudy).r;
  sum += w1 * textureGrad(tex, uv + vec2(r.x, r.y), dudx, dudy).r;

  return sum / norm;
}

float lst(float edge0, float edge1, float x) {
  return clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
}

void main() {

  const float firstFrameOffset = 2.8;
  float t = .3 * (u_time + firstFrameOffset);

  vec2 uv = v_imageUV;
  vec2 dudx = dFdx(v_imageUV);
  vec2 dudy = dFdy(v_imageUV);
  vec4 img = textureGrad(u_image, uv, dudx, dudy);

  if (u_isImage == false) {
    uv = v_objectUV + .5;
    uv.y = 1. - uv.y;
  }

  float cycleWidth = u_repetition;
  float edge = 0.;
  float contOffset = 1.;

  vec2 rotatedUV = uv - vec2(.5);
  float angle = (-u_angle + 70.) * PI / 180.;
  float cosA = cos(angle);
  float sinA = sin(angle);
  rotatedUV = vec2(
  rotatedUV.x * cosA - rotatedUV.y * sinA,
  rotatedUV.x * sinA + rotatedUV.y * cosA
  ) + vec2(.5);

  // u_contour is applied in 2 separate ranges:
  // - 0 to .4 sets the edge hardness, saturated above .4 (both branches below)
  // - .5 to 1 warps the stripes direction along the edges, inactive below .5 (see u_contour range 2)
  if (u_isImage == true) {
    float edgeRaw = img.r;
    edge = blurEdge3x3(u_image, uv, dudx, dudy, 6., edgeRaw);
    edge = pow(edge, 1.6);
    edge *= mix(0.0, 1.0, smoothstep(0.0, 0.4, u_contour));
  } else {
    if (u_shape < 1.) {
      // full-fill on canvas
      vec2 borderUV = v_responsiveUV + .5;
      float ratio = v_responsiveBoxGivenSize.x / v_responsiveBoxGivenSize.y;
      vec2 mask = min(borderUV, 1. - borderUV);
      vec2 pixel_thickness = min(250. / v_responsiveBoxGivenSize, vec2(.5));
      float maskX = smoothstep(0.0, pixel_thickness.x, mask.x);
      float maskY = smoothstep(0.0, pixel_thickness.y, mask.y);
      maskX = pow(maskX, .25);
      maskY = pow(maskY, .25);
      edge = clamp(1. - maskX * maskY, 0., 1.);

      uv = v_responsiveUV;
      if (ratio > 1.) {
        uv.y /= ratio;
      } else {
        uv.x *= ratio;
      }
      uv += .5;
      uv.y = 1. - uv.y;

      cycleWidth *= 2.;
      contOffset = 1.5;

    } else if (u_shape < 2.) {
      // circle
      vec2 shapeUV = uv - .5;
      shapeUV *= .67;
      edge = pow(clamp(3. * length(shapeUV), 0., 1.), 18.);
    } else if (u_shape < 3.) {
      // daisy
      vec2 shapeUV = uv - .5;
      shapeUV *= 1.68;

      float r = length(shapeUV) * 2.;
      float a = atan(shapeUV.y, shapeUV.x) + .2;
      r *= (1. + .05 * sin(3. * a + 2. * t));
      float f = abs(cos(a * 3.));
      edge = smoothstep(f, f + .7, r);
      edge *= edge;

      uv *= .8;
      cycleWidth *= 1.6;

    } else if (u_shape < 4.) {
      // diamond
      vec2 shapeUV = uv - .5;
      shapeUV = rotate(shapeUV, .25 * PI);
      shapeUV *= 1.42;
      shapeUV += .5;
      vec2 mask = min(shapeUV, 1. - shapeUV);
      vec2 pixel_thickness = vec2(.15);
      float maskX = smoothstep(0.0, pixel_thickness.x, mask.x);
      float maskY = smoothstep(0.0, pixel_thickness.y, mask.y);
      maskX = pow(maskX, .25);
      maskY = pow(maskY, .25);
      edge = clamp(1. - maskX * maskY, 0., 1.);
    } else if (u_shape < 5.) {
      // metaballs
      vec2 shapeUV = uv - .5;
      shapeUV *= 1.3;
      edge = 0.;
      for (int i = 0; i < 5; i++) {
        float fi = float(i);
        float speed = 1.5 + 2./3. * sin(fi * 12.345);
        float angle = -fi * 1.5;
        vec2 dir1 = vec2(cos(angle), sin(angle));
        vec2 dir2 = vec2(cos(angle + 1.57), sin(angle + 1.));
        vec2 traj = .4 * (dir1 * sin(t * speed + fi * 1.23) + dir2 * cos(t * (speed * 0.7) + fi * 2.17));
        float d = length(shapeUV + traj);
        edge += pow(1.0 - clamp(d, 0.0, 1.0), 4.0);
      }
      edge = 1. - smoothstep(.65, .9, edge);
      edge = pow(edge, 4.);
    }

    edge = mix(smoothstep(.9 - 2. * fwidth(edge), .9, edge), edge, smoothstep(0.0, 0.4, u_contour));

  }

  float opacity = 0.;
  if (u_isImage == true) {
    opacity = img.g;
    float frame = getImgFrame(v_imageUV, 0.);
    opacity *= frame;
  } else {
    opacity = 1. - smoothstep(.9 - 2. * fwidth(edge), .9, edge);
    if (u_shape < 2.) {
      edge = 1.2 * edge;
    } else if (u_shape < 5.) {
      edge = 1.8 * pow(edge, 1.5);
    }
  }

  float diagBLtoTR = rotatedUV.x - rotatedUV.y;
  float diagTLtoBR = rotatedUV.x + rotatedUV.y;

  vec3 color = vec3(0.);
  vec3 color1 = vec3(.98, 0.98, 1.);
  vec3 color2 = vec3(.1, .1, .1 + .1 * smoothstep(.7, 1.3, diagTLtoBR));

  vec2 grad_uv = uv - .5;

  float dist = length(grad_uv + vec2(0., .2 * diagBLtoTR));
  grad_uv = rotate(grad_uv, (.25 - .2 * diagBLtoTR) * PI);
  float direction = grad_uv.x;

  float bump = pow(1.8 * dist, 1.2);
  bump = 1. - bump;
  bump *= pow(uv.y, .3);


  float thin_strip_1_ratio = .12 / cycleWidth * (1. - .4 * bump);
  float thin_strip_2_ratio = .07 / cycleWidth * (1. + .4 * bump);
  float wide_strip_ratio = (1. - thin_strip_1_ratio - thin_strip_2_ratio);

  float thin_strip_1_width = cycleWidth * thin_strip_1_ratio;
  float thin_strip_2_width = cycleWidth * thin_strip_2_ratio;

  float noise = snoise(uv - t);

  edge += (1. - edge) * u_distortion * noise;

  direction += diagBLtoTR;
  float contour = 0.;
  direction -= 2. * noise * diagBLtoTR * (smoothstep(0., 1., edge) * (1.0 - smoothstep(0., 1., edge)));
  // u_contour range 2
  direction *= mix(1., 1. - edge, smoothstep(.5, 1., u_contour));
  direction -= 1.7 * edge * smoothstep(.5, 1., u_contour);
  direction += .2 * pow(u_contour, 4.) * (1.0 - smoothstep(0., 1., edge));

  bump *= clamp(pow(uv.y, .1), .3, 1.);
  direction *= (.1 + (1.1 - edge) * bump);

  direction *= (.4 + .6 * (1.0 - smoothstep(.5, 1., edge)));
  direction += .18 * (smoothstep(.1, .2, uv.y) * (1.0 - smoothstep(.2, .4, uv.y)));
  direction += .03 * (smoothstep(.1, .2, 1. - uv.y) * (1.0 - smoothstep(.2, .4, 1. - uv.y)));

  direction *= (.5 + .5 * pow(uv.y, 2.));
  direction *= cycleWidth;
  direction -= t;


  float colorDispersion = (1. - bump);
  colorDispersion = clamp(colorDispersion, 0., 1.);
  float dispersionRed = colorDispersion;
  dispersionRed += .03 * bump * noise;
  dispersionRed += 5. * (smoothstep(-.1, .2, uv.y) * (1.0 - smoothstep(.1, .5, uv.y))) * (smoothstep(.4, .6, bump) * (1.0 - smoothstep(.4, 1., bump)));
  dispersionRed -= diagBLtoTR;

  float dispersionBlue = colorDispersion;
  dispersionBlue *= 1.3;
  dispersionBlue += (smoothstep(0., .4, uv.y) * (1.0 - smoothstep(.1, .8, uv.y))) * (smoothstep(.4, .6, bump) * (1.0 - smoothstep(.4, .8, bump)));
  dispersionBlue -= .2 * edge;

  dispersionRed *= (u_shiftRed / 20.);
  dispersionBlue *= (u_shiftBlue / 20.);

  float blur = 0.;
  float rExtraBlur = 0.;
  float gExtraBlur = 0.;
  if (u_isImage == true) {
    float softness = 0.05 * u_softness;
    blur = softness + .5 * smoothstep(1., 10., u_repetition) * smoothstep(.0, 1., edge);
    float smallCanvasT = 1.0 - smoothstep(100., 500., min(u_resolution.x, u_resolution.y));
    blur += smallCanvasT * smoothstep(.0, 1., edge);
    rExtraBlur = softness * (0.05 + .1 * (u_shiftRed / 20.) * bump);
    gExtraBlur = softness * 0.05 / max(0.001, abs(1. - diagBLtoTR));
  } else {
    blur = u_softness / 15. + .3 * contour;
  }

  vec3 w = vec3(thin_strip_1_width, thin_strip_2_width, wide_strip_ratio);
  w[1] -= .02 * smoothstep(.0, 1., edge + bump);
  float stripe_r = fract(direction + dispersionRed);
  float r = getColorChanges(color1.r, color2.r, stripe_r, w, blur + fwidth(stripe_r) + rExtraBlur, bump, u_colorTint.r);
  float stripe_g = fract(direction);
  float g = getColorChanges(color1.g, color2.g, stripe_g, w, blur + fwidth(stripe_g) + gExtraBlur, bump, u_colorTint.g);
  float stripe_b = fract(direction - dispersionBlue);
  float b = getColorChanges(color1.b, color2.b, stripe_b, w, blur + fwidth(stripe_b), bump, u_colorTint.b);

  color = vec3(r, g, b);
  color *= opacity;

  vec3 bgColor = u_colorBack.rgb * u_colorBack.a;
  color = color + bgColor * (1. - opacity);
  opacity = opacity + u_colorBack.a * (1. - opacity);

  ${re}

  fragColor = vec4(color, opacity);
}
`,z={measurePerformance:!1,workingSize:512,iterations:40};function Y(t){let e=document.createElement("canvas"),o=e.getContext("2d"),i=typeof t=="string"&&t.startsWith("blob:");return new Promise((c,n)=>{if(!t||!o){n(new Error("Invalid file or canvas context"));return}let a=i&&fetch(t).then(u=>u.headers.get("Content-Type")),r=new Image;r.crossOrigin="anonymous";let m=performance.now();r.onload=async()=>{let u,f=await a;f?u=f==="image/svg+xml":typeof t=="string"?u=t.endsWith(".svg")||t.startsWith("data:image/svg+xml"):u=t.type==="image/svg+xml";let h=r.width||r.naturalWidth,p=r.height||r.naturalHeight;if(u){let g=h/p;h>p?(h=4096,p=4096/g):(p=4096,h=4096*g),r.width=h,r.height=p}let T=Math.min(h,p),x=z.workingSize/T,l=Math.round(h*x),v=Math.round(p*x);z.measurePerformance&&(console.log("[Processing Mode]"),console.log(`  Original: ${h}\xD7${p}`),console.log(`  Working: ${l}\xD7${v} (${(x*100).toFixed(1)}% scale)`),x<1&&console.log(`  Speedup: ~${Math.round(1/(x*x))}\xD7`)),e.width=h,e.height=p;let R=document.createElement("canvas");R.width=l,R.height=v;let V=R.getContext("2d");V.drawImage(r,0,0,l,v);let F=performance.now(),w=V.getImageData(0,0,l,v).data,b=new Uint8Array(l*v),L=new Uint8Array(l*v),G=0;for(let d=0,g=0;d<w.length;d+=4,g++){let S=w[d+3]===0?0:1;b[g]=S,G+=S}let W=[],M=[];for(let d=0;d<v;d++)for(let g=0;g<l;g++){let _=d*l+g;if(!b[_])continue;let S=!1;g===0||g===l-1||d===0||d===v-1?S=!0:S=!b[_-1]||!b[_+1]||!b[_-l]||!b[_+l]||!b[_-l-1]||!b[_-l+1]||!b[_+l-1]||!b[_+l+1],S?(L[_]=1,W.push(_)):M.push(_)}z.measurePerformance&&(console.log(`[Mask Building] Time: ${(performance.now()-F).toFixed(2)}ms`),console.log(`  Shape pixels: ${G} / ${l*v} (${(G/(l*v)*100).toFixed(1)}%)`),console.log(`  Interior pixels: ${M.length}`),console.log(`  Boundary pixels: ${W.length}`));let ae=fe(b,L,new Uint32Array(M),new Uint32Array(W),l,v),ne=performance.now(),k=ge(ae,b,L,l,v);z.measurePerformance&&console.log(`[Poisson Solve] Time: ${(performance.now()-ne).toFixed(2)}ms`);let $=0,Z;for(let d=0;d<M.length;d++){let g=M[d];k[g]>$&&($=k[g])}let D=document.createElement("canvas");D.width=l,D.height=v;let K=D.getContext("2d"),E=K.createImageData(l,v);for(let d=0;d<v;d++)for(let g=0;g<l;g++){let _=d*l+g,S=_*4;if(!b[_])E.data[S]=255,E.data[S+1]=255,E.data[S+2]=255,E.data[S+3]=0;else{let X=255*(1-k[_]/$);E.data[S]=X,E.data[S+1]=X,E.data[S+2]=X,E.data[S+3]=255}}K.putImageData(E,0,0),o.imageSmoothingEnabled=!0,o.imageSmoothingQuality="high",o.drawImage(D,0,0,l,v,0,0,h,p);let y=o.getImageData(0,0,h,p),H=document.createElement("canvas");H.width=h,H.height=p;let Q=H.getContext("2d");Q.drawImage(r,0,0,h,p);let le=Q.getImageData(0,0,h,p);for(let d=0;d<y.data.length;d+=4){let g=le.data[d+3],_=y.data[d+3];g===0?(y.data[d]=255,y.data[d+1]=0):(y.data[d]=_===0?0:y.data[d],y.data[d+1]=g),y.data[d+2]=255,y.data[d+3]=255}o.putImageData(y,0,0),Z=y,e.toBlob(d=>{if(!d){n(new Error("Failed to create PNG blob"));return}if(z.measurePerformance){let g=performance.now()-m;if(console.log(`[Total Processing Time] ${g.toFixed(2)}ms`),x<1){let _=g*Math.pow(h*p/(l*v),1.5);console.log(`[Estimated time at full resolution] ~${_.toFixed(0)}ms`),console.log(`[Time saved] ~${(_-g).toFixed(0)}ms (${Math.round(_/g)}\xD7 faster)`)}}c({imageData:Z,pngBlob:d})},"image/png")},r.onerror=()=>n(new Error("Failed to load image")),r.src=typeof t=="string"?t:URL.createObjectURL(t)})}function fe(t,e,o,i,c,n){let a=o.length,r=new Int32Array(a*4);for(let m=0;m<a;m++){let u=o[m],f=u%c,h=Math.floor(u/c);r[m*4+0]=f<c-1&&t[u+1]?u+1:-1,r[m*4+1]=f>0&&t[u-1]?u-1:-1,r[m*4+2]=h>0&&t[u-c]?u-c:-1,r[m*4+3]=h<n-1&&t[u+c]?u+c:-1}return{interiorPixels:o,boundaryPixels:i,pixelCount:a,neighborIndices:r}}function ge(t,e,o,i,c){let n=z.iterations,a=.01,r=new Float32Array(i*c),{interiorPixels:m,neighborIndices:u,pixelCount:f}=t,h=performance.now(),p=1.9,T=[],B=[];for(let x=0;x<f;x++){let l=m[x],v=l%i,R=Math.floor(l/i);(v+R)%2===0?T.push(x):B.push(x)}for(let x=0;x<n;x++){for(let l of T){let v=m[l],R=u[l*4+0],V=u[l*4+1],F=u[l*4+2],I=u[l*4+3],w=0;R>=0&&(w+=r[R]),V>=0&&(w+=r[V]),F>=0&&(w+=r[F]),I>=0&&(w+=r[I]);let b=(a+w)/4;r[v]=p*b+(1-p)*r[v]}for(let l of B){let v=m[l],R=u[l*4+0],V=u[l*4+1],F=u[l*4+2],I=u[l*4+3],w=0;R>=0&&(w+=r[R]),V>=0&&(w+=r[V]),F>=0&&(w+=r[F]),I>=0&&(w+=r[I]);let b=(a+w)/4;r[v]=p*b+(1-p)*r[v]}}if(z.measurePerformance){let x=performance.now()-h;console.log(`[Optimized Poisson Solver (SOR \u03C9=${p})]`),console.log(`  Working size: ${i}\xD7${c}`),console.log(`  Iterations: ${n}`),console.log(`  Time: ${x.toFixed(2)}ms`),console.log(`  Interior pixels processed: ${f}`),console.log(`  Speed: ${(n*f/(x*1e3)).toFixed(2)} Mpixels/sec`)}return r}var q={none:0,circle:1,daisy:2,diamond:3,metaballs:4};function O(t){if(Array.isArray(t))return t.length===4?t:t.length===3?[...t,1]:P;if(typeof t!="string")return P;let e,o,i,c=1;if(t.startsWith("#"))[e,o,i,c]=ve(t);else if(t.startsWith("rgb")){let n=xe(t);if(n===null)return P;[e,o,i,c]=n}else if(t.startsWith("hsl")){let n=_e(t);if(n===null)return P;[e,o,i,c]=be(n)}else return console.error("Unsupported color format",t),P;return[C(e,0,1),C(o,0,1),C(i,0,1),C(c,0,1)]}function ve(t){if(t=t.replace(/^#/,""),(t.length===3||t.length===4)&&(t=t.split("").map(n=>n+n).join("")),t.length===6&&(t=t+"ff"),!/^[0-9a-f]{8}$/i.test(t))return console.warn("Invalid hex color"),P;let e=parseInt(t.slice(0,2),16)/255,o=parseInt(t.slice(2,4),16)/255,i=parseInt(t.slice(4,6),16)/255,c=parseInt(t.slice(6,8),16)/255;return[e,o,i,c]}function xe(t){let e=t.match(/^rgba?\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*([0-9.]+))?\s*\)$/i);return e?[parseInt(e[1]??"0")/255,parseInt(e[2]??"0")/255,parseInt(e[3]??"0")/255,e[4]===void 0?1:parseFloat(e[4])]:null}function _e(t){let e=t.match(/^hsla?\s*\(\s*(\d+)\s*,\s*(\d+)%\s*,\s*(\d+)%\s*(?:,\s*([0-9.]+))?\s*\)$/i);return e?[parseInt(e[1]??"0"),parseInt(e[2]??"0"),parseInt(e[3]??"0"),e[4]===void 0?1:parseFloat(e[4])]:null}function be(t){let[e,o,i,c]=t,n=e/360,a=o/100,r=i/100,m,u,f;if(o===0)m=u=f=r;else{let h=(B,x,l)=>(l<0&&(l+=1),l>1&&(l-=1),l<.16666666666666666?B+(x-B)*6*l:l<.5?x:l<.6666666666666666?B+(x-B)*(.6666666666666666-l)*6:B),p=r<.5?r*(1+a):r+a-r*a,T=2*r-p;m=h(T,p,n+1/3),u=h(T,p,n),f=h(T,p,n-1/3)}return[m,u,f,c]}var C=(t,e,o)=>Math.min(Math.max(t,e),o),P=[.5,.5,.5,1];var Se={colorBack:"#00000000",colorTint:"#f3d27a",contour:.45,distortion:.08,softness:.12,repetition:2.2,shiftRed:.25,shiftBlue:.25,angle:70,shape:"none",scale:.92};function we(t){return new Promise((e,o)=>{let i=new Image;i.onload=()=>e(i),i.onerror=()=>o(new Error("Failed to load processed logo image")),i.src=t})}async function ke(t,{image:e,speed:o=1,minPixelRatio:i=1,maxPixelCount:c=1920*1080,params:n={}}={}){let a={...Se,...n},{pngBlob:r}=await Y(e),m=URL.createObjectURL(r),u=await we(m),f={u_colorBack:O(a.colorBack),u_colorTint:O(a.colorTint),u_image:u,u_contour:a.contour,u_distortion:a.distortion,u_softness:a.softness,u_repetition:a.repetition,u_shiftRed:a.shiftRed,u_shiftBlue:a.shiftBlue,u_angle:a.angle,u_isImage:!0,u_shape:q[a.shape],u_fit:N[U.fit],u_scale:a.scale,u_rotation:U.rotation,u_offsetX:U.offsetX,u_offsetY:U.offsetY,u_originX:U.originX,u_originY:U.originY,u_worldWidth:U.worldWidth,u_worldHeight:U.worldHeight},h=new A(t,j,f,void 0,o,0,i,c,["u_image"]);return{destroy(){h.dispose(),URL.revokeObjectURL(m)}}}export{ke as mount};
