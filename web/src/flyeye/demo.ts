// Manual test page for FlyEye. Optional query params: ?img=<url>&mix=<0..1>
import { FlyEye, FLY_EYE_LEGEND, flyEyeSupported } from './flyeye';

const canvas = document.getElementById('view') as HTMLCanvasElement;
const slider = document.getElementById('mix') as HTMLInputElement;
const file = document.getElementById('file') as HTMLInputElement;
document.getElementById('legend')!.textContent = flyEyeSupported()
  ? FLY_EYE_LEGEND
  : 'WebGL2 is not available in this browser.';

const eye = new FlyEye(canvas);
const params = new URLSearchParams(location.search);
if (params.has('mix')) slider.value = params.get('mix')!;

const draw = () => eye.render(Number(slider.value));
slider.addEventListener('input', draw);
new ResizeObserver(draw).observe(canvas);

async function load(blob: Blob) {
  eye.setImage(await createImageBitmap(blob));
  draw();
  document.body.dataset.ready = '1'; // signal for screenshot scripts
}

file.addEventListener('change', () => {
  const f = file.files?.[0];
  if (f) void load(f);
});

const img = params.get('img');
if (img) void fetch(img).then((r) => r.blob()).then(load);
