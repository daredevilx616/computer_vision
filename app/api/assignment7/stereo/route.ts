import { NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';
import { spawn } from 'child_process';

const MODULE_UPLOAD = path.join(process.cwd(), 'module7', 'uploads');

type PythonResult = { stdout: string; stderr: string };

async function ensureDirs() {
  await fs.mkdir(MODULE_UPLOAD, { recursive: true });
}

async function spawnPython(args: string[]): Promise<PythonResult> {
  const candidates = [
    process.env.PYTHON_BIN,
    process.env.VIRTUAL_ENV ? path.join(process.env.VIRTUAL_ENV, 'Scripts', 'python.exe') : undefined,
    'python',
    'python3',
    'py',
  ].filter(Boolean) as string[];
  let lastError: unknown = null;
  for (const command of candidates) {
    try {
      return await new Promise<PythonResult>((resolve, reject) => {
        const child = spawn(command, args, { cwd: process.cwd() });
        let stdout = '';
        let stderr = '';
        child.stdout.on('data', (chunk) => (stdout += chunk.toString()));
        child.stderr.on('data', (chunk) => (stderr += chunk.toString()));
        child.on('close', (code) => {
          if (code === 0) resolve({ stdout, stderr });
          else reject(new Error(stderr || `Python exited with code ${code}`));
        });
        child.on('error', reject);
      });
    } catch (error: any) {
      if (error?.code === 'ENOENT') {
        lastError = error;
        continue;
      }
      throw error;
    }
  }
  throw lastError ?? new Error('Unable to find python interpreter.');
}

function sanitize(name: string | null): string {
  if (!name) return 'image.png';
  return name.replace(/\s+/g, '_').replace(/[^a-zA-Z0-9._-]/g, '');
}

async function saveBlob(blob: Blob, fallback: string): Promise<string> {
  const buffer = Buffer.from(await blob.arrayBuffer());
  const filename = `${Date.now()}_${sanitize((blob as any).name ?? fallback)}`;
  const filePath = path.join(MODULE_UPLOAD, filename);
  await fs.writeFile(filePath, buffer);
  return filePath;
}

export async function POST(request: Request) {
  try {
    await ensureDirs();
    const formData = await request.formData();
    const left = formData.get('left');
    const right = formData.get('right');
    if (!(left instanceof Blob) || !(right instanceof Blob)) {
      return NextResponse.json({ error: 'Left and right images are required.' }, { status: 400 });
    }
    const polygonRaw = formData.get('polygon');
    if (typeof polygonRaw !== 'string' || !polygonRaw.trim()) {
      return NextResponse.json({ error: 'Polygon vertices are required.' }, { status: 400 });
    }
    const polygon: Array<{ x: number; y: number }> = JSON.parse(polygonRaw);
    if (polygon.length < 2) {
      return NextResponse.json({ error: 'Polygon requires at least two vertices.' }, { status: 400 });
    }
    const focal = Number(formData.get('focalMm'));
    const sensorWidth = Number(formData.get('sensorWidthMm'));
    const baseline = Number(formData.get('baselineMm'));
    if (!isFinite(focal) || !isFinite(sensorWidth) || !isFinite(baseline)) {
      return NextResponse.json({ error: 'Focal length, sensor width, and baseline must be numbers.' }, { status: 400 });
    }

    const leftPath = await saveBlob(left, 'left.png');
    const rightPath = await saveBlob(right, 'right.png');
    const polygonSpec = polygon.map((pt) => `${pt.x}:${pt.y}`).join(',');
    const args = [
      '-m',
      'module7.cli',
      '--left',
      leftPath,
      '--right',
      rightPath,
      '--polygon',
      polygonSpec,
      '--focal-mm',
      focal.toString(),
      '--sensor-width-mm',
      sensorWidth.toString(),
      '--baseline-mm',
      baseline.toString(),
    ];
    const { stdout } = await spawnPython(args);
    const payload = JSON.parse(stdout);
    return NextResponse.json(payload);
  } catch (error: any) {
    console.error('[assignment7/stereo]', error);
    return NextResponse.json({ error: error?.message ?? 'Unexpected error' }, { status: 500 });
  }
}

