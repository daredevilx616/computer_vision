import { NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';
import { spawn } from 'child_process';

const BASE_DIR = process.env.VERCEL ? '/tmp' : process.cwd();
const MODULE_UPLOAD = path.join(BASE_DIR, 'module4', 'uploads');

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
        const child = spawn(command, args, {
          cwd: process.cwd(),
          env: {
            ...process.env,
          },
        });
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
  const buf = Buffer.from(await blob.arrayBuffer());
  const filename = `${Date.now()}_${sanitize((blob as any).name ?? fallback)}`;
  const filePath = path.join(MODULE_UPLOAD, filename);
  await fs.writeFile(filePath, buf);
  return filePath;
}

async function fileToDataUrl(filePath: string): Promise<string> {
  const buffer = await fs.readFile(filePath);
  const base64 = buffer.toString('base64');
  return `data:image/png;base64,${base64}`;
}

export async function POST(request: Request) {
  try {
    await ensureDirs();
    const formData = await request.formData();
    const imageA = formData.get('imageA');
    const imageB = formData.get('imageB');
    if (!(imageA instanceof Blob) || !(imageB instanceof Blob)) {
      return NextResponse.json({ error: 'Two image uploads are required.' }, { status: 400 });
    }

    const pathA = await saveBlob(imageA, 'a.png');
    const pathB = await saveBlob(imageB, 'b.png');
    const args = ['-m', 'module4.cli', 'sift', '--image-a', pathA, '--image-b', pathB];
    const { stdout } = await spawnPython(args);
    const payload = JSON.parse(stdout);

    // Convert custom SIFT visual path to data URL
    if (payload.visual_path) {
      const absolute = path.isAbsolute(payload.visual_path)
        ? payload.visual_path
        : path.join(process.cwd(), 'module4', payload.visual_path);
      payload.visual = await fileToDataUrl(absolute);
    }

    // Convert OpenCV SIFT visual path to data URL
    if (payload.cv_visual_path) {
      const absolute = path.isAbsolute(payload.cv_visual_path)
        ? payload.cv_visual_path
        : path.join(process.cwd(), 'module4', payload.cv_visual_path);
      payload.cv_visual = await fileToDataUrl(absolute);
    }

    return NextResponse.json(payload);
  } catch (error: any) {
    console.error('[assignment4/sift]', error);
    return NextResponse.json({ error: error?.message ?? 'Unexpected error.' }, { status: 500 });
  }
}
