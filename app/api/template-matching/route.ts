import { NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';
import { spawn } from 'child_process';

const BASE_DIR = process.env.VERCEL ? '/tmp' : process.cwd();
const UPLOAD_DIR = path.join(BASE_DIR, 'module2', 'uploads');
const OUTPUT_DIR = path.join(BASE_DIR, 'module2', 'output');

type PythonResult = {
  stdout: string;
  stderr: string;
};

async function ensureDirectories() {
  await fs.mkdir(UPLOAD_DIR, { recursive: true });
  await fs.mkdir(OUTPUT_DIR, { recursive: true });
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
            MODULE2_OUTPUT_DIR: OUTPUT_DIR,
          },
        });
        let stdout = '';
        let stderr = '';

        child.stdout.on('data', (data) => {
          stdout += data.toString();
        });
        child.stderr.on('data', (data) => {
          stderr += data.toString();
        });
        child.on('error', (err) => reject(err));
        child.on('close', (code) => {
          if (code === 0) {
            resolve({ stdout, stderr });
          } else {
            const error = new Error(`Python exited with code ${code}: ${stderr}`);
            Object.assign(error, { stdout, stderr, code });
            reject(error);
          }
        });
      });
    } catch (error: any) {
      if (error?.code === 'ENOENT') {
        lastError = error;
        continue;
      }
      throw error;
    }
  }

  throw lastError ?? new Error('Unable to locate a working Python interpreter.');
}

export async function POST(request: Request) {
  try {
    await ensureDirectories();
    const formData = await request.formData();
    const file = formData.get('scene');
    if (!file || !(file instanceof Blob)) {
      return NextResponse.json({ error: 'Missing scene file upload.' }, { status: 400 });
    }

    // Get threshold from form data, default to 0.7
    const thresholdStr = formData.get('threshold');
    const threshold = thresholdStr ? parseFloat(thresholdStr.toString()) : 0.7;

    const fileName = typeof (file as any).name === 'string' ? (file as any).name : 'scene.png';
    const arrayBuffer = await file.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);
    const filenameSafe = fileName.replace(/\s+/g, '_').replace(/[^a-zA-Z0-9._-]/g, '');
    const uniqueName = `${Date.now()}_${filenameSafe || 'scene.png'}`;
    const savedPath = path.join(UPLOAD_DIR, uniqueName);
    await fs.writeFile(savedPath, buffer);

    // Optional label filter (comma or space separated)
    const onlyRaw = formData.get('only');
    const onlyLabels =
      typeof onlyRaw === 'string'
        ? onlyRaw
            .split(/[,\s]+/)
            .map((s) => s.trim())
            .filter(Boolean)
        : [];

    const args = ['-m', 'module2.run_template_matching', '--scene', savedPath, '--threshold', threshold.toString(), '--json'];
    if (onlyLabels.length) {
      args.push('--only', ...onlyLabels);
    }

    const { stdout } = await spawnPython(args);
    const parsed = JSON.parse(stdout);

    const annotatedPath = path.isAbsolute(parsed.annotated_image)
      ? parsed.annotated_image
      : path.join(process.cwd(), parsed.annotated_image);

    let annotatedDataUrl: string | null = null;
    try {
      const annotatedBuffer = await fs.readFile(annotatedPath);
      const base64 = annotatedBuffer.toString('base64');
      annotatedDataUrl = `data:image/png;base64,${base64}`;
    } catch (err) {
      annotatedDataUrl = null;
    }

    return NextResponse.json({
      detections: parsed.detections,
      threshold: parsed.threshold,
      annotatedImage: annotatedDataUrl,
      sourceScene: path.relative(process.cwd(), savedPath),
    });
  } catch (error: any) {
    console.error('[template-matching API]', error);
    return NextResponse.json(
      { error: error?.message ?? 'Unknown error running template detection.' },
      { status: 500 },
    );
  }
}
