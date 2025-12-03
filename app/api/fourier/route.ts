import { NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';
import { spawn } from 'child_process';

const UPLOAD_DIR = path.join(process.cwd(), 'module2', 'uploads');
const OUTPUT_DIR = path.join(process.cwd(), 'module2', 'output');

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
            PYTHONPATH: [process.cwd(), process.env.PYTHONPATH].filter(Boolean).join(path.delimiter),
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

async function fileToDataUrl(imagePath: string): Promise<string | null> {
  try {
    const buffer = await fs.readFile(imagePath);
    const base64 = buffer.toString('base64');
    return `data:image/png;base64,${base64}`;
  } catch (error) {
    console.error('[fourier API] Failed to read image at', imagePath, error);
    return null;
  }
}

export async function POST(request: Request) {
  try {
    await ensureDirectories();
    const formData = await request.formData();
    const file = formData.get('image');
    if (!file || !(file instanceof Blob)) {
      return NextResponse.json({ error: 'Missing image upload.' }, { status: 400 });
    }

    const fileName = typeof (file as any).name === 'string' ? (file as any).name : 'input.png';
    const arrayBuffer = await file.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);
    const filenameSafe = fileName.replace(/\s+/g, '_').replace(/[^a-zA-Z0-9._-]/g, '');
    const uniqueName = `${Date.now()}_${filenameSafe || 'input.png'}`;
    const savedPath = path.join(UPLOAD_DIR, uniqueName);
    await fs.writeFile(savedPath, buffer);

    const args = [
      '-m',
      'module2.fourier_deblur',
      '--input',
      savedPath,
      '--output-dir',
      OUTPUT_DIR,
      '--json',
    ];

    const { stdout } = await spawnPython(args);
    const parsed = JSON.parse(stdout);

    const resolvePath = (p: string) => (path.isAbsolute(p) ? p : path.join(process.cwd(), p));

    const blurPath = resolvePath(parsed.blur_path);
    const restorePath = resolvePath(parsed.restore_path);
    const montagePath = resolvePath(parsed.montage_path);

    const [blurImage, restoredImage, montageImage] = await Promise.all([
      fileToDataUrl(blurPath),
      fileToDataUrl(restorePath),
      fileToDataUrl(montagePath),
    ]);

    return NextResponse.json({
      blurImage,
      restoredImage,
      montageImage,
      psnrBlur: parsed.psnr_blur,
      psnrRestore: parsed.psnr_restore,
      sourceImage: path.relative(process.cwd(), savedPath),
    });
  } catch (error: any) {
    console.error('[fourier API]', error);
    return NextResponse.json(
      { error: error?.message ?? 'Unknown error running Fourier experiment.' },
      { status: 500 },
    );
  }
}
